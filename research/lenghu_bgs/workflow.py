"""Portable entry points for the frozen Lenghu BGS simulation and Redrock scan.

The numerical simulator and the Redrock cache wrapper are archived unchanged in
``frozen_code``. The driver constructs a temporary historical directory layout
from explicitly supplied data and JUST ETC source paths. It never writes into
the historical data tree. See README.md for the scientific scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen_code"
STUDY_REL = Path("doc/validation/lenghu_redshift_recovery_20261111")
EXT = Path("leader_195_205_20261116/extended_20p5")
POP = Path("population_4200_camera_overlap")
HIGH = Path("high_sersic_legacy_20261121")
TRACTOR = Path("tractor_redrock_20261121")
SHORT = {125, 250, 500}
LONG = {750, 1500, 2250, 3000, 3600}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_code():
    """Check that every staged script still has its historical byte hash."""
    manifest = json.loads((HERE / "frozen_code_manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        path = FROZEN / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen code changed or is missing: {relative}")
    return sha(HERE / "frozen_code_manifest.json")


def _link(source, destination):
    source = Path(source).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare_layout(work_root, data_root, etc_package):
    """Create a disposable layout, copying code and linking read-only inputs.

    ``data_root`` is the archived *study directory*, not its parent. The three
    historical absolute links under extended_20p5/shared are reconstructed from
    the canonical population output files and are never modified in place.
    """
    verify_frozen_code()
    root = Path(work_root).resolve()
    data_root = Path(data_root).resolve(strict=True)
    etc_package = Path(etc_package).resolve(strict=True)
    if etc_package.name != "just_etc" or not (etc_package / "mixture_fiber_coupling.py").is_file():
        raise ValueError("--etc-package must be a just_etc source package directory")
    study = root / STUDY_REL
    study.mkdir(parents=True)
    _link(etc_package, root / "py/just_etc")
    manifest = json.loads((HERE / "frozen_code_manifest.json").read_text())
    for relative in manifest["files_sha256"]:
        dst = study / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FROZEN / relative, dst)
    direct = [
        "sky", "pilot_4200/extinction", str(EXT / "input"),
        str(HIGH / "mixture_verification.json"),
        "sersic_fiber_model_20261121/fiber_coupling_per_target.csv",
        "sersic_fiber_model_20261121/summary.json",
        str(POP / "output/redrock_cache_validation.json"),
    ]
    for name in direct:
        _link(data_root / name, study / name)
    shared = study / EXT / "shared"
    _link(data_root / EXT / "shared/desi_camera_mapping.json", shared / "desi_camera_mapping.json")
    for name in ("desi_camera_mapping.npz", "instrument_reference_4200.npz", "instrument_validation.json"):
        _link(data_root / POP / "output" / name, shared / name)
    return root, study


def _output_paths(path, suffixes):
    path = Path(path).resolve()
    return [path.with_name(path.stem + suffix) for suffix in suffixes]


def _check_output_outside_input(output, data_root, candidates):
    output = Path(output).resolve()
    if output.is_relative_to(Path(data_root).resolve()):
        raise ValueError("Output must be outside the archived historical data directory")
    if any(path.exists() or path.is_symlink() for path in candidates):
        raise FileExistsError("Refusing to overwrite an existing or partial output")
    output.parent.mkdir(parents=True, exist_ok=True)


def _run(script, arguments, layout_root, extra_environment=None):
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env["PYTHONPATH"] = str(layout_root / "py") + os.pathsep + env.get("PYTHONPATH", "")
    if extra_environment:
        env.update(extra_environment)
    command = [sys.executable, "-s", "-B", str(script), *map(str, arguments)]
    subprocess.run(command, cwd=layout_root, env=env, check=True)


def simulate(args):
    times = set(args.times)
    if not times or len(times) != len(args.times) or not (times <= SHORT or times <= LONG):
        raise ValueError("Choose unique times wholly within 125/250/500 or 750/1500/2250/3000/3600 s")
    output = Path(args.output).resolve()
    candidates = [output, *_output_paths(output, ["_truth.csv", "_masks.csv", "_manifest.json"])]
    _check_output_outside_input(output, args.data_root, candidates)
    if args.count < 1 or args.start < 0:
        raise ValueError("start must be nonnegative and count positive")
    with tempfile.TemporaryDirectory(prefix="just-bgs-layout-", dir=output.parent) as temp:
        root, study = prepare_layout(temp, args.data_root, args.etc_package)
        name = "simulate_short.py" if times <= SHORT else "simulate_tractor.py"
        arguments = ["--split", args.split, "--start", args.start, "--count", args.count,
                     "--times", *args.times, "--resolution-file", study / EXT / "input/combined_resolution_nodes.npz",
                     "--output", output]
        _run(study / TRACTOR / name, arguments, root)
    print(f"Simulation complete: {output}")


def verify_templates(template_dir, data_root):
    template_dir = Path(template_dir).resolve(strict=True)
    record = json.loads((Path(data_root) / "just_desi_550s_20261115/stable_environment.json").read_text())
    for name, expected in record["templates"].items():
        path = template_dir / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Historical Redrock template missing or changed: {name}")
    return template_dir


def fit(args):
    spectra = Path(args.spectra).resolve(strict=True)
    prefix = Path(args.output_prefix).resolve()
    candidates = [prefix.with_suffix(s) for s in (".fits", ".csv", ".json", ".log")]
    _check_output_outside_input(prefix, args.data_root, candidates)
    if args.processes < 1:
        raise ValueError("processes must be positive")
    templates = verify_templates(args.template_dir, args.data_root)
    with tempfile.TemporaryDirectory(prefix="just-bgs-layout-", dir=prefix.parent) as temp:
        root, study = prepare_layout(temp, args.data_root, args.etc_package)
        arguments = ["--spectra", spectra, "--output-prefix", prefix, "--processes", args.processes]
        _run(study / POP / "fit_batch_cached.py", arguments, root, {"RR_TEMPLATE_DIR": str(templates)})
    print(f"Redrock fit complete: {prefix}")


def summarize(args):
    """Summarize all attempted targets, including failed and missing fits."""
    import numpy as np
    import pandas as pd
    from scipy.stats import beta

    results = pd.read_csv(args.results)
    sample = pd.read_csv(Path(args.data_root) / EXT / "input/frozen_sample.csv")
    fields = ["TARGETID", "sky", "total_s", "z_fit", "z_true_vi", "ZWARN", "DELTACHI2"]
    if not set(fields).issubset(results.columns) or results.duplicated(["TARGETID", "sky", "total_s"]).any():
        raise ValueError("Result table lacks required fields or contains duplicate cases")
    if sample.TARGETID.duplicated().any():
        raise ValueError("Frozen sample has duplicate target IDs")
    rows = results.merge(sample[["TARGETID", "r_dered"]], on="TARGETID", how="left", validate="many_to_one")
    if rows.r_dered.isna().any():
        raise ValueError("Some result targets are missing from the frozen sample")
    if args.scan_count < 1 or not 0 < args.confidence < 1:
        raise ValueError("Invalid confidence or number of scanned times")
    if rows.groupby("sky").total_s.nunique().max() > args.scan_count:
        raise ValueError("scan-count cannot be smaller than the number of supplied times in one sky")
    alpha = (1.0 - args.confidence) / args.scan_count
    output = []
    for limit in (19.5, 20.5):
        group = rows[rows.r_dered < limit]
        for (sky, total), cases in group.groupby(["sky", "total_s"], sort=True):
            redshift = pd.to_numeric(cases.z_fit, errors="coerce").to_numpy(float)
            truth = pd.to_numeric(cases.z_true_vi, errors="coerce").to_numpy(float)
            zwarn = pd.to_numeric(cases.ZWARN, errors="coerce").to_numpy(float)
            delta = pd.to_numeric(cases.DELTACHI2, errors="coerce").to_numpy(float)
            returned = np.isfinite(redshift)
            quality = returned & (zwarn == 0) & (delta > 40)
            success = quality & np.isfinite(truth) & (np.abs(redshift - truth) / (1 + truth) < .0033)
            n, k, q = len(cases), int(success.sum()), int(quality.sum())
            output.append({"r_dered_lt": limit, "sky": sky, "total_s": int(total), "attempted": n,
                           "returned": int(returned.sum()), "quality_pass": q, "success": k,
                           "success_fraction": k / n, "purity": k / q if q else np.nan,
                           "one_sided_lower": 0. if k == 0 else float(beta.ppf(alpha, k, n-k+1)),
                           "confidence": args.confidence, "scan_count": args.scan_count})
    result = pd.DataFrame(output)
    if result.empty:
        raise ValueError("No cases for the requested magnitude limits")
    if args.output:
        path = Path(args.output).resolve()
        _check_output_outside_input(path, args.data_root, [path])
        result.to_csv(path, index=False)
    print(result.to_string(index=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("simulate", "fit", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--data-root", type=Path, required=True, help="Archived Lenghu study directory")
        if name != "summary":
            s.add_argument("--etc-package", type=Path, required=True, help="Path to just_etc package directory")
        if name == "simulate":
            s.add_argument("--split", choices=("test", "calibration"), default="test")
            s.add_argument("--start", type=int, default=0)
            s.add_argument("--count", type=int, required=True)
            s.add_argument("--times", type=int, nargs="+", required=True)
            s.add_argument("--output", type=Path, required=True)
            s.set_defaults(func=simulate)
        elif name == "fit":
            s.add_argument("--spectra", type=Path, required=True)
            s.add_argument("--template-dir", type=Path, required=True)
            s.add_argument("--output-prefix", type=Path, required=True)
            s.add_argument("--processes", type=int, default=2)
            s.set_defaults(func=fit)
        else:
            s.add_argument("--results", type=Path, required=True)
            s.add_argument("--scan-count", type=int, required=True)
            s.add_argument("--confidence", type=float, default=.95)
            s.add_argument("--output", type=Path)
            s.set_defaults(func=summarize)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
