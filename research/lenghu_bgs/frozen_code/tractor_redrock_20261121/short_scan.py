"""Resumable short-exposure follow-up after the original scan has finished."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import run_scan as base
from simulate_tractor import DATA, EXT, HERE, HIGH, POP, ROOT, SHARED, sha

SHORT = HERE / "short_run"
RUN = SHORT / "run"
PILOT = SHORT / "pilot"
SIM = HERE / "simulate_short.py"
TIMES = [125, 250, 500]
SKIES = ["dark", "bright"]
PLAN = base.PLAN


def paths(start):
    d = PILOT if start == 0 else RUN
    return d / f"test_{start:04d}_spectra.fits", d / f"test_{start:04d}_redrock"


def validate_spectra(start, count, sample):
    p, _ = paths(start)
    truthfile = p.with_name(p.stem + "_truth.csv")
    maskfile = p.with_name(p.stem + "_masks.csv")
    manifestfile = p.with_name(p.stem + "_manifest.json")
    if not all(x.is_file() for x in (p, truthfile, maskfile, manifestfile)):
        raise ValueError(f"incomplete short spectra batch {start}")
    m = json.loads(manifestfile.read_text())
    expected = {
        "spectra_sha256": p, "truth_sha256": truthfile, "mask_sha256": maskfile,
        "frozen_design_sha256": DATA / "frozen_design.json", "resolution_sha256": base.LSF,
        "instrument_sha256": SHARED / "instrument_reference_4200.npz",
        "camera_mapping_sha256": SHARED / "desi_camera_mapping.npz",
        "simulation_script_sha256": POP / "simulate_batch.py", "short_simulator_sha256": SIM,
        "tractor_simulator_sha256": HERE / "simulate_tractor.py",
        "tractor_adapter_sha256": HIGH / "tractor_adapter.py",
        "tractor_coefficients_reader_sha256": HIGH / "compare_tractor.py",
        "tractor_sersic_reference_sha256": HIGH / "reference/tractor_dr9_4/sersic.py",
        "tractor_mixture_reference_sha256": HIGH / "reference/tractor_dr9_4/mixture_profiles.py",
        "mixture_module_sha256": ROOT / "py/just_etc/mixture_fiber_coupling.py",
    }
    for key, path in expected.items():
        if m[key] != sha(path):
            raise ValueError(f"short batch {start} provenance mismatch: {key}")
    if (m["selection_start"] != start or m["galaxies"] != count or m["split"] != "test" or
            m["times_s"] != TIMES or m["sky_cases"] != SKIES or m["spectra_rows"] != 6 * count or
            m["short_times_exposure_s"] != {str(t): [t] for t in TIMES}):
        raise ValueError(f"short batch {start} scenario plan mismatch")
    truth = pd.read_csv(truthfile)
    subset = sample.iloc[start:start + count]
    roster = {(int(tid), sky, t) for tid in subset.TARGETID for sky in SKIES for t in TIMES}
    actual = set(zip(truth.TARGETID.astype(int), truth.sky, truth.total_s.astype(int)))
    if len(truth) != 6 * count or len(actual) != len(truth) or actual != roster or not truth.SYNTHETIC_TARGETID.is_unique:
        raise ValueError(f"short batch {start} target coverage mismatch")
    matched = truth.merge(subset[["TARGETID", "VI_Z", "source_row"]], on="TARGETID", validate="many_to_one", suffixes=("", "_catalog"))
    np.testing.assert_array_equal(matched.source_row, matched.source_row_catalog)
    np.testing.assert_allclose(matched.z_true_vi, matched.VI_Z, rtol=0, atol=1e-12)
    if not np.array_equal(matched.exposures_s.astype(str), matched.total_s.astype(str)):
        raise ValueError("short exposures must each have exactly one read")
    with fits.open(p, memmap=True) as hdus:
        np.testing.assert_array_equal(hdus["FIBERMAP"].data["TARGETID"], truth.SYNTHETIC_TARGETID)
    return p


def main():
    # Never run competing Redrock fits or mutate the original scan.
    old_progress = json.loads((HERE / "run/progress.json").read_text())
    if old_progress["status"] != "complete":
        raise RuntimeError("finish the main Redrock scan before the short follow-up")
    sample = base.preflight()
    pilot = json.loads((PILOT / "input_validation.json").read_text())
    if pilot["status"] != "passed" or pilot["short_spectra_sha256"] != sha(paths(0)[0]):
        raise ValueError("short-exposure pilot input validation is missing or stale")
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / ".run.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.RUN, base.paths = RUN, paths
    plan = {"n_test": len(sample), "times_s": TIMES, "skies": SKIES, "redrock_processes": 2,
            "batches": [{"start": i, "galaxies": n} for i, n in PLAN],
            "sample_sha256": sha(DATA / "frozen_sample.csv"),
            "design_sha256": sha(DATA / "frozen_design.json"), "lsf_sha256": sha(base.LSF),
            "simulation_wrapper_sha256": sha(SIM), "fit_wrapper_sha256": sha(base.FIT),
            "pilot_input_validation_sha256": sha(PILOT / "input_validation.json"),
            "driver_sha256": sha(Path(__file__)), "parent_driver_sha256": sha(HERE / "run_scan.py"),
            "main_run_plan_sha256": sha(HERE / "run/run_plan.json"),
            "single_exposure_per_time": True,
            "synthetic_id_scope": "unique within batch, join runs by TARGETID, sky, total_s"}
    planfile = RUN / "run_plan.json"
    if planfile.exists() and json.loads(planfile.read_text()) != plan:
        raise ValueError("short run plan changed; refusing to overwrite")
    if not planfile.exists():
        planfile.write_text(json.dumps(plan, indent=2) + "\n")
    done = []
    try:
        for start, count in PLAN:
            base.write_progress("running", done, start)
            spectra, prefix = paths(start)
            input_files = [spectra, spectra.with_name(spectra.stem + "_truth.csv"),
                           spectra.with_name(spectra.stem + "_masks.csv"),
                           spectra.with_name(spectra.stem + "_manifest.json")]
            base.preserve_partial(input_files, start, "spectra")
            if not spectra.exists():
                if start == 0:
                    raise ValueError("verified short pilot spectra are missing")
                base.run_command([str(base.HOST), "-s", "-B", "-u", str(SIM), "--split", "test",
                                  "--start", str(start), "--count", str(count), "--times", *map(str, TIMES),
                                  "--resolution-file", str(base.LSF), "--output", str(spectra)],
                                 RUN / f"test_{start:04d}_simulation.log")
            validate_spectra(start, count, sample)
            output_files = [prefix.with_suffix(s) for s in (".json", ".csv", ".fits")]
            base.preserve_partial(output_files, start, "redrock")
            if not any(p.exists() for p in output_files):
                base.run_command([str(base.HOST), "-s", "-B", "-u", str(base.FIT), "--spectra", str(spectra),
                                  "--output-prefix", str(prefix), "--processes", "2"],
                                 RUN / f"test_{start:04d}_runner.log")
            meta = base.validate_fit(start, spectra)
            done.append({"start": start, "galaxies": count, "spectra": meta["attempted"],
                         "returned": meta["returned"], "elapsed_seconds": meta["elapsed_seconds"]})
            base.write_progress("running", done)
            print(f"Finished short batch {start}: {len(done)}/{len(PLAN)}", flush=True)
        base.write_progress("complete", done)
        print("Completed the short-exposure follow-up.", flush=True)
    except Exception as exc:
        base.write_progress("failed", done, start, str(exc))
        raise


if __name__ == "__main__":
    main()
