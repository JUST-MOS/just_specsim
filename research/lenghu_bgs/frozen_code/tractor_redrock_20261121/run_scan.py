"""Resumable five-time bright/dark Redrock scan with Tractor fiber coupling.

Run only after the paired five-galaxy pilot passes. This driver never modifies
the historical scan products or shared frozen source/instrument inputs.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits

from simulate_tractor import DATA, EXT, HERE, HIGH, POP, ROOT, SHARED, sha

RUN = HERE / "run"
HOST = Path("/Users/rain/miniconda3/envs/desi/bin/python")
SIM = HERE / "simulate_tractor.py"
PRIOR = HERE.parent / "just_desi_550s_20261115"
FIT = PRIOR / "fit_stable.py"
LSF = DATA / "combined_resolution_nodes.npz"
TIMES = [750, 1500, 2250, 3000, 3600]
SKIES = ["dark", "bright"]
PLAN = [(0, 5)] + [(start, min(50, 1332 - start)) for start in range(5, 1332, 50)]


def paths(start):
    directory = HERE / "pilot" if start == 0 else RUN
    return directory / f"test_{start:04d}_spectra.fits", directory / f"test_{start:04d}_redrock"


def preflight():
    design = json.loads((DATA / "frozen_design.json").read_text())
    resolution = json.loads((DATA / "combined_resolution_manifest.json").read_text())
    instrument = json.loads((SHARED / "instrument_validation.json").read_text())
    camera = json.loads((SHARED / "desi_camera_mapping.json").read_text())
    profile = json.loads((HIGH / "mixture_verification.json").read_text())
    pilot = json.loads((HERE / "pilot/input_validation.json").read_text())
    stable = json.loads((PRIOR / "stable_runtime_validation.json").read_text())
    checks = {
        DATA / "frozen_sample.csv": design["sample_sha256"],
        DATA / "normalized_sources.fits": design["normalized_sources_sha256"],
        LSF: resolution["combined_nodes_sha256"],
        SHARED / "instrument_reference_4200.npz": instrument["instrument_sha256"],
        SHARED / "desi_camera_mapping.npz": camera["mapping_sha256"],
        PRIOR / "stable_environment.json": stable["environment_sha256"],
        POP / "rrdesi_015_cached_spawn.py": stable["wrapper_sha256"],
        HERE / "pilot/test_0000_spectra.fits": pilot["improved_pilot_sha256"],
    }
    for path, expected in checks.items():
        if sha(path) != expected:
            raise ValueError("frozen or pilot input changed: " + str(path))
    for relative, expected in profile["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("mixture verification dependency changed: " + relative)
    if any(record["status"] != "passed" for record in (resolution, instrument, camera, profile, pilot, stable)):
        raise ValueError("one or more input/pilot/runtime validations failed")
    if not Path(stable["python"]).is_file():
        raise ValueError("validated Redrock runtime is unavailable")
    sample = pd.read_csv(DATA / "frozen_sample.csv")
    selected = sample[sample.split == "test"].reset_index(drop=True)
    if (len(selected) != 1332 or len(selected[selected.r_dered < 19.5]) != 578 or
        not selected.TARGETID.is_unique or sum(count for _, count in PLAN) != len(selected) or
        design["noise_seed_version"] != "sha256(20261114:TARGETID:sky:total_seconds:arm), first 16 hex digits"):
        raise ValueError("frozen sample or fixed noise rule differs from expected")
    return selected


def validate_spectra(start, count, selected):
    p, _ = paths(start)
    truthfile = p.with_name(p.stem + "_truth.csv")
    maskfile = p.with_name(p.stem + "_masks.csv")
    manifestfile = p.with_name(p.stem + "_manifest.json")
    if not all(x.is_file() for x in (p, truthfile, maskfile, manifestfile)):
        raise ValueError(f"incomplete simulated batch {start}")
    m = json.loads(manifestfile.read_text())
    expected_hashes = {
        "spectra_sha256": p, "truth_sha256": truthfile, "mask_sha256": maskfile,
        "frozen_design_sha256": DATA / "frozen_design.json", "resolution_sha256": LSF,
        "instrument_sha256": SHARED / "instrument_reference_4200.npz",
        "camera_mapping_sha256": SHARED / "desi_camera_mapping.npz",
        "simulation_script_sha256": POP / "simulate_batch.py", "tractor_simulator_sha256": SIM,
        "tractor_adapter_sha256": HIGH / "tractor_adapter.py",
        "tractor_coefficients_reader_sha256": HIGH / "compare_tractor.py",
        "tractor_sersic_reference_sha256": HIGH / "reference/tractor_dr9_4/sersic.py",
        "tractor_mixture_reference_sha256": HIGH / "reference/tractor_dr9_4/mixture_profiles.py",
        "mixture_module_sha256": ROOT / "py/just_etc/mixture_fiber_coupling.py",
    }
    for key, path in expected_hashes.items():
        if m[key] != sha(path):
            raise ValueError(f"batch {start} provenance mismatch: {key}")
    if (m["selection_start"] != start or m["galaxies"] != count or m["split"] != "test" or
        m["times_s"] != TIMES or m["sky_cases"] != SKIES or m["spectra_rows"] != 10 * count):
        raise ValueError(f"batch {start} scenario plan mismatch")
    truth = pd.read_csv(truthfile)
    subset = selected.iloc[start:start + count]
    expected = {(int(tid), sky, t) for tid in subset.TARGETID for sky in SKIES for t in TIMES}
    actual = set(zip(truth.TARGETID.astype(int), truth.sky, truth.total_s.astype(int)))
    if len(truth) != 10 * count or len(actual) != len(truth) or actual != expected or not truth.SYNTHETIC_TARGETID.is_unique:
        raise ValueError(f"batch {start} target coverage mismatch")
    joined = truth.merge(subset[["TARGETID", "VI_Z", "source_row"]], on="TARGETID", validate="many_to_one", suffixes=("", "_catalog"))
    np.testing.assert_array_equal(joined.source_row, joined.source_row_catalog)
    np.testing.assert_allclose(joined.z_true_vi, joined.VI_Z, atol=1e-12, rtol=0)
    with fits.open(p, memmap=True) as hdus:
        np.testing.assert_array_equal(hdus["FIBERMAP"].data["TARGETID"], truth.SYNTHETIC_TARGETID)
    return p


def validate_fit(start, spectra):
    _, prefix = paths(start)
    meta_file, csvfile, fitsfile = [prefix.with_suffix(suffix) for suffix in (".json", ".csv", ".fits")]
    if not all(p.is_file() for p in (meta_file, csvfile, fitsfile)):
        raise ValueError(f"incomplete Redrock batch {start}")
    meta = json.loads(meta_file.read_text())
    if (meta["child_exit_code"] != 0 or meta["redrock_version"] != "0.15.4" or
        meta["spectra_sha256"] != sha(spectra) or meta["results_sha256"] != sha(csvfile) or
        meta["output_sha256"] != sha(fitsfile) or meta["stable_fit_script_sha256"] != sha(FIT) or
        meta["stable_validation_sha256"] != sha(PRIOR / "stable_runtime_validation.json")):
        raise ValueError(f"Redrock provenance mismatch in batch {start}")
    truth = pd.read_csv(spectra.with_name(spectra.stem + "_truth.csv"))
    rows = pd.read_csv(csvfile)
    if len(rows) != len(truth) or not rows.SYNTHETIC_TARGETID.is_unique or set(rows.SYNTHETIC_TARGETID) != set(truth.SYNTHETIC_TARGETID):
        raise ValueError(f"Redrock denominator mismatch in batch {start}")
    rows = truth.merge(rows, on="SYNTHETIC_TARGETID", validate="one_to_one", suffixes=("_truth", ""))
    for column in ("TARGETID", "source_row", "sky", "total_s", "exposures_s"):
        np.testing.assert_array_equal(rows[column], rows[column + "_truth"])
    with fits.open(fitsfile, memmap=False) as h:
        record = h["REDSHIFTS"].data
        ids = np.asarray(record["TARGETID"], dtype=np.int64)
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(truth.SYNTHETIC_TARGETID)):
            raise ValueError(f"Redrock FITS identifiers mismatch in batch {start}")
        matched = rows.set_index("SYNTHETIC_TARGETID").loc[ids]
        for actual, saved in (("Z", "z_fit"), ("ZERR", "zerr"), ("DELTACHI2", "DELTACHI2")):
            np.testing.assert_allclose(np.asarray(record[actual], dtype=float), matched[saved], rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(record["ZWARN"], matched.ZWARN)
    returned = np.isfinite(rows.z_fit)
    quality = returned & (rows.ZWARN == 0) & (rows.DELTACHI2 > 40)
    correct = returned & (np.abs(rows.z_fit - rows.z_true_vi) / (1 + rows.z_true_vi) < .0033)
    for expected, saved in ((returned, rows.returned), (quality, rows.quality_pass),
                            (correct, rows.truth_pass), (quality & correct, rows.success)):
        np.testing.assert_array_equal(expected, saved)
    if meta["attempted"] != len(rows) or meta["returned"] != int(returned.sum()):
        raise ValueError(f"Redrock attempted/returned counts mismatch in batch {start}")
    return meta


def write_progress(status, done, start=None, detail=None):
    payload = {"status": status, "completed": done, "current_start": start,
               "updated_utc": datetime.now(timezone.utc).isoformat(), "detail": detail}
    temporary = RUN / ".progress.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, RUN / "progress.json")


def run_command(command, logfile, timeout=3600):
    print("Running", " ".join(map(str, command)), flush=True)
    with logfile.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = proc.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started > timeout:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=30)
                        raise TimeoutError(f"batch timed out: {logfile}")
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            raise
    if code:
        raise RuntimeError(f"batch exited {code}; inspect {logfile}")


def preserve_partial(paths_to_check, start, kind):
    """Move incomplete outputs aside so a restarted batch can be regenerated."""
    existing = [p for p in paths_to_check if p.exists()]
    if not existing or len(existing) == len(paths_to_check):
        return
    if start == 0:
        raise ValueError("incomplete pilot output requires inspection")
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = RUN / "interrupted" / f"{kind}_{start:04d}_{name}"
    destination.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(destination / path.name))
    print(f"Preserved incomplete {kind} batch {start} in {destination}", flush=True)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / ".run.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = preflight()
    plan = {"n_test": len(selected), "times_s": TIMES, "skies": SKIES, "redrock_processes": 2,
            "batches": [{"start": start, "galaxies": count} for start, count in PLAN],
            "sample_sha256": sha(DATA / "frozen_sample.csv"), "design_sha256": sha(DATA / "frozen_design.json"),
            "lsf_sha256": sha(LSF), "simulation_wrapper_sha256": sha(SIM), "fit_wrapper_sha256": sha(FIT),
            "pilot_input_validation_sha256": sha(HERE / "pilot/input_validation.json"),
            "driver_sha256": sha(Path(__file__)), "reference_population": "frozen DESI selected field"}
    planfile = RUN / "run_plan.json"
    if planfile.exists() and json.loads(planfile.read_text()) != plan:
        raise ValueError("existing run plan differs; refusing to overwrite it")
    if not planfile.exists():
        planfile.write_text(json.dumps(plan, indent=2) + "\n")
    done = []
    try:
        for start, count in PLAN:
            write_progress("running", done, start)
            spectra, prefix = paths(start)
            input_files = [spectra, spectra.with_name(spectra.stem + "_truth.csv"),
                           spectra.with_name(spectra.stem + "_masks.csv"),
                           spectra.with_name(spectra.stem + "_manifest.json")]
            preserve_partial(input_files, start, "spectra")
            if not spectra.exists():
                if start == 0:
                    raise ValueError("validated pilot input is missing")
                run_command([str(HOST), "-s", "-B", "-u", str(SIM), "--split", "test", "--start", str(start),
                             "--count", str(count), "--times", *map(str, TIMES), "--resolution-file", str(LSF),
                             "--output", str(spectra)], RUN / f"test_{start:04d}_simulation.log")
            spectra = validate_spectra(start, count, selected)
            files = [prefix.with_suffix(s) for s in (".json", ".csv", ".fits")]
            preserve_partial(files, start, "redrock")
            if not any(p.exists() for p in files):
                if start == 0:
                    raise ValueError("pilot Redrock fit must finish before the full scan")
                run_command([str(HOST), "-s", "-B", "-u", str(FIT), "--spectra", str(spectra),
                             "--output-prefix", str(prefix), "--processes", "2"],
                            RUN / f"test_{start:04d}_runner.log")
            meta = validate_fit(start, spectra)
            done.append({"start": start, "galaxies": count, "spectra": meta["attempted"],
                         "returned": meta["returned"], "elapsed_seconds": meta["elapsed_seconds"]})
            print(f"Finished batch {start}: {len(done)}/{len(PLAN)}", flush=True)
            write_progress("running", done)
        write_progress("complete", done)
        print("Completed the full catalog-consistent bright/dark scan.", flush=True)
    except Exception as exc:
        write_progress("failed", done, start, str(exc))
        raise


if __name__ == "__main__":
    main()
