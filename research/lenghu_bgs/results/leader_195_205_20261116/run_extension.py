"""Checkpointed r < 20.5 extension: simulate and fit only new test galaxies.

The 891 original test galaxies retain their frozen existing results.  Each
new target is simulated once per time and sky with the unchanged forward model;
all targets, including missing Redrock outcomes, remain in the denominator.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from astropy.io import fits
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
OLD = STUDY / "population_4200_camera_overlap"
PRIOR = STUDY / "just_desi_550s_20261115"
ROOT = HERE / "extended_20p5"
DATA, SHARED, OUT = ROOT / "input", ROOT / "shared", ROOT / "run"
HOST = Path("/Users/rain/miniconda3/envs/desi/bin/python")
SIM = HERE / "simulate_extension.py"
FIT = PRIOR / "fit_stable.py"
LSF = DATA / "combined_resolution_nodes.npz"
TIMES = [750, 1500, 2250, 3000, 3600]
SKY = ["dark", "bright"]
PLAN = [(891, 8)] + [(s, min(50, 1332-s)) for s in range(899, 1332, 50)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight():
    design = json.loads((DATA / "frozen_design.json").read_text())
    prepared = json.loads((DATA / "extension_manifest.json").read_text())
    resolution = json.loads((DATA / "combined_resolution_manifest.json").read_text())
    reproduced = json.loads((ROOT / "smoke/original_reproduction_validation.json").read_text())
    pilot = json.loads((OUT / "faint_pilot_input_validation.json").read_text())
    if (sha(DATA / "frozen_design.json") != prepared["new_design_sha256"] or
        sha(DATA / "frozen_sample.csv") != design["sample_sha256"] or
        sha(DATA / "normalized_sources.fits") != design["normalized_sources_sha256"] or
        sha(LSF) != resolution["combined_nodes_sha256"] or resolution["status"] != "passed" or
        reproduced["status"] != "passed" or reproduced["wrapper_sha256"] != sha(SIM) or
        pilot["status"] != "passed" or pilot["wrapper_sha256"] != sha(SIM) or
        pilot["input_fits_sha256"] != sha(OUT / "test_0891_spectra.fits") or
        pilot["combined_lsf_sha256"] != sha(LSF)):
        raise ValueError("frozen input, reference LSF, or pilot validation changed")
    stable = json.loads((PRIOR / "stable_runtime_validation.json").read_text())
    if (stable["status"] != "passed" or not Path(stable["python"]).is_file() or
        stable["environment_sha256"] != sha(PRIOR / "stable_environment.json") or
        stable["reference_redrock_sha256"] != sha(PRIOR / "batch_0000_redrock.fits") or
        stable["new_redrock_sha256"] != sha(PRIOR / "stable_validation_redrock.fits")):
        raise ValueError("validated persistent Redrock environment unavailable")
    old_design = json.loads((OLD / "input/frozen_design.json").read_text())
    frozen = json.loads((OLD / "output/quality_frozen.json").read_text())
    if (frozen["status"] != "frozen" or frozen["cut"] != old_design["quality_reference"] or
        frozen["design_sha256"] != sha(OLD / "input/frozen_design.json")):
        raise ValueError("original reference quality cut is not frozen")
    cat = pd.read_csv(DATA / "frozen_sample.csv")
    chosen = cat[cat.split == "test"].reset_index(drop=True)
    new = chosen.iloc[891:]
    old_ids = set(pd.read_csv(OLD / "input/frozen_sample.csv").query("split == 'test'").TARGETID)
    if (len(chosen) != 1332 or len(new) != 441 or not chosen.TARGETID.is_unique or
        set(chosen.iloc[:891].TARGETID) != old_ids or
        not ((new.r_dered > 20) & (new.r_dered < 20.5)).all() or
        sum(n for _, n in PLAN) != 441):
        raise ValueError("unexpected expanded held-out roster or batch plan")
    return chosen


def validate_spectra(start, count, selected):
    p = OUT / f"test_{start:04d}_spectra.fits"
    truth = p.with_name(p.stem+"_truth.csv")
    masks = p.with_name(p.stem+"_masks.csv")
    man = p.with_name(p.stem+"_manifest.json")
    if not all(x.is_file() for x in (p, truth, masks, man)):
        raise ValueError(f"incomplete spectrum input for batch {start}")
    m = json.loads(man.read_text())
    for key, path in (("spectra_sha256", p), ("truth_sha256", truth), ("mask_sha256", masks),
                      ("extension_design_sha256", DATA / "frozen_design.json"),
                      ("simulation_script_sha256", OLD / "simulate_batch.py"),
                      ("extension_wrapper_sha256", SIM), ("resolution_sha256", LSF),
                      ("instrument_sha256", SHARED / "instrument_reference_4200.npz"),
                      ("camera_mapping_sha256", SHARED / "desi_camera_mapping.npz")):
        if m[key] != sha(path):
            raise ValueError(f"input provenance mismatch for batch {start}: {key}")
    if (m["selection_start"] != start or m["galaxies"] != count or m["split"] != "test" or
        m["times_s"] != TIMES or m["sky_cases"] != SKY or m["spectra_rows"] != 10*count):
        raise ValueError(f"input scenario specification mismatch for batch {start}")
    t = pd.read_csv(truth)
    selection = selected.iloc[start:start+count]
    expected = {(int(target), time_s, sky) for target in selection.TARGETID for time_s in TIMES for sky in SKY}
    actual = set(zip(t.TARGETID.astype(int), t.total_s.astype(int), t.sky))
    if (len(t) != 10*count or len(actual) != len(t) or actual != expected or
        not t.SYNTHETIC_TARGETID.is_unique):
        raise ValueError(f"input target or scenario coverage mismatch for batch {start}")
    mapped = t.merge(selection[["TARGETID", "VI_Z", "source_row"]], on="TARGETID", validate="many_to_one",
                     suffixes=("", "_catalog"))
    if (not np.array_equal(mapped.source_row, mapped.source_row_catalog) or
        not np.allclose(mapped.z_true_vi, mapped.VI_Z, atol=1e-12, rtol=0)):
        raise ValueError(f"reference truth mismatch for batch {start}")
    with fits.open(p, memmap=False) as h:
        np.testing.assert_array_equal(h["FIBERMAP"].data["TARGETID"], t.SYNTHETIC_TARGETID.to_numpy())
    return p


def validate_fit(start, spectra):
    prefix = OUT / f"test_{start:04d}_redrock"
    runtime, csvfile, fitsfile = (prefix.with_suffix(x) for x in (".json", ".csv", ".fits"))
    if not all(x.is_file() for x in (runtime, csvfile, fitsfile)):
        raise ValueError(f"incomplete Redrock fit for batch {start}")
    meta = json.loads(runtime.read_text())
    if (meta["child_exit_code"] != 0 or meta["redrock_version"] != "0.15.4" or
        meta["spectra_sha256"] != sha(spectra) or meta["results_sha256"] != sha(csvfile) or
        meta["output_sha256"] != sha(fitsfile) or
        meta.get("stable_validation_sha256") != sha(PRIOR / "stable_runtime_validation.json") or
        meta.get("stable_fit_script_sha256") != sha(FIT)):
        raise ValueError(f"Redrock execution provenance mismatch for batch {start}")
    truth = pd.read_csv(spectra.with_name(spectra.stem+"_truth.csv"))
    rows = pd.read_csv(csvfile)
    if (len(rows) != len(truth) or not rows.SYNTHETIC_TARGETID.is_unique or
        set(rows.SYNTHETIC_TARGETID) != set(truth.SYNTHETIC_TARGETID)):
        raise ValueError(f"Redrock result denominator mismatch for batch {start}")
    rows = truth.merge(rows, on="SYNTHETIC_TARGETID", validate="one_to_one", suffixes=("_truth", ""))
    for column in ("TARGETID", "source_row", "sky", "total_s", "exposures_s"):
        np.testing.assert_array_equal(rows[column], rows[column+"_truth"])
    with fits.open(fitsfile, memmap=False) as h:
        a = h["REDSHIFTS"].data
        ids = np.asarray(a["TARGETID"], dtype=np.int64)
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(truth.SYNTHETIC_TARGETID)):
            raise ValueError(f"Redrock FITS target identity mismatch for batch {start}")
        compare = rows.set_index("SYNTHETIC_TARGETID").loc[ids]
        for name, col in (("Z", "z_fit"), ("ZERR", "zerr"), ("DELTACHI2", "DELTACHI2")):
            np.testing.assert_allclose(np.asarray(a[name], dtype=float), compare[col], rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(a["ZWARN"], compare.ZWARN)
        np.testing.assert_array_equal(np.asarray(a["SPECTYPE"]).astype(str), compare.SPECTYPE.str.strip())
    returned = np.isfinite(rows.z_fit)
    quality = returned & (rows.ZWARN == 0) & (rows.DELTACHI2 > 40)
    correct = returned & (np.abs(rows.z_fit-rows.z_true_vi)/(1+rows.z_true_vi) < .0033)
    for computed, saved in ((returned, rows.returned), (quality, rows.quality_pass),
                            (correct, rows.truth_pass), (quality & correct, rows.success)):
        np.testing.assert_array_equal(computed, saved)
    if int(rows.returned.sum()) != meta["returned"] or len(rows) != meta["attempted"]:
        raise ValueError(f"returned count mismatch for batch {start}")
    return meta


def write_progress(status, done, start=None, detail=None):
    payload = {"status": status, "completed": done, "current_start": start,
               "updated_utc": datetime.now(timezone.utc).isoformat(), "detail": detail}
    temporary = OUT / ".progress.tmp"
    temporary.write_text(json.dumps(payload, indent=2)+"\n")
    os.replace(temporary, OUT / "progress.json")


def run_command(command, logfile, timeout=3600):
    print("Running", " ".join(map(str, command)), flush=True)
    with logfile.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy(), start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started > timeout:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=30)
                        raise TimeoutError(f"batch timed out after {timeout} s: {logfile}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            raise
    if code:
        raise RuntimeError(f"batch exited {code}; inspect {logfile}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / ".run.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = preflight()
    plan = {"scope": "441 new held-out galaxies only; reuse old 891 held-out galaxies",
        "times_s": TIMES, "sky": SKY, "processes": 2, "timeout_s": 3600,
        "exposure_splits_s": [[750], [750,750], [750,750,750],
                              [750,750,750,750], [750,750,750,750,600]],
        "batches": [{"start": a, "galaxies": b} for a, b in PLAN],
        "design_sha256": sha(DATA / "frozen_design.json"), "lsf_sha256": sha(LSF),
        "simulation_wrapper_sha256": sha(SIM), "fit_script_sha256": sha(FIT),
        "driver_sha256": sha(Path(__file__))}
    path = OUT / "run_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("existing run plan differs; refusing to replace it")
    if not path.exists():
        path.write_text(json.dumps(plan, indent=2)+"\n")
    done = []
    try:
        for start, count in PLAN:
            write_progress("running", done, start)
            spectra = OUT / f"test_{start:04d}_spectra.fits"
            if not spectra.exists():
                if start == 891:
                    raise ValueError("validated pilot input missing")
                run_command([str(HOST), "-s", "-B", "-u", str(SIM), "--split", "test",
                             "--start", str(start), "--count", str(count), "--times", *map(str, TIMES),
                             "--resolution-file", str(LSF), "--output", str(spectra)],
                            OUT / f"test_{start:04d}_simulation.log")
            spectra = validate_spectra(start, count, selected)
            prefix = OUT / f"test_{start:04d}_redrock"
            files = [prefix.with_suffix(x) for x in (".json", ".csv", ".fits")]
            if not any(p.exists() for p in files):
                if start == 891:
                    raise ValueError("pilot fit must be completed and independently checked before the scan")
                run_command([str(HOST), "-s", "-B", "-u", str(FIT), "--spectra", str(spectra),
                             "--output-prefix", str(prefix), "--processes", "2"],
                            OUT / f"test_{start:04d}_runner.log")
            meta = validate_fit(start, spectra)
            done.append({"start": start, "galaxies": count, "spectra": meta["attempted"],
                         "returned": meta["returned"], "elapsed_seconds": meta["elapsed_seconds"]})
            print(f"Finished batch {start}: {len(done)}/{len(PLAN)}", flush=True)
            write_progress("running", done)
        write_progress("complete", done)
        print("Completed all new held-out galaxy simulations and Redrock fits.", flush=True)
    except Exception as exc:
        write_progress("failed", done, start, str(exc))
        raise


if __name__ == "__main__":
    main()
