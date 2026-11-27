"""Compute disjoint tail batches with the frozen simulator and two-worker fitter.

The original ascending driver retains its plan, lock and progress ledger. This
optional second driver computes tail batches in a staging directory, checks each
with the original validator, then hard-links immutable complete files into the
original output directory. It stops before approaching the ascending driver.
The staging paths are retained because fit metadata records the command path.
"""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nominal_scan as scan

MAIN = scan.OUT
STAGE = HERE / "nominal_run/prefetch_2worker"


def save(status, done, current=None, error=None):
    out = {"status": status, "completed": done, "current_start": current,
           "error": error, "updated_utc": datetime.now(timezone.utc).isoformat(),
           "worker_count_per_redrock": 2,
           "note": "Disjoint tail batches; staging and main paths refer to the same inodes after linking."}
    tmp = STAGE / ".progress.tmp"
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, STAGE / "progress.json")


def safely_ahead(start):
    main = json.loads((MAIN / "progress.json").read_text())
    if main["status"] != "running":
        return False
    cursor = main["current_start"]
    if cursor is None:
        cursor = max((x["start"] for x in main["completed"]), default=0)
    return int(cursor) + 100 < start


def publish(start):
    if not safely_ahead(start):
        return False
    files = sorted(STAGE.glob(f"test_{start:04d}_*"))
    spectra = STAGE / f"test_{start:04d}_spectra.fits"
    if spectra not in files:
        raise ValueError("missing staged spectrum")
    # Publish the sentinel spectrum last, after all validated companion files.
    files = [x for x in files if x != spectra] + [spectra]
    made = []
    try:
        for source in files:
            destination = MAIN / source.name
            if destination.exists():
                if scan.sha(destination) != scan.sha(source):
                    raise FileExistsError(f"different existing main file: {destination}")
            else:
                os.link(source, destination)
                made.append(destination)
    except Exception:
        for destination in made:
            destination.unlink()
        raise
    return True


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    with (STAGE / ".prefetch.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = scan.preflight()  # uses MAIN and its validated pilot
        # We never write the main progress ledger or alter its run plan.
        scan.OUT = STAGE
        done = []
        try:
            for start, count in reversed(scan.PLAN):
                if not safely_ahead(start):
                    break
                save("running", done, start)
                spectra = STAGE / f"test_{start:04d}_spectra.fits"
                if not spectra.exists():
                    scan.run([str(scan.HOST), "-s", "-B", "-u", str(HERE / "simulate_nominal_batch.py"),
                              "--split", "test", "--start", str(start), "--count", str(count),
                              "--times", *map(str, scan.TIMES), "--resolution-file",
                              str(scan.DATA / "combined_resolution_nodes.npz"), "--output", str(spectra)],
                             STAGE / f"test_{start:04d}_simulation.log")
                spectra = scan.validate_input(start, count, selected)
                prefix = STAGE / f"test_{start:04d}_redrock"
                outputs = [prefix.with_suffix(s) for s in (".json", ".csv", ".fits")]
                if not any(p.exists() for p in outputs):
                    scan.run([str(scan.HOST), "-s", "-B", "-u", str(scan.FIT),
                              "--spectra", str(spectra), "--output-prefix", str(prefix),
                              "--processes", "2"], STAGE / f"test_{start:04d}_runner.log")
                fit = scan.validate_fit(start, spectra)
                if not publish(start):
                    print(f"Stopped before publishing batch {start}: main driver is too close", flush=True)
                    break
                # Revalidate by the same rules from the original output path.
                scan.OUT = MAIN
                scan.validate_input(start, count, selected)
                scan.validate_fit(start, MAIN / spectra.name)
                scan.OUT = STAGE
                done.append({"start": start, "galaxies": count, "attempted": fit["attempted"],
                             "returned": fit["returned"], "elapsed_seconds": fit["elapsed_seconds"]})
                save("running", done)
                print(f"Prefetched batch {start}: {sum(x['attempted'] for x in done)} fits", flush=True)
            save("complete", done)
            print("Prefetch stopped safely; the ascending driver remains authoritative.", flush=True)
        except Exception as exc:
            save("failed", done, start, str(exc))
            raise


if __name__ == "__main__":
    main()
