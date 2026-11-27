"""Reuse only the intact staged input from the stopped optional prefetch."""
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nominal_scan as scan

MAIN = scan.OUT
STAGE = HERE / "nominal_run/prefetch_2worker"


def main():
    status = json.loads((STAGE / "stop_record.json").read_text())
    if status["status"] != "stopped" or status["current_start"] != 1258:
        raise ValueError("unexpected staged-stop status")
    selected = scan.preflight()
    progress = json.loads((MAIN / "progress.json").read_text())
    if progress["status"] != "running" or progress["current_start"] >= 1158:
        raise ValueError("main driver is too close to staged batch")
    scan.OUT = STAGE
    spectra = scan.validate_input(1258, 50, selected)
    files = [spectra.with_name(spectra.stem + suffix) for suffix in
             ("_truth.csv", "_masks.csv", "_manifest.json")]
    files += [STAGE / "test_1258_simulation.log", spectra]  # input sentinel last
    if any((MAIN / p.name).exists() for p in files):
        raise FileExistsError("refusing to replace an existing main input")
    made = []
    try:
        for source in files:
            destination = MAIN / source.name
            os.link(source, destination)
            made.append(destination)
        scan.OUT = MAIN
        scan.validate_input(1258, 50, selected)
    except Exception:
        for destination in made:
            destination.unlink()
        raise
    info = {"status": "passed", "start": 1258, "galaxies": 50,
            "scope": "Only validated spectra and sidecars hard-linked; interrupted Redrock outputs remain in staging.",
            "files": {p.name: scan.sha(p) for p in files}}
    (STAGE / "reused_input.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
