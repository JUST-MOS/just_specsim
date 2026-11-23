"""Run full-default-template Redrock on a frozen simulation batch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = Path(__file__).resolve().parent
WRAPPER = HERE.parent / "instrument/rrdesi_015_spawn.py"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectra", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--processes", type=int, default=4)
    args = p.parse_args()
    spectra = args.spectra.resolve()
    truthfile = spectra.with_name(spectra.stem+"_truth.csv")
    maskfile = spectra.with_name(spectra.stem+"_masks.csv")
    manifestfile = spectra.with_name(spectra.stem+"_manifest.json")
    manifest = json.loads(manifestfile.read_text())
    if (sha(spectra) != manifest["spectra_sha256"] or sha(truthfile) != manifest["truth_sha256"] or
        sha(maskfile) != manifest["mask_sha256"]):
        raise ValueError("batch inputs differ from simulation manifest")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    redfile = prefix.with_suffix(".fits")
    logfile = prefix.with_suffix(".log")
    runtimefile = prefix.with_suffix(".json")
    resultfile = prefix.with_suffix(".csv")
    cmd = [sys.executable,"-s","-B",str(WRAPPER),"-i",str(spectra),"--mp",str(args.processes),"-o",str(redfile)]
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
                "PYTHONDONTWRITEBYTECODE":"1"})
    started = time.monotonic()
    with logfile.open("w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
    runtime = {"command": cmd, "child_exit_code": result.returncode,
               "elapsed_seconds": round(time.monotonic()-started,3), "spectra_sha256": sha(spectra),
               "input_manifest_sha256": sha(manifestfile), "fit_script_sha256": sha(Path(__file__))}
    runtimefile.write_text(json.dumps(runtime,indent=2)+"\n")
    if result.returncode != 0:
        raise RuntimeError(f"Redrock process exit {result.returncode}; see {logfile}")
    truth = pd.read_csv(truthfile)
    if not truth.SYNTHETIC_TARGETID.is_unique:
        raise ValueError("duplicate simulated target IDs")
    with fits.open(redfile, memmap=False) as h:
        version = h[0].header.get("RRVER",h["REDSHIFTS"].header.get("RRVER","unknown"))
        table = h["REDSHIFTS"].data
        ids = np.asarray(table["TARGETID"], dtype=np.int64)
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(truth.SYNTHETIC_TARGETID)):
            raise ValueError("unexpected or duplicate Redrock target IDs")
        spectype = np.asarray(table["SPECTYPE"]).astype(str)
        red = pd.DataFrame({"SYNTHETIC_TARGETID":ids,"z_fit":np.asarray(table["Z"]),
            "zerr":np.asarray(table["ZERR"]),"ZWARN":np.asarray(table["ZWARN"]),
            "DELTACHI2":np.asarray(table["DELTACHI2"]),"SPECTYPE":spectype})
    if version != "0.15.4":
        raise ValueError(f"unexpected Redrock version: {version}")
    rows = truth.merge(red,on="SYNTHETIC_TARGETID",how="left",validate="one_to_one")
    rows["returned"] = rows.z_fit.notna()
    rows["delta_z_norm"] = (rows.z_fit-rows.z_true_vi)/(1+rows.z_true_vi)
    rows["quality_pass"] = rows.returned & (rows.ZWARN==0) & (rows.DELTACHI2>40)
    rows["truth_pass"] = rows.returned & (rows.delta_z_norm.abs()<.0033)
    rows["success"] = rows.quality_pass & rows.truth_pass
    rows["outcome"] = np.select([~rows.returned, rows.returned & ~rows.quality_pass,
        rows.quality_pass & ~rows.truth_pass, rows.success],
        ["no_result","quality_fail","wrong_redshift","success"],default="internal_error")
    if (rows.outcome == "internal_error").any():
        raise ValueError("outcome partition failed")
    rows.to_csv(resultfile,index=False)
    runtime.update({"redrock_version":version,"output_sha256":sha(redfile),"results_sha256":sha(resultfile),
                    "attempted":len(rows),"returned":int(rows.returned.sum()),
                    "quality_pass":int(rows.quality_pass.sum()),"success":int(rows.success.sum()),
                    "outcomes":rows.outcome.value_counts().to_dict()})
    runtimefile.write_text(json.dumps(runtime,indent=2)+"\n")
    print(json.dumps(runtime,indent=2),flush=True)


if __name__ == "__main__":
    main()
