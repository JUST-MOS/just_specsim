"""Checkpointed held-out Redrock run at two 4200 m nominal reference depths."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
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
POP = STUDY/"population_4200_camera_overlap"
DATA = STUDY/"leader_195_205_20261116/extended_20p5/input"
OUT = HERE/"nominal_run/v2"
SHARED = HERE/"nominal_run/shared"
FIT = STUDY/"just_desi_550s_20261115/fit_stable.py"
HOST = Path("/Users/rain/miniconda3/envs/desi/bin/python")
sys.path.insert(0,str(HERE))
from build_analysis import sha

TIMES = [1817,3436]
PLAN = [(0,8)]+[(i,min(50,1332-i)) for i in range(8,1332,50)]


def preflight():
    design = json.loads((DATA/"frozen_design.json").read_text())
    for name,key in (("frozen_sample.csv","sample_sha256"),("normalized_sources.fits","normalized_sources_sha256")):
        if sha(DATA/name) != design[key]:
            raise ValueError(f"frozen source changed: {name}")
    lsf=json.loads((DATA/"combined_resolution_manifest.json").read_text())
    if lsf["status"]!="passed" or sha(DATA/"combined_resolution_nodes.npz")!=lsf["combined_nodes_sha256"]:
        raise ValueError("combined LSF not verified")
    v=json.loads((SHARED/"instrument_validation.json").read_text())
    if (v["status"]!="passed" or sha(SHARED/"instrument_reference_4200.npz")!=v["instrument_sha256"] or
        sha(HERE/"depth_inputs.npz")!=v["depth_inputs_sha256"]):
        raise ValueError("nominal instrument has changed")
    pilot=json.loads((OUT/"pilot_input_audit.json").read_text())
    if (pilot["status"]!="passed" or pilot["simulator_sha256"]!=sha(HERE/"simulate_nominal_batch.py") or
        pilot["spectra_sha256"]!=sha(OUT/"test_0000_spectra.fits")):
        raise ValueError("pilot spectrum inputs not validated")
    stable=json.loads((FIT.parent/"stable_runtime_validation.json").read_text())
    if stable["status"]!="passed" or not Path(stable["python"]).is_file():
        raise ValueError("stable Redrock runtime unavailable")
    cat=pd.read_csv(DATA/"frozen_sample.csv")
    selected=cat[cat.split=="test"].reset_index(drop=True)
    if (len(selected)!=1332 or not selected.TARGETID.is_unique or
        int((selected.r_dered<19.5).sum())!=578 or sum(n for _,n in PLAN)!=1332):
        raise ValueError("held-out sample or plan mismatch")
    return selected


def validate_input(start,count,selected):
    path=OUT/f"test_{start:04d}_spectra.fits"
    truthpath=path.with_name(path.stem+"_truth.csv")
    maskpath=path.with_name(path.stem+"_masks.csv")
    manpath=path.with_name(path.stem+"_manifest.json")
    if not all(x.is_file() for x in (path,truthpath,maskpath,manpath)):
        raise ValueError(f"incomplete batch {start}")
    m=json.loads(manpath.read_text())
    for f,key in ((path,"spectra_sha256"),(truthpath,"truth_sha256"),(maskpath,"mask_sha256"),
                  (HERE/"simulate_nominal_batch.py","wrapper_sha256"),
                  (POP/"simulate_batch.py","simulation_script_sha256"),
                  (DATA/"combined_resolution_nodes.npz","resolution_sha256"),
                  (SHARED/"instrument_reference_4200.npz","instrument_sha256"),
                  (SHARED/"desi_camera_mapping.npz","camera_mapping_sha256")):
        if sha(f)!=m[key]:
            raise ValueError(f"batch hash mismatch {start}: {key}")
    if (m["selection_start"]!=start or m["galaxies"]!=count or m["times_s"]!=TIMES or
        m["sky_cases"]!=["nominal"] or m["target_ebv_applied"]!=0 or
        m["spectra_rows"]!=2*count):
        raise ValueError(f"wrong batch definition {start}")
    truth=pd.read_csv(truthpath)
    exp={(int(t),s) for t in selected.iloc[start:start+count].TARGETID for s in TIMES}
    if (len(truth)!=len(exp) or not truth.SYNTHETIC_TARGETID.is_unique or
        set(zip(truth.TARGETID.astype(int),truth.total_s.astype(int)))!=exp):
        raise ValueError(f"missing or duplicate spectra {start}")
    with fits.open(path,memmap=True) as h:
        np.testing.assert_array_equal(h["FIBERMAP"].data["TARGETID"],truth.SYNTHETIC_TARGETID.to_numpy())
    return path


def validate_fit(start,spectra):
    prefix=OUT/f"test_{start:04d}_redrock"
    meta_path, csv_path, fits_path=(prefix.with_suffix(s) for s in (".json",".csv",".fits"))
    if not all(p.is_file() for p in (meta_path,csv_path,fits_path)):
        raise ValueError(f"incomplete fit {start}")
    m=json.loads(meta_path.read_text())
    if (m["child_exit_code"]!=0 or m["redrock_version"]!="0.15.4" or
        m["spectra_sha256"]!=sha(spectra) or m["results_sha256"]!=sha(csv_path) or
        m["output_sha256"]!=sha(fits_path) or m["stable_fit_script_sha256"]!=sha(FIT)):
        raise ValueError(f"fit provenance mismatch {start}")
    truth=pd.read_csv(spectra.with_name(spectra.stem+"_truth.csv"))
    rows=pd.read_csv(csv_path)
    if (len(rows)!=len(truth) or not rows.SYNTHETIC_TARGETID.is_unique or
        set(rows.SYNTHETIC_TARGETID)!=set(truth.SYNTHETIC_TARGETID)):
        raise ValueError(f"fit denominator or IDs mismatch {start}")
    rows=truth.merge(rows,on="SYNTHETIC_TARGETID",validate="one_to_one",suffixes=("_truth",""))
    for key in ("TARGETID","source_row","sky","total_s","exposures_s"):
        np.testing.assert_array_equal(rows[key],rows[key+"_truth"])
    with fits.open(fits_path,memmap=False) as h:
        fitted=h["REDSHIFTS"].data
        ids=np.asarray(fitted["TARGETID"],dtype=np.int64)
        if len(ids)!=len(set(ids)) or not set(ids).issubset(set(truth.SYNTHETIC_TARGETID)):
            raise ValueError(f"Redrock FITS IDs mismatch {start}")
        values=rows.set_index("SYNTHETIC_TARGETID").loc[ids]
        for src,dst in (("Z","z_fit"),("ZERR","zerr"),("DELTACHI2","DELTACHI2")):
            np.testing.assert_allclose(np.asarray(fitted[src],dtype=float),values[dst],atol=1e-11,rtol=1e-11)
        np.testing.assert_array_equal(fitted["ZWARN"],values.ZWARN)
    returned=np.isfinite(rows.z_fit)
    success=returned&(rows.ZWARN==0)&(rows.DELTACHI2>40)&(
        np.abs(rows.z_fit-rows.z_true_vi)/(1+rows.z_true_vi)<.0033)
    np.testing.assert_array_equal(returned,rows.returned)
    np.testing.assert_array_equal(success,rows.success)
    if int(returned.sum())!=m["returned"] or len(rows)!=m["attempted"]:
        raise ValueError(f"returned count mismatch {start}")
    return m


def save_progress(status, done, current=None, error=None):
    p={"status":status,"completed":done,"current_start":current,"error":error,
       "completed_spectra":sum(x["attempted"] for x in done),
       "total_spectra":2664,"updated_utc":datetime.now(timezone.utc).isoformat()}
    tmp=OUT/".progress.tmp"
    tmp.write_text(json.dumps(p,indent=2)+"\n")
    os.replace(tmp,OUT/"progress.json")


def run(command, logfile):
    with logfile.open("w") as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
        try:
            code=proc.wait(timeout=3600)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=30)
            raise
    if code!=0:
        raise RuntimeError(f"process exit {code}; inspect {logfile}")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/".run.lock").open("w")
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=preflight()
    plan={"objective":"two nominal depths for all 1332 held-out galaxies",
          "times_s":TIMES,"splits_s":[[750,750,317],[750,750,750,750,436]],
          "batches":[{"start":i,"count":n} for i,n in PLAN],"redrock_processes":2,
          "selection_sha256":sha(DATA/"frozen_sample.csv"),
          "normalized_spectra_sha256":sha(DATA/"normalized_sources.fits"),
          "lsf_sha256":sha(DATA/"combined_resolution_nodes.npz"),
          "nominal_instrument_sha256":sha(SHARED/"instrument_reference_4200.npz"),
          "simulator_sha256":sha(HERE/"simulate_nominal_batch.py"),
          "driver_sha256":sha(Path(__file__)),"fitter_sha256":sha(FIT)}
    plan_path=OUT/"run_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError("run plan changed; refusing to overwrite")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan,indent=2)+"\n")
    done=[]
    try:
        for start,count in PLAN:
            save_progress("running",done,start)
            spectra=OUT/f"test_{start:04d}_spectra.fits"
            if not spectra.exists():
                run([str(HOST),"-s","-B","-u",str(HERE/"simulate_nominal_batch.py"),
                     "--split","test","--start",str(start),"--count",str(count),
                     "--times",*map(str,TIMES),"--resolution-file",str(DATA/"combined_resolution_nodes.npz"),
                     "--output",str(spectra)],OUT/f"test_{start:04d}_simulation.log")
            spectra=validate_input(start,count,selected)
            prefix=OUT/f"test_{start:04d}_redrock"
            outputs=[prefix.with_suffix(s) for s in (".json",".csv",".fits")]
            if not any(p.exists() for p in outputs):
                run([str(HOST),"-s","-B","-u",str(FIT),"--spectra",str(spectra),
                     "--output-prefix",str(prefix),"--processes","2"],OUT/f"test_{start:04d}_runner.log")
            m=validate_fit(start,spectra)
            done.append({"start":start,"galaxies":count,"attempted":m["attempted"],
                         "returned":m["returned"],"elapsed_seconds":m["elapsed_seconds"]})
            save_progress("running",done)
            print(f"Completed batch {start}: {sum(x['attempted'] for x in done)}/2664 fits",flush=True)
        save_progress("complete",done)
        print("All 2664 nominal-condition Redrock cases complete.",flush=True)
    except Exception as exc:
        save_progress("failed",done,start,str(exc))
        raise


if __name__=="__main__":
    main()
