"""Summarize the full r<19.5 holdout once all of its targets are finished."""
import json
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_nominal_scan as scan
from build_analysis import sha


def main():
    progress=json.loads((scan.OUT/"progress.json").read_text())
    done={int(a["start"]) for a in progress["completed"]}
    sample=scan.preflight()
    bgs=sample[sample.r_dered<19.5]
    expected=set(bgs.TARGETID)
    chunks=[]
    for start,count in scan.PLAN:
        if start not in done:continue
        spectra=scan.validate_input(start,count,sample)
        scan.validate_fit(start,spectra)
        chunks.append(pd.read_csv(scan.OUT/f"test_{start:04d}_redrock.csv"))
    if not chunks:raise ValueError("no completed batches")
    data=pd.concat(chunks,ignore_index=True)
    data=data[data.TARGETID.isin(expected)]
    for t in scan.TIMES:
        sub=data[data.total_s==t]
        if len(sub)!=578 or set(sub.TARGETID)!=expected:
            print(f"BGS not complete at {t}: {len(sub)}/578",flush=True)
            return
    result=[]
    for t in scan.TIMES:
        sub=data[data.total_s==t];n=len(sub);k=int(sub.success.sum())
        lo=binomtest(k,n,alternative="greater").proportion_ci(confidence_level=.975,method="exact").low
        result.append({"nominal_s":t,"success":k,"n":n,"fraction":k/n,
                       "two_time_adjusted_95_lower":lo,"returned":int(sub.returned.sum())})
    table=pd.DataFrame(result)
    path=HERE/"nominal_bgs_complete_early.csv"
    table.to_csv(path,index=False)
    meta={"status":"complete for r<19.5 only; full r<20.5 scan continues",
          "n":578,"times":scan.TIMES,"correction":"one-sided 95% Bonferroni over 2 nominal times",
          "total_complete_cases_at_export":progress["completed_spectra"],
          "table_sha256":sha(path),"script_sha256":sha(Path(__file__))}
    (HERE/"nominal_bgs_complete_early.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(table.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
