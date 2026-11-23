"""Independent raw-redshift and exact-binomial audit of the final scan."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from build_analysis import sha

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/"leader_195_205_20261116/extended_20p5/input/frozen_sample.csv"


def main():
    d=pd.read_csv(HERE/"nominal_individual_results.csv")
    c=pd.read_csv(HERE/"nominal_success_curves.csv")
    selected=pd.read_csv(INPUT)
    selected=selected[selected.split=="test"]
    ids=set(selected.TARGETID)
    if len(ids)!=1332 or len(d)!=2664 or d.duplicated(["TARGETID","total_s"]).any():
        raise ValueError("individual roster or row count invalid")
    for t in (1817,3436):
        if set(d[d.total_s==t].TARGETID)!=ids:
            raise ValueError(f"missing or extra target at {t}")
    returned=np.isfinite(d.z_fit)
    quality=returned&(d.ZWARN==0)&(d.DELTACHI2>40)
    truth=returned&(np.abs(d.z_fit-d.z_true_vi)/(1+d.z_true_vi)<.0033)
    for actual,stored in ((returned,d.returned),(quality,d.quality_pass),
                          (truth,d.truth_pass),(quality&truth,d.success)):
        np.testing.assert_array_equal(np.asarray(actual,dtype=bool),np.asarray(stored,dtype=bool))
    max_abs=0.;checks=[]
    for row in c.itertuples():
        limit=19.5 if row.sample=="r<19.5" else 20.5
        sub=d[(d.total_s==row.actual_s)&(d.r_dered<limit)]
        n=len(sub);k=int(sub.success.sum())
        if (n,k)!=(row.n,row.k_success) or row.n_returned!=int(sub.returned.sum()):
            raise ValueError("summary numerator, denominator or returned count mismatch")
        ci=binomtest(k,n,alternative="greater").proportion_ci(confidence_level=.975,method="exact").low
        p=binomtest(k,n,alternative="greater").proportion_ci(confidence_level=.95,method="exact").low
        max_abs=max(max_abs,abs(ci-row.lower),abs(p-row.lower_pointwise))
        if abs(k/n-row.success_fraction)>1e-14:raise ValueError("fraction mismatch")
        checks.append({"sample":row.sample,"time_s":row.actual_s,"n":n,"success":k,
                       "pointwise_lower":p,"two_time_adjusted_lower":ci})
    if max_abs>1e-10:raise ValueError(f"binomial confidence mismatch: {max_abs}")
    result={"status":"passed","returned":int(returned.sum()),"attempted":len(d),
            "all_target_ids_and_outcomes_checked":True,"checks":checks,
            "max_abs_confidence_difference":max_abs,
            "method":"raw conditions plus scipy.stats.binomtest exact one-sided intervals",
            "individual_sha256":sha(HERE/"nominal_individual_results.csv"),
            "curves_sha256":sha(HERE/"nominal_success_curves.csv"),
            "script_sha256":sha(Path(__file__))}
    (HERE/"nominal_independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
