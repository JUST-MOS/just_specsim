"""Exploratory paired test of Redrock success at nearly equal proxy depths."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/"leader_195_205_20261116/extended_20p5/analysis_20261119/individual_results.csv"
sys.path.insert(0,str(HERE))
from effective_model import DepthModel
from build_analysis import sha


def main():
    d=pd.read_csv(DATA)
    dark=d[(d.sky=="dark")&(d.total_s==3000)]
    bright=d[(d.sky=="bright")&(d.total_s==3600)]
    x=dark.merge(bright,on="TARGETID",suffixes=("_dark","_bright"),validate="one_to_one")
    if len(x)!=1332 or not x.TARGETID.is_unique:
        raise ValueError("unexpected paired sample")
    np.testing.assert_allclose(x.r_dered_dark,x.r_dered_bright,rtol=0,atol=1e-12)
    np.testing.assert_allclose(x.VI_Z_dark,x.VI_Z_bright,rtol=0,atol=1e-12)
    groups={"r<19.5":x.r_dered_dark<19.5,
            "19.5<=r<=20":(x.r_dered_dark>=19.5)&(x.r_dered_dark<=20),
            "20<r<20.5":x.r_dered_dark>20,
            "r<20.5":np.ones(len(x),dtype=bool),
            "star_forming":x.spectral_type_dark=="bpt_star_forming",
            "weak_line_old":x.spectral_type_dark=="weak_line_old",
            "other":x.spectral_type_dark=="other_unclassified"}
    rng=np.random.default_rng(20261121)
    rows=[]
    for name,mask in groups.items():
        q=x.loc[mask];n=len(q)
        if not n:raise ValueError(f"empty group {name}")
        darkok=q.success_dark.astype(bool).to_numpy()
        brightok=q.success_bright.astype(bool).to_numpy()
        darkonly=int((darkok&~brightok).sum())
        brightonly=int((brightok&~darkok).sum())
        delta=(brightonly-darkonly)/n
        # Cluster bootstrap of pairs, reduced to a multinomial over outcomes.
        draws=rng.multinomial(n,[darkonly/n,brightonly/n,1-(darkonly+brightonly)/n],size=20000)
        delta_boot=(draws[:,1]-draws[:,0])/n
        discordant=darkonly+brightonly
        p=binomtest(darkonly,discordant,.5).pvalue if discordant else 1.
        rows.append({"group":name,"n":n,"dark_success":int(darkok.sum()),
                     "bright_success":int(brightok.sum()),"dark_only":darkonly,
                     "bright_only":brightonly,"difference_bright_minus_dark":delta,
                     "paired_bootstrap_95_lo":float(np.quantile(delta_boot,.025)),
                     "paired_bootstrap_95_hi":float(np.quantile(delta_boot,.975)),
                     "exploratory_two_sided_mcnemar_p":p})
    result=pd.DataFrame(rows)
    result.to_csv(HERE/"paired_transfer_check.csv",index=False)
    model=DepthModel()
    dark_arm=model.information("dark",3000,per_arm=True)
    bright_arm=model.information("bright",3600,per_arm=True)
    meta={"status":"exploratory; no sky/condition or multiple-group correction",
          "source_sha256":sha(DATA),"model_inputs_sha256":sha(HERE/"depth_inputs.npz"),
          "dark_actual_s":3000,"bright_actual_s":3600,
          "dark_effective_s":model.effective_time("dark",3000),
          "bright_effective_s":model.effective_time("bright",3600),
          "feature_information_by_arm_dark":dict(zip("BRZ",dark_arm.tolist())),
          "feature_information_by_arm_bright":dict(zip("BRZ",bright_arm.tolist())),
          "bright_to_dark_feature_information_ratio_by_arm":dict(zip("BRZ",(bright_arm/dark_arm).tolist())),
          "bright_to_dark_feature_information_ratio_total":float(bright_arm.sum()/dark_arm.sum()),
          "bootstrap":"20000 paired-galaxy multinomial resamples, seed 20261121; conditional iid targets",
          "p_value":"exact two-sided binomial McNemar test on discordant same-target outcomes",
          "limitations":"post-hoc comparison, same single input field, independent noise draws, 1% mismatch in proxy depth; not an on-sky validation"}
    (HERE/"paired_transfer_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(result.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
