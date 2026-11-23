"""Provisional per-arm nominal-equivalent depths; no per-arm success calibration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from effective_model import DepthModel
from build_analysis import sha

HERE=Path(__file__).resolve().parent


def arm_reference_time(model, arm_index, goal):
    for n in range(1,17):
        lo=(n-1)*750+(1e-7 if n>1 else 1e-7)
        hi=n*750.
        f=lambda t:model.information("nominal",t,per_arm=True)[arm_index]-goal
        if abs(f(hi)) < 1e-8*goal:return hi
        if f(hi)>0:
            return lo if f(lo)>=0 else brentq(f,lo,hi,xtol=1e-9)
    return float("nan")


def main():
    m=DepthModel();rows=[]
    for sky,t in (("dark",1500),("bright",2250),("dark",3000),("bright",3600)):
        q=m.information(sky,t,per_arm=True)
        total=m.effective_time(sky,t)
        for i,arm in enumerate("BRZ"):
            rows.append({"sky":sky,"actual_s":t,"arm":arm,"arm_information":q[i],
                         "arm_equivalent_s":arm_reference_time(m,i,q[i]),
                         "total_equivalent_s":total})
    out=pd.DataFrame(rows)
    out.to_csv(HERE/"per_arm_depths.csv",index=False)
    meta={"status":"engineering diagnostic; no arm-specific Redrock success calibration",
          "method":"same nominal inverse-information procedure applied separately to each arm",
          "weights":"same 110 calibration-spectrum mean squared features as scalar model",
          "depth_inputs_sha256":sha(HERE/"depth_inputs.npz"),"script_sha256":sha(Path(__file__))}
    (HERE/"per_arm_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(out.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
