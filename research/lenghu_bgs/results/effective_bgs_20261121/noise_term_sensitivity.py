"""Model-only impact of omitting specified noise terms on the time mapping."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from effective_model import DepthModel
from build_analysis import sha

HERE=Path(__file__).resolve().parent


def main():
    rows=[]
    for name in ("baseline","no_read_noise","no_sky_residual","no_source_poisson"):
        m=DepthModel()
        for arm in "BRZ":
            if name=="no_read_noise":m.d[f"{arm}_read"]=np.array(0.)
            if name=="no_source_poisson":m.d[f"{arm}_continuum"][:]=0.
            if name=="no_sky_residual":
                for sky in ("nominal","dark","bright"):
                    m.d[f"{sky}_{arm}_sky_ref"][:]=0.
        for t in (1500.,3000.):
            eff=m.effective_time("dark",t)
            rows.append({"case":name,"dark_anchor_s":t,"nominal_equivalent_s":eff,
                         "bright_equal_information_s":m.exposure_for_depth("bright",eff)})
    result=pd.DataFrame(rows)
    result.to_csv(HERE/"noise_term_sensitivity.csv",index=False)
    meta={"status":"hypothetical model sensitivity; not new Redrock success estimates",
          "interpretation":"Each term is removed in all scenes, including the reference; nominal depth is recalculated.",
          "depth_inputs_sha256":sha(HERE/"depth_inputs.npz"),"script_sha256":sha(Path(__file__))}
    (HERE/"noise_term_sensitivity_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(result.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
