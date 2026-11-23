"""Coherent +1-sigma tabulated Qiu extinction perturbation, proxy only."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path[:0]=[str(HERE),str(STUDY/"pilot_4200")]
from effective_model import DepthModel
from build_analysis import sha
from lenghu_etc import QiuObservedAtmosphere,QIU_AIRMASS,QIU_CSV


def main():
    qiu=QiuObservedAtmosphere()
    x_nom=1./(1.+.5057*(6.0799+90.)**(-1.6364))
    rows=[]
    # A coherent -1 sigma variant would make some tabulated k negative. Do not
    # clip it or present it as a physical atmospheric transmission.
    for sign in (0,1):
        m=DepthModel()
        for arm in "BRZ":
            w=m.d[f"{arm}_wave_nm"]
            sigma=np.interp(w,qiu.wave_nm,qiu.sigma_k)
            inside=(w>=qiu.wave_nm[0])&(w<=qiu.wave_nm[-1])
            # Test a coherent shift; positive source transmission must remain <=1.
            basek=np.interp(w,qiu.wave_nm,qiu.k)
            if np.any(basek[inside]+sign*sigma[inside]<0):
                raise ValueError("coherent shift would imply negative extinction")
            for scene,x in (("nominal",x_nom),("dark",QIU_AIRMASS),("bright",QIU_AIRMASS)):
                correction=np.where(inside,10**(-.4*sign*sigma*x),1.)
                m.d[f"{scene}_{arm}_response_no_dust"]*=correction
        for t in (1500.,3000.):
            eff=m.effective_time("dark",t)
            rows.append({"qiu_sigma_shift":sign,"dark_anchor_s":t,"nominal_equivalent_s":eff,
                         "bright_equal_information_s":m.exposure_for_depth("bright",eff)})
    tab=pd.DataFrame(rows);tab.to_csv(HERE/"qiu_extinction_sensitivity.csv",index=False)
    info={"status":"sensitivity only; not an uncertainty bound or new Redrock run",
          "perturbation":"coherently shift Qiu k bins by +sigma_k; outside 400-790 nm unchanged; -sigma omitted because some extinction coefficients become negative",
          "ground_sky":"held fixed, as Qiu extinction measures source transmission not sky radiance",
          "qiu_table_sha256":sha(QIU_CSV),"depth_inputs_sha256":sha(HERE/"depth_inputs.npz"),
          "script_sha256":sha(Path(__file__))}
    (HERE/"qiu_extinction_sensitivity_manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    print(tab.to_string(index=False));print(json.dumps(info,indent=2))


if __name__=="__main__":main()
