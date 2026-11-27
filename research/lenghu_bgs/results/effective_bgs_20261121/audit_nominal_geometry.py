"""Check the nominal seeing's 33-node geometric interpolation directly."""
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[2]
sys.path[:0]=[str(ROOT/"py"),str(HERE)]
from just_etc import JUSTExposureTimeCalculator,ETC_py_optimized as eng
from build_analysis import SEEING_NOMINAL_800,sha


def main():
    path=HERE/"nominal_run/shared/instrument_reference_4200.npz"
    with np.load(path,allow_pickle=False) as h:
        d={k:h[k].copy() for k in h.files}
    c=JUSTExposureTimeCalculator(calc_mode="accurate")
    c.set_obs_conditions(seeing_fwhm_800=SEEING_NOMINAL_800,ebv=0.,r_eff=0.,decenter=.03)
    c._set_mode();c._configure_spectro(1)
    obs,sp=c._make_obs(),c._spectro
    result={}
    for arm in "BRZ":
        wave=d[f"{arm}_wave_nm"];nodes=d[f"{arm}_geo_wave_nodes_nm"]
        ix=np.linspace(3,len(wave)-4,97).round().astype(int)
        worst=0.
        for radius in (0.,.2925868,.5,1.5,3.470199625,5.,15.3008):
            y=np.array([eng.gsGeometricThroughput(sp,obs,float(w),radius,.03,0.,0) for w in nodes])
            actual=np.array([eng.gsGeometricThroughput(sp,obs,float(w),radius,.03,0.,0) for w in wave[ix]])
            if np.any(actual<=0) or np.any(y<=0):raise ValueError("nonpositive geometric throughput")
            worst=max(worst,float(np.max(np.abs(np.interp(wave[ix],nodes,y)/actual-1))))
        if worst>=1e-3:raise ValueError(f"interpolation above 0.1% for {arm}: {worst}")
        result[arm]=worst
    out={"status":"passed","scope":"33 nodes, 97 wavelengths and seven exponential radii at nominal seeing",
         "max_relative_error_by_arm":result,"nominal_instrument_sha256":sha(path),"script_sha256":sha(Path(__file__))}
    (HERE/"nominal_geometry_audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
