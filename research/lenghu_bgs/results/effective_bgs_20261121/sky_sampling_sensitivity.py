"""Compare 0.05 and 0.01 nm zenith-sky input grids at fixed r brightness."""
import json
from pathlib import Path
import sys

from astropy.io import fits
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path[:0]=[str(HERE),str(STUDY/"pilot")]
from build_analysis import sky_r_ab,sha
from effective_model import DepthModel
from just_etc import JUSTExposureTimeCalculator
from refine_sky_sampling import apply_sky,setup_arm

COARSE=STUDY/"sky/height_4200/derived/zenith_dark_4200m_log_sqm_as_vab.fits"


def main():
    with fits.open(COARSE,memmap=False,checksum=True) as h:
        wave=np.asarray(h[1].data["lam"],dtype=float)
        flux=np.asarray(h[1].data["flux"],dtype=float)
    oldmag=sky_r_ab(wave,flux)
    scale=10**(-.4*(21.07-oldmag))
    coarse=DepthModel();base=DepthModel()
    calc=JUSTExposureTimeCalculator(calc_mode="accurate")
    detail={}
    for i,arm in enumerate("BRZ"):
        setup=setup_arm(calc,"dark",i)
        b,_,_,_=apply_sky(wave,flux*scale,setup)
        b0=base.d[f"nominal_{arm}_sky"]
        valid=base.d[f"{arm}_valid"].astype(bool)&(b0>0)
        rel=b[valid]/b0[valid]-1
        s=calc._spectro
        coarse.d[f"nominal_{arm}_sky"]=b
        coarse.d[f"nominal_{arm}_sky_ref"]=np.maximum.reduce([np.r_[b[0],b[:-1]],b,np.r_[b[1:],b[-1]]])
        coarse.d[f"nominal_{arm}_stray"]=np.array(np.sum(b)*s.width[i]*s.pix[i]/s.sep[i]/len(b))
        detail[arm]={"median_abs_relative_pixel_difference":float(np.median(np.abs(rel))),
                     "p95_abs_relative_pixel_difference":float(np.quantile(np.abs(rel),.95)),
                     "integrated_sky_e_ratio":float(b[valid].sum()/b0[valid].sum())}
    depths={str(t):{"0p01nm_s":base.effective_time("dark",t),
                    "0p05nm_s":coarse.effective_time("dark",t)} for t in (1500,3000)}
    out={"status":"numerical input-grid sensitivity, not a sky-model uncertainty bound",
         "coarse_sky_sha256":sha(COARSE),"coarse_r_before_rescaling":oldmag,
         "coarse_grey_scale_to_desi_r21p07":scale,"per_arm":detail,
         "mapped_nominal_depths":depths,"script_sha256":sha(Path(__file__))}
    (HERE/"sky_sampling_sensitivity.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
