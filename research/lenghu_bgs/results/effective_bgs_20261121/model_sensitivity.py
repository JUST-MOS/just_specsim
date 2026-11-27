"""Test how the provisional time mapping changes with fiducial assumptions."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[2]
sys.path.insert(0,str(HERE))
from build_analysis import INSTRUMENT, SAMPLE, SEEING_NOMINAL_800, geometry, training_weights, sha
from effective_model import DepthModel
from just_etc import JUSTExposureTimeCalculator


def evaluate(model,kind,n=None):
    records=[]
    for anchor in (1500.,3000.):
        nominal=model.effective_time("dark",anchor)
        arm_q=model.information("dark",anchor,per_arm=True)
        records.append({"case":kind,"n_training":n,"dark_anchor_s":anchor,
                        "nominal_eff_s":nominal,
                        "bright_equal_information_s":model.exposure_for_depth("bright",nominal),
                        "dark_B_information_fraction":arm_q[0]/arm_q.sum(),
                        "dark_R_information_fraction":arm_q[1]/arm_q.sum(),
                        "dark_Z_information_fraction":arm_q[2]/arm_q.sum()})
    return records


def main():
    with np.load(INSTRUMENT,allow_pickle=False) as z:
        inst={k:z[k].copy() for k in z.files}
    cat=pd.read_csv(SAMPLE)
    train=cat[(cat.split=="calibration")&(cat.r_dered<19.5)&(cat.VI_Z>.13)&(cat.VI_Z<.37)]
    rows=evaluate(DepthModel(),"baseline_110_calibrators_rhalf1p5",len(train))
    for label in ("bpt_star_forming","weak_line_old","other_unclassified"):
        subset=train[train.spectral_type==label]
        if len(subset)<10:raise ValueError(f"insufficient calibration subset for {label}")
        alt,_=training_weights(inst,subset)
        model=DepthModel()
        for arm in "BRZ":
            for name in ("feature_weight","continuum","valid"):
                model.d[f"{arm}_{name}"]=alt[f"{arm}_{name}"]
        rows+=evaluate(model,"weight_"+label,len(subset))
    calc=JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8,ebv=0.,r_eff=0.,decenter=.03)
    calc._set_mode();calc._configure_spectro(1)
    for radius in (.5,3.):
        model=DepthModel()
        for arm in "BRZ":
            for sky,seeing in (("nominal",SEEING_NOMINAL_800),("dark",.8),("bright",.8)):
                base=geometry(calc,inst,arm,seeing,1.5)
                alt=geometry(calc,inst,arm,seeing,radius)
                model.d[f"{sky}_{arm}_response_no_dust"]*=alt/base
        rows+=evaluate(model,f"exponential_rhalf_{radius:.1f}_arcsec",len(train))
    for mag in (18.5,20.,20.5):
        model=DepthModel()
        # A common rescaling of all feature weights cancels from equal-Q
        # comparisons; change the fiducial continuum for source Poisson noise.
        factor=10**(-.4*(mag-19.5))
        for arm in "BRZ":model.d[f"{arm}_continuum"]*=factor
        rows+=evaluate(model,f"fiducial_total_r_{mag:.1f}",len(train))
    model=DepthModel()
    for arm in "BZ":model.d[f"{arm}_feature_weight"][:]=0.
    rows+=evaluate(model,"R_arm_only",len(train))
    out=pd.DataFrame(rows)
    out.to_csv(HERE/"model_sensitivity.csv",index=False)
    meta={"status":"deterministic sensitivity, not a calibrated uncertainty interval",
          "weight_training_n":len(train),"subsets":train.spectral_type.value_counts().to_dict(),
          "shape_scope":"circular exponential half-light radii 0.5, 1.5 and 3.0 arcsec",
          "magnitude_scope":"r=18.5, 19.5, 20.0, 20.5; common feature-amplitude scaling cancels, but source-Poisson term changes",
          "note":"Subtypes, magnitudes, sizes and R-only change the proxy, not galaxy outcomes; no fresh Redrock fit here.",
          "input_model_sha256":sha(HERE/"depth_inputs.npz"),"script_sha256":sha(Path(__file__))}
    (HERE/"model_sensitivity_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(out.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
