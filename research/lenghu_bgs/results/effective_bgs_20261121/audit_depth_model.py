"""Numerical physical-limit checks for the research depth model."""
import json
from pathlib import Path

import numpy as np

from effective_model import DepthModel
from build_analysis import sha

HERE=Path(__file__).resolve().parent


def main():
    # Identical scene spectra and zero dust must recover the nominal seconds.
    m=DepthModel()
    for arm in "BRZ":
        for suffix in ("response_no_dust","sky","sky_ref","stray"):
            m.d[f"dark_{arm}_{suffix}"]=m.d[f"nominal_{arm}_{suffix}"].copy()
    identical=[]
    for t in (300.,750.,1500.,2250.,3000.,3600.):
        got=m.effective_time("dark",t,ebv=0.)
        identical.append(abs(got-t))
    if max(identical)>1e-3:
        raise ValueError(f"identical scenes failed {identical}")
    # In a background-dominated, no-read, no-floor limit Q is linear in time.
    for arm in "BRZ":
        m.d[f"{arm}_read"]=np.array(0.)
        m.d[f"{arm}_dark"]=np.array(0.)
        m.d[f"{arm}_continuum"][:]=0.
        m.d[f"nominal_{arm}_sky_ref"][:]=0.
    q1=m.information("nominal",750.)
    q2=m.information("nominal",3000.)
    sky_limit=abs(q2/q1-4.)
    if sky_limit>1e-12:
        raise ValueError(f"sky-dominated limit is not linear: {sky_limit}")
    # Source-only transparency changes source and its Poisson variance; Q must
    # decrease for a lower transparency with the same ground sky.
    m=DepthModel()
    q_full=m.information("bright",1500.,transparency=1.)
    q_dim=m.information("bright",1500.,transparency=.8)
    if not 0<q_dim<q_full:
        raise ValueError("transparency ordering failed")
    # Increasing Galactic extinction must reduce information for all modeled arms.
    q_clear=m.information("dark",1500.,ebv=0.,per_arm=True)
    q_dust=m.information("dark",1500.,ebv=.1,per_arm=True)
    if np.any(q_dust>=q_clear) or np.any(q_dust<=0):
        raise ValueError("Galactic-extinction ordering failed")
    out={"status":"passed; numerical identities and limits only",
         "identical_scene_max_time_error_s":max(identical),
         "sky_dominated_4x_time_ratio_abs_error":sky_limit,
         "bright_0p8_to_1_transparency_Q_ratio":q_dim/q_full,
         "dust_0p1_to_0_Q_ratio_by_arm":(q_dust/q_clear).tolist(),
         "depth_inputs_sha256":sha(HERE/"depth_inputs.npz"),"script_sha256":sha(Path(__file__))}
    (HERE/"depth_limit_checks.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
