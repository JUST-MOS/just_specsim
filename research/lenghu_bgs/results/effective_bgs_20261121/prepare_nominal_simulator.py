"""Construct frozen nominal-condition inputs for the existing simulator."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
POP = STUDY / "population_4200_camera_overlap"
EXT = STUDY / "leader_195_205_20261116/extended_20p5"
OUT = HERE / "nominal_run/shared"
sys.path[:0] = [str(HERE), str(ROOT / "py"), str(POP)]
from build_analysis import sha, SEEING_NOMINAL_800
from effective_model import DepthModel
from just_etc import JUSTExposureTimeCalculator, ETC_py_optimized as eng


def main():
    meta = json.loads((HERE / "model_manifest.json").read_text())
    if meta["verification"]["status"].split(";")[0] != "passed":
        raise ValueError("effective-depth input verification failed")
    original = POP / "output/instrument_reference_4200.npz"
    if sha(original) != meta["input_sha256"][str(original)]:
        raise ValueError("frozen prior instrument changed")
    with np.load(original, allow_pickle=False) as h:
        inst = {k:h[k].copy() for k in h.files}
    model = DepthModel()
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=SEEING_NOMINAL_800, ebv=0., r_eff=0.,
                            decenter=.03, sysfrac=.01, diffuse_stray=.02)
    calc._set_mode(); calc._configure_spectro(1)
    obs, sp = calc._make_obs(), calc._spectro
    residual = {}
    for arm in "BRZ":
        w = inst[f"{arm}_wave_nm"]
        nodes = inst[f"{arm}_geo_wave_nodes_nm"]
        oldgeo = inst[f"{arm}_geo0_nodes"].copy()
        newgeo = np.array([eng.gsGeometricThroughput(sp,obs,float(x),0.,.03,0.,0)
                           for x in nodes])
        galgeo = np.array([eng.gsGeometricThroughput(sp,obs,float(x),1.5,.03,0.,0)
                           for x in nodes])
        if np.any(newgeo <= 0) or np.any(oldgeo <= 0):
            raise ValueError("invalid point-source geometry")
        ratio = np.interp(w,nodes,newgeo)/np.interp(w,nodes,oldgeo)
        atmratio = model.d[f"{arm}_nominal_atm"]/model.d[f"{arm}_scenario_atm"]
        inst[f"{arm}_flat_response_750_e"] *= ratio*atmratio
        inst[f"{arm}_geo0_nodes"] = newgeo
        fid_response = inst[f"{arm}_flat_response_750_e"] * (
            np.interp(w,nodes,galgeo)/np.interp(w,nodes,newgeo))
        reference = model.d[f"nominal_{arm}_response_no_dust"]
        residual[arm] = float(np.max(np.abs(fid_response/reference-1)))
        if residual[arm] > 1e-10:
            raise ValueError(f"nominal response differs from depth model in {arm}")
        inst[f"nominal_{arm}_sky_e"] = model.d[f"nominal_{arm}_sky"]
        inst[f"nominal_{arm}_sky_ref_e"] = model.d[f"nominal_{arm}_sky_ref"]
        inst[f"nominal_{arm}_stray_e"] = model.d[f"nominal_{arm}_stray"]
        inst[f"nominal_{arm}_mask"] = inst[f"dark_{arm}_mask"].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"instrument_reference_4200.npz",**inst)
    output = OUT/"instrument_reference_4200.npz"
    (OUT/"instrument_validation.json").write_text(json.dumps({"status":"passed",
        "instrument_sha256":sha(output), "prior_instrument_sha256":sha(original),
        "depth_inputs_sha256":sha(HERE/"depth_inputs.npz"),
        "reference_fiducial_response_max_relative_difference":residual,
        "scope":"nominal sky, X=1 atmosphere and seeing; numerical verification only"},indent=2)+"\n")
    for name in ("desi_camera_mapping.json","desi_camera_mapping.npz"):
        src = EXT/"shared"/name
        dst = OUT/name
        shutil.copyfile(src,dst)
        if sha(src) != sha(dst):
            raise ValueError("camera mapping copy differs")
    print(json.dumps({"status":"passed","instrument":str(output),
                      "sha256":sha(output),"fiducial_response_max_relative_difference":residual},indent=2))


if __name__ == "__main__":
    main()
