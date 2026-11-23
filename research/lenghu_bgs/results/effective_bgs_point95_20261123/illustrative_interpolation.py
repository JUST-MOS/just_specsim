"""Illustrate point-estimate 95% interpolation, without new Redrock fits.

Inputs are two independently noised, same-galaxy Redrock outcomes at 250 and
500 s for each sky. Linear interpolation is only a descriptive assumption.
The fitted nominal crossing is not directly observed and is not a calibrated
confidence interval or a scientifically established minimum.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "py"), str(STUDY / "effective_bgs_20261121"),
                str(STUDY / "high_sersic_legacy_20261121")]
from weighted_effective_model import WeightedDepthModel
from compare_tractor import TractorDR94Profiles
from just_etc import JUSTExposureTimeCalculator, ETC_py_optimized as engine
from just_etc.mixture_fiber_coupling import GaussianMixtureProfile, etc_mixture_fiber_fraction


def dev_model():
    with np.load(STUDY / "population_4200_camera_overlap/output/instrument_reference_4200.npz") as z:
        inst = {key: z[key].copy() for key in z.files}
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, ebv=0., r_eff=0., decenter=.03,
                            sysfrac=.01, diffuse_stray=.02)
    calc._set_mode()
    calc._configure_spectro(1)
    amps, variances = TractorDR94Profiles().profile("DEV", 4.)
    profile = GaussianMixtureProfile(1.5, 0., 0., amps, variances)
    model = WeightedDepthModel()
    for scene, seeing in (("nominal", 1.1*(800./635.5)**(-.2)), ("dark", .8)):
        calc.set_obs_conditions(seeing_fwhm_800=seeing)
        obs = calc._make_obs()
        for arm in "BRZ":
            nodes = inst[f"{arm}_geo_wave_nodes_nm"]
            old = np.array([engine.gsGeometricThroughput(calc._spectro, obs, float(w),
                            1.5, .03, 0., 0) for w in nodes])
            new = etc_mixture_fiber_fraction(profile, calc._spectro, obs, nodes,
                                             decenter_arcsec=.03, field_angle_deg=0.)
            wave = model.d[f"{arm}_wave_nm"]
            ratio = np.interp(wave, nodes, new)/np.interp(wave, nodes, old)
            model.d[f"{scene}_{arm}_response_no_dust"] *= ratio
            if scene == "dark":
                model.d[f"bright_{arm}_response_no_dust"] *= ratio
    return model


def main():
    path = STUDY / "tractor_redrock_20261121/analysis_extended/success_curves.csv"
    curves = pd.read_csv(path)
    df = curves[(curves.group == "r_lt_19p5") & (curves.model == "mixture")]
    model = dev_model()
    output = {"status": "illustrative interpolations, not a direct nominal Redrock result",
              "sample": "578 selected DESI EDR VI galaxies, r0<19.5, 0.1<zVI<0.4",
              "interpolation": "linear 250-500s and monotone PCHIP 125-750s; independent noise at each time",
              "reference_source": "Tractor DEV, circular 1.5 arcsec catalog scale, total r0=19.5",
              "scenario_ebv": model.default_ebv, "scenarios": {}}
    for sky in ("dark", "bright"):
        rows = df[(df.sky == sky) & df.time_s.isin([250, 500])].sort_values("time_s")
        if rows.time_s.tolist() != [250, 500] or len(rows) != 2 or not (rows.n == 578).all():
            raise ValueError("Unexpected input grid or sample size")
        fractions = rows.success_fraction.to_numpy(float)
        if not fractions[0] < .95 < fractions[1]:
            raise ValueError("95% is not bracketed by these observed point estimates")
        actual_95 = float(np.interp(.95, fractions, rows.time_s.to_numpy(float)))
        eff_endpoints = [model.effective_time(sky, float(x)) for x in rows.time_s]
        eff_by_actual = model.effective_time(sky, actual_95)
        eff_by_direct_interp = float(np.interp(.95, fractions, eff_endpoints))
        pchip_rows = df[(df.sky == sky) & df.time_s.isin([125, 250, 500, 750])].sort_values("time_s")
        if pchip_rows.time_s.tolist() != [125, 250, 500, 750]:
            raise ValueError("Unexpected PCHIP input grid")
        shape = PchipInterpolator(pchip_rows.time_s.to_numpy(float),
                                  pchip_rows.success_fraction.to_numpy(float))
        pchip_actual = float(brentq(lambda t: shape(t)-.95, 250., 500.))
        pchip_eff = model.effective_time(sky, pchip_actual)
        output["scenarios"][sky] = {"counts": rows.success.to_list(),
            "times_s": rows.time_s.to_list(), "success_fraction": fractions.tolist(),
            "effective_times_s": eff_endpoints,
            "interpolated_actual_s": actual_95,
            "effective_from_interpolated_actual_s": eff_by_actual,
            "interpolated_effective_s": eff_by_direct_interp,
            "pchip_actual_s": pchip_actual, "pchip_effective_s": pchip_eff}
    print(json.dumps(output, indent=2))
    (Path(__file__).resolve().parent / "illustrative_interpolation.json").write_text(
        json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
