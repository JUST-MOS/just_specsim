"""Explore a reference-source effective-time transfer without Redrock.

This is a sensitivity calculation, not a recalibrated redshift-success model.
It reuses the frozen wavelength weights and sky/noise arrays, then replaces
only the circular r_eff=1.5 arcsec reference galaxy's aperture response by a
Tractor EXP or DEV mixture of the same catalog radius and zero ellipticity.
The reference SED remains at total BASS-r0=19.5.  An equal scalar information
does not guarantee equal Redrock success for a sample of different profiles.
"""
import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(ROOT / "py"), str(STUDY / "effective_bgs_20261121"),
                str(STUDY / "high_sersic_legacy_20261121")]
from just_etc import JUSTExposureTimeCalculator, ETC_py_optimized as engine
from just_etc.mixture_fiber_coupling import GaussianMixtureProfile, etc_mixture_fiber_fraction
from compare_tractor import TractorDR94Profiles
from weighted_effective_model import WeightedDepthModel


def main():
    existing = STUDY / "population_4200_camera_overlap/output/instrument_reference_4200.npz"
    with np.load(existing, allow_pickle=False) as z:
        inst = {k: z[k].copy() for k in z.files}
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, ebv=0., r_eff=0., decenter=.03,
                            sysfrac=.01, diffuse_stray=.02)
    calc._set_mode()
    calc._configure_spectro(1)
    tractor = TractorDR94Profiles()
    cases = [("dark", 500.), ("bright", 500.), ("dark", 750.), ("bright", 1500.)]
    baseline = WeightedDepthModel()
    def describe(model):
        mapping = {f"{scene}_{int(t)}": model.effective_time(scene, t) for scene, t in cases}
        mapping["nominal_600_to_dark_s"] = model.exposure_for_depth("dark", 600.)
        mapping["nominal_600_to_bright_s"] = model.exposure_for_depth("bright", 600.)
        for scene in ("dark", "bright"):
            recovered = model.effective_time(scene, mapping[f"nominal_600_to_{scene}_s"])
            if not np.isclose(recovered, 600., rtol=1e-8, atol=1e-5):
                raise ValueError(f"600-s inverse mapping failed for {scene}: {recovered}")
            teff = mapping[f"{scene}_500"]
            nom = model.information("nominal", teff, per_arm=True)
            obs = model.information(scene, 500., per_arm=True)
            mapping[f"{scene}_500_arm_q_ratio"] = (obs / nom).tolist()
        return mapping
    result = {"old_exponential": describe(baseline)}
    for morph in ("EXP", "DEV"):
        amplitudes, variances = tractor.profile(morph, 1. if morph == "EXP" else 4.)
        p = GaussianMixtureProfile(1.5, 0., 0., amplitudes, variances)
        model = WeightedDepthModel()
        geo = {}
        for scene, seeing in (("nominal", 1.1*(800./635.5)**(-.2)), ("dark", .8)):
            calc.set_obs_conditions(seeing_fwhm_800=seeing)
            obs = calc._make_obs()
            for arm in "BRZ":
                nodes = inst[f"{arm}_geo_wave_nodes_nm"]
                old = np.array([engine.gsGeometricThroughput(calc._spectro, obs, float(w),
                                1.5, .03, 0., 0) for w in nodes])
                new = etc_mixture_fiber_fraction(p, calc._spectro, obs, nodes,
                                                 decenter_arcsec=.03, field_angle_deg=0.)
                if not np.all(np.isfinite(old)) or np.any(old <= 0) or np.any(new <= 0):
                    raise ValueError("Invalid reference geometry")
                wave = model.d[f"{arm}_wave_nm"]
                ratio = np.interp(wave, nodes, new)/np.interp(wave, nodes, old)
                geo[f"{scene}_{arm}_median_ratio"] = float(np.median(ratio))
                model.d[f"{scene}_{arm}_response_no_dust"] *= ratio
                if scene == "dark":
                    model.d[f"bright_{arm}_response_no_dust"] *= ratio
        # These are within-scene identities, not validation against Redrock.
        for t in (125., 500., 750., 1500.):
            got = model.effective_time("nominal", t)
            if not np.isclose(got, t, rtol=1e-8, atol=1e-5):
                raise ValueError(f"Nominal identity check failed at {t}: {got}")
        result[f"tractor_{morph.lower()}"] = describe(model)
        result[f"tractor_{morph.lower()}_geometry"] = geo
    output = {"status":"exploratory equal-information transfer, not Redrock-calibrated",
        "nominal":"X=1, EBV=0, r sky=21.07 AB mag/arcsec2, seeing=1.1 arcsec at 635.5nm",
        "scenario":"X=1.2, seeing=0.8 arcsec at 800nm, EBV=0.053360362, existing 4200m skies",
        "galaxy_reference":"circular r_eff=1.5 arcsec, total BASS-r0=19.5",
        "note":"Idealized inverse-covariance weighted Q; common feature weights from 110 calibration spectra; no new spectra or Redrock fits.",
        "time_mappings": result}
    out = Path(__file__).resolve().parent / "exploratory_mappings.json"
    out.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
