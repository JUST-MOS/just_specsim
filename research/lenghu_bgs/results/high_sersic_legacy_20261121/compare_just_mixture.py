"""Compare catalog-consistent mixtures to the analytic JUST aperture model.

This evaluates only deterministic spatial coupling, not spectra or Redrock.
Each star's ellipticity and radial scale come from the frozen sample, while
the Gaussian coefficients are read as literal data from the retained DR9.4
reference source.  The previous analytic-model results remain untouched.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_tractor import HERE, PRIOR, ROOT, TractorDR94Profiles, sha
from just_etc import JUSTExposureTimeCalculator
from just_etc.mixture_fiber_coupling import GaussianMixtureProfile, etc_mixture_fiber_fraction


def main():
    data = pd.read_csv(PRIOR)
    model = TractorDR94Profiles()
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, field_angle=0., decenter=.03)
    calc._set_mode()
    calc._configure_spectro(1)
    sp = calc._spectro
    conditions = {"nom": 1.1 * (800. / 635.5)**(-.2), "survey": .8}
    observations = {}
    for name, seeing in conditions.items():
        calc.set_obs_conditions(seeing_fwhm_800=seeing)
        observations[name] = calc._make_obs()
    rows = []
    for i, row in enumerate(data.itertuples(index=False)):
        amp, var = model.profile(row.morphology, row.SERSIC)
        profile = GaussianMixtureProfile(row.SHAPE_R, row.SHAPE_E1, row.SHAPE_E2, amp, var)
        result = {"TARGETID": row.TARGETID}
        for condition, obs in observations.items():
            fraction = etc_mixture_fiber_fraction(profile, sp, obs, 623.1,
                                                  decenter_arcsec=.03, field_angle_deg=0.)
            result[f"just_{condition}_mixture_623p1"] = fraction
            result[f"just_{condition}_mixture_over_analytic"] = fraction / getattr(row, f"just_{condition}_new_623p1")
            result[f"just_{condition}_mixture_over_original"] = fraction / getattr(row, f"just_{condition}_old_623p1")
        rows.append(result)
        if (i + 1) % 250 == 0:
            print(f"Evaluated {i + 1}/{len(data)} galaxies", flush=True)
    comparison = data.merge(pd.DataFrame(rows), on="TARGETID", validate="one_to_one")
    comparison.to_csv(HERE / "just_mixture_coupling.csv", index=False)
    test = comparison[comparison.split == "test"]
    # Calibration rows have missing prior outcomes, so pandas stores the
    # boolean values as Python objects; bitwise ~ on that dtype is not logical
    # negation. Select explicit False values for the independent test split.
    failed_bright = test[(test.r_dered < 19.5) & (test.prior_1817_success == False)]  # noqa: E712
    if len(failed_bright) != 23:
        raise ValueError("unexpected frozen 1817-s bright-sample failure count")
    groups = {
        "test_all_r20p5": test,
        "test_r19p5": test[test.r_dered < 19.5],
        "test_faint_19p5_20p5": test[test.r_dered >= 19.5],
        "test_n_ge_5": test[test.SERSIC >= 5.],
        "test_r19p5_prior_1817_failure": failed_bright,
    }
    groups.update({f"test_{m}": test[test.morphology == m] for m in ("REX", "EXP", "DEV", "SER")})
    results = []
    for name, group in groups.items():
        record = {"group": name, "n": len(group)}
        for condition in conditions:
            for kind in ("analytic", "original"):
                col = f"just_{condition}_mixture_over_{kind}"
                for percentile, value in zip((16, 50, 84), np.percentile(group[col], (16, 50, 84))):
                    record[f"{col}_p{percentile}"] = float(value)
        results.append(record)
    pd.DataFrame(results).to_csv(HERE / "just_mixture_summary.csv", index=False)
    summary = {
        "groups": results,
        "n_all": len(data), "n_test": len(test),
        "wavelength_nm": 623.1, "seeing_fwhm_800_arcsec": conditions,
        "decenter_arcsec": .03, "fiber_diameter_arcsec": float(sp.fiber_ent_rad / sp.EFL[0] * 2 * 0.20626480624709636),
        "offset_angle": "along galaxy major axis",
        "profile": "continuous DR9.4 Tractor mixture, no Legacy image pixelization",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          (PRIOR, Path(__file__), Path(__file__).parent / "compare_tractor.py",
                           ROOT / "py/just_etc/mixture_fiber_coupling.py")},
        "redrock_run": False,
    }
    (HERE / "just_mixture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
