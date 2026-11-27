"""Illustrate an ideal inverse-covariance coadd; this does not alter the mocks.

All exposures in a scene have the same stationary rates. Their independent
photon/read variances are diagonal and the 1% sky residual is fully correlated
across exposures. Q for the ideal source-amplitude estimator follows from the
Sherman-Morrison identity. Recorded units follow effective_model.py.
"""
from pathlib import Path
import json
import math

import numpy as np
from scipy.optimize import brentq

from effective_model import ARM_NAMES, DepthModel, T0

HERE = Path(__file__).resolve().parent


def information_weighted(model, scene, time_s, ebv=None):
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("time_s must be finite and nonnegative")
    if ebv is None:
        ebv = 0.0 if scene == "nominal" else model.default_ebv
    if not np.isfinite(ebv) or ebv < 0:
        raise ValueError("ebv must be finite and nonnegative")
    if time_s == 0:
        return 0.0
    n_full = math.floor(time_s / T0)
    remainder = (time_s - n_full * T0) / T0
    q = 0.0
    for arm in ARM_NAMES:
        d = model.d
        response = d[f"{scene}_{arm}_response_no_dust"] * 10 ** (-.4 * d[f"{arm}_kgal"] * ebv)
        poisson = (response * d[f"{arm}_continuum"] + d[f"{scene}_{arm}_sky"]
                   + .02 * float(d[f"{scene}_{arm}_stray"]) + float(d[f"{arm}_dark"]))
        read = float(d[f"{arm}_read"])
        sky_systematic = .01 * d[f"{scene}_{arm}_sky_ref"]
        a = n_full / (poisson + read)
        if remainder > 0:
            a = a + remainder**2 / (remainder * poisson + read)
        valid = d[f"{arm}_valid"].astype(bool)
        value = d[f"{arm}_feature_weight"] * response**2 * a / (1 + sky_systematic**2 * a)
        if np.any(~np.isfinite(value[valid])):
            raise ValueError("nonfinite information")
        q += float(np.sum(value[valid]))
    return q


def shutter(model, scene, nominal_s, ebv):
    goal = information_weighted(model, "nominal", nominal_s)
    f = lambda t: information_weighted(model, scene, t, ebv) - goal
    if f(12000.) < 0:
        return None
    return brentq(f, 0., 12000., xtol=1e-8)


def main():
    model = DepthModel()
    # Equal-duration subexposures recover the frozen sum-coadd formula.
    max_rel = 0.0
    for scene in ("nominal", "dark", "bright"):
        for time_s in (750., 1500., 2250., 3000.):
            a = model.information(scene, time_s)
            b = information_weighted(model, scene, time_s)
            max_rel = max(max_rel, abs(a - b) / a)
    if max_rel > 1e-12:
        raise ValueError("equal-exposure model identity failed")
    ebvs = [0.053360361999999995, 0.053683195]
    for scene in ("nominal", "dark", "bright"):
        for boundary in (750., 1500., 2250., 3000.):
            values = [information_weighted(model, scene, boundary + delta)
                      for delta in (-.001, 0., .001)]
            if not values[0] <= values[1] <= values[2]:
                raise ValueError("weighted information unexpectedly decreases")
    q_left = information_weighted(model, "dark", 3000., ebvs[1])
    q_right = information_weighted(model, "dark", 3000.001, ebvs[1])
    # Independently compare the scalar Sherman-Morrison expression to a
    # direct five-exposure covariance matrix inversion in selected pixels.
    fractions = np.array([1., 1., 1., 1., 436. / T0])
    direct_max_rel = 0.0
    for scene in ("nominal", "dark", "bright"):
        for arm in ARM_NAMES:
            d = model.d
            ebv = 0.0 if scene == "nominal" else model.default_ebv
            response = d[f"{scene}_{arm}_response_no_dust"] * 10 ** (-.4 * d[f"{arm}_kgal"] * ebv)
            poisson = (response * d[f"{arm}_continuum"] + d[f"{scene}_{arm}_sky"]
                       + .02 * float(d[f"{scene}_{arm}_stray"]) + float(d[f"{arm}_dark"]))
            read = float(d[f"{arm}_read"])
            sky_sys = .01 * d[f"{scene}_{arm}_sky_ref"]
            valid = np.flatnonzero(d[f"{arm}_valid"].astype(bool))
            for pixel in valid[np.linspace(0, len(valid)-1, 20).astype(int)]:
                c = (np.diag(poisson[pixel] * fractions + read)
                     + sky_sys[pixel]**2 * np.outer(fractions, fractions))
                direct = fractions @ np.linalg.solve(c, fractions)
                a = np.sum(fractions**2 / (fractions * poisson[pixel] + read))
                formula = a / (1 + sky_sys[pixel]**2 * a)
                direct_max_rel = max(direct_max_rel, abs(formula-direct) / direct)
    if direct_max_rel > 1e-12:
        raise ValueError("weighted covariance formula mismatch")
    cases = []
    for nominal_s in (1817., 3436.):
        q_opt = information_weighted(model, "nominal", nominal_s)
        q_frozen = model.information("nominal", nominal_s)
        for ebv in ebvs:
            cases.append({
                "nominal_s": nominal_s, "ebv_mag": ebv,
                "nominal_weighted_over_frozen_q": q_opt / q_frozen,
                "dark_weighted_s": shutter(model, "dark", nominal_s, ebv),
                "bright_weighted_s": shutter(model, "bright", nominal_s, ebv),
                "dark_frozen_s": model.exposure_for_depth("dark", nominal_s, ebv=ebv),
                "bright_frozen_s": model.exposure_for_depth("bright", nominal_s, ebv=ebv),
            })
    out = {
        "status": "passed",
        "interpretation": "Ideal inverse-covariance weighted stationary-scene sensitivity, not used to generate the Redrock spectra and not validated against Redrock.",
        "equal_duration_relative_error_max": max_rel,
        "direct_covariance_relative_error_max": direct_max_rel,
        "weighted_q_just_after_over_at_3000": q_right / q_left,
        "cases": cases,
    }
    (HERE / "weighted_coadd_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
