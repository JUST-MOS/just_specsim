"""Continuous candidate JUST BGS effective depth using optimal exposure weights.

This research candidate is NOT the model that generated the Redrock spectra.
At every wavelength, independent photon/read noise and a sky residual that is
fully correlated between exposures form an exposure covariance matrix.  The
inverse-covariance source-amplitude information is evaluated analytically.
The observing conditions are stationary within each of the three frozen
scenes.  Wavelengths are treated independently, as in the existing simulator.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import brentq

from effective_model import ARM_NAMES, DepthModel, T0


class WeightedDepthModel:
    """Single-scene, exposure-weighted information with a monotone reference.

    Rates and weights come from the frozen scientific input bundle.  Each
    exposure is at most ``max_single_s``; all but the last have that length.
    The source and sky rates are assumed fixed during every exposure.
    """

    def __init__(self):
        self.frozen = DepthModel()
        self.d = self.frozen.d
        self.default_ebv = self.frozen.default_ebv

    def information(self, scene, time_s, *, ebv=None, transparency=1.,
                    max_single_s=T0, per_arm=False):
        if scene not in ("nominal", "dark", "bright"):
            raise ValueError("scene must be nominal, dark, or bright")
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("time must be nonnegative and finite")
        if not np.isfinite(max_single_s) or not 0 < max_single_s <= T0:
            raise ValueError("max_single_s must be in (0, 750]")
        if not np.isfinite(transparency) or not 0 < transparency <= 1:
            raise ValueError("transparency must be in (0, 1]")
        if ebv is None:
            ebv = 0. if scene == "nominal" else self.default_ebv
        if not np.isfinite(ebv) or ebv < 0:
            raise ValueError("E(B-V) must be finite and nonnegative")
        if time_s == 0:
            return np.zeros(3) if per_arm else 0.
        n = math.floor(time_s / max_single_s)
        f_full = max_single_s / T0
        f_last = (time_s - n * max_single_s) / T0
        parts = []
        for arm in ARM_NAMES:
            d = self.d
            response = d[f"{scene}_{arm}_response_no_dust"] * transparency
            response = response * 10**(-.4 * d[f"{arm}_kgal"] * ebv)
            linear = (response * d[f"{arm}_continuum"] + d[f"{scene}_{arm}_sky"]
                      + .02 * float(d[f"{scene}_{arm}_stray"]) + float(d[f"{arm}_dark"]))
            read = float(d[f"{arm}_read"])
            systematic = .01 * d[f"{scene}_{arm}_sky_ref"]
            a = n * f_full**2 / (f_full * linear + read)
            if f_last > 0:
                a = a + f_last**2 / (f_last * linear + read)
            valid = d[f"{arm}_valid"].astype(bool)
            values = d[f"{arm}_feature_weight"] * response**2 * a / (1 + systematic**2 * a)
            if np.any(~np.isfinite(values[valid])) or np.any(values[valid] < 0):
                raise ValueError("invalid information")
            parts.append(float(np.sum(values[valid])))
        return np.asarray(parts) if per_arm else float(sum(parts))

    def time_for_information(self, scene, goal, *, ebv=None, transparency=1.,
                             max_single_s=T0, max_time_s=12000.):
        if not np.isfinite(goal) or goal < 0 or not np.isfinite(max_time_s) or max_time_s <= 0:
            raise ValueError("invalid information goal or maximum time")
        if goal == 0:
            return 0.
        def residual(t):
            return self.information(scene, t, ebv=ebv, transparency=transparency,
                                    max_single_s=max_single_s) - goal
        if residual(max_time_s) < 0:
            return float("nan")
        return float(brentq(residual, 0., max_time_s, xtol=1e-8, rtol=1e-12))

    def effective_time(self, scene, actual_s, **kwargs):
        """Weighted reference seconds at equal idealized information."""
        goal = self.information(scene, actual_s, **kwargs)
        return self.time_for_information("nominal", goal)

    def exposure_for_depth(self, scene, nominal_s, **kwargs):
        """Weighted-scene shutter seconds for the weighted nominal depth."""
        goal = self.information("nominal", nominal_s)
        return self.time_for_information(scene, goal, **kwargs)

    def equivalent_to_frozen_nominal(self, scene, frozen_nominal_s, **kwargs):
        """Proxy-map a nominal SUM-coadd Redrock test onto weighted conditions.

        Equality of this scalar information is not independent validation of
        equal redshift success for the different extraction/coadd algorithm.
        """
        goal = self.frozen.information("nominal", frozen_nominal_s)
        return {"weighted_nominal_s": self.time_for_information("nominal", goal),
                "weighted_scene_s": self.time_for_information(scene, goal, **kwargs)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", choices=("nominal", "dark", "bright"))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--actual-s", type=float)
    group.add_argument("--target-eff-s", type=float)
    group.add_argument("--frozen-nominal-s", type=float)
    p.add_argument("--ebv", type=float)
    p.add_argument("--transparency", type=float, default=1.)
    p.add_argument("--max-single-s", type=float, default=T0)
    args = p.parse_args()
    model = WeightedDepthModel()
    opts = {"ebv": args.ebv, "transparency": args.transparency,
            "max_single_s": args.max_single_s}
    if args.actual_s is not None:
        result = {"weighted_t_eff_s": model.effective_time(args.scene, args.actual_s, **opts)}
    elif args.target_eff_s is not None:
        result = {"weighted_shutter_s": model.exposure_for_depth(args.scene, args.target_eff_s, **opts)}
    else:
        result = model.equivalent_to_frozen_nominal(args.scene, args.frozen_nominal_s, **opts)
        result["calibration"] = "equal proxy information only; weighted Redrock success has not been tested"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
