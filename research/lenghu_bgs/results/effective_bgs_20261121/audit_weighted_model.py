"""Numerical checks for the ideal weighted depth and the proxy calibration."""
from pathlib import Path
import hashlib
import json

import numpy as np

from weighted_effective_model import WeightedDepthModel
from weighted_coadd_sensitivity import information_weighted

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    m = WeightedDepthModel()
    times = [0., .001, 100., 749.999, 750., 750.001, 1499.999, 1500.,
             1500.001, 1817., 2250.001, 2999.999, 3000., 3000.001, 3436., 3600.]
    formula_error = 0.
    identity_error = 0.
    for scene in ("nominal", "dark", "bright"):
        qs = []
        for t in times:
            a = m.information(scene, t)
            b = information_weighted(m.frozen, scene, t)
            formula_error = max(formula_error, abs(a-b)/max(abs(a), 1e-50))
            qs.append(a)
        if np.any(np.diff(qs) < -1e-12 * max(qs)):
            raise ValueError(f"information is not monotone for {scene}")
    for t in times:
        identity_error = max(identity_error, abs(m.effective_time("nominal", t)-t))
    calibration_error = 0.
    cases = []
    for anchor in (1817., 3436.):
        goal = m.frozen.information("nominal", anchor)
        for scene in ("dark", "bright"):
            q = m.equivalent_to_frozen_nominal(scene, anchor)
            for actual, got_scene in ((q["weighted_nominal_s"], "nominal"),
                                      (q["weighted_scene_s"], scene)):
                got = m.information(got_scene, actual)
                calibration_error = max(calibration_error, abs(got-goal)/goal)
            cases.append({"frozen_nominal_actual_s":anchor, "scene":scene, **q})
    median_ebv_sensitivity = []
    for ebv in (.053360362, .053683195):
        for scene in ("dark", "bright"):
            q = m.equivalent_to_frozen_nominal(scene, 3436., ebv=ebv)
            median_ebv_sensitivity.append({"ebv":ebv,"scene":scene,**q})
    # Arbitrary equal-length exposures must reduce to the sum-coadd model.
    equal_error = 0.
    for scene in ("nominal", "dark", "bright"):
        for split in (250., 500., 750.):
            for n in (1, 2, 4):
                a = m.information(scene, split*n, max_single_s=split)
                b = m.frozen.information(scene, split*n, max_single_s=split)
                equal_error = max(equal_error, abs(a-b)/a)
    if formula_error > 1e-12 or identity_error > 1e-5 or calibration_error > 1e-9 or equal_error > 1e-12:
        raise ValueError("weighted model audit failed")
    result = {
      "status":"passed", "scope":"numerical identity and covariance formula checks only, not a Redrock or on-sky validation",
      "points_tested":times, "nominal_identity_max_abs_s":identity_error,
      "formula_max_relative_difference":formula_error,
      "equal_duration_max_relative_difference":equal_error,
      "proxy_calibration_max_relative_difference":calibration_error,
      "proxy_cases":cases,
      "median_ebv_sensitivity_for_frozen_3436_s":median_ebv_sensitivity,
      "sha256":{p.name:sha(p) for p in (HERE/"weighted_effective_model.py", HERE/"weighted_coadd_sensitivity.py", HERE/"depth_inputs.npz")}
    }
    (HERE/"weighted_model_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
