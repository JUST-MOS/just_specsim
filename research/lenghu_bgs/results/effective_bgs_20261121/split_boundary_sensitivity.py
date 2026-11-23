"""Illustrate EBV and read-count sensitivity at a fixed nominal depth.

This is a diagnostic of the frozen sum-coadd model, not a new Redrock run.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from effective_model import DepthModel

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "leader_195_205_20261116/extended_20p5/input/frozen_sample.csv"


def main():
    sample = pd.read_csv(DATA)
    sample = sample[sample.split == "test"]
    bright = sample[sample.r_dered < 19.5]
    if len(sample) != 1332 or len(bright) != 578:
        raise ValueError("unexpected sample identity")
    model = DepthModel()
    target_nominal = 3436.0
    target_q = model.information("nominal", target_nominal)
    cases = {}
    for label, ebv in (("bgs_bright_median", float(bright.EBV.median())),
                       ("expanded_median", float(sample.EBV.median())),
                       ("expanded_5th", float(sample.EBV.quantile(.05))),
                       ("expanded_95th", float(sample.EBV.quantile(.95)))):
        cases[label] = {
            "ebv_mag": ebv,
            "dark_equal_q_s": model.exposure_for_depth("dark", target_nominal, ebv=ebv),
            "bright_equal_q_s": model.exposure_for_depth("bright", target_nominal, ebv=ebv),
            "dark_q_at_3000_over_target": model.information("dark", 3000, ebv=ebv) / target_q,
            "dark_q_just_after_3000_over_target": model.information("dark", 3000.001, ebv=ebv) / target_q,
        }
    out = {
        "status": "passed",
        "note": "Sensitivity of fixed sum-coadd inversion; not a new Redrock success measurement or physical confidence interval.",
        "target_nominal_s": target_nominal,
        "maximum_single_exposure_s": 750,
        "cases": cases,
    }
    (HERE / "split_boundary_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
