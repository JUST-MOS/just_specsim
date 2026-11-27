"""Map complete SUM-coadd Redrock anchors to the continuous candidate proxy."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from weighted_effective_model import WeightedDepthModel
from audit_weighted_model import sha

HERE = Path(__file__).resolve().parent


def main():
    audit = json.loads((HERE / "weighted_model_audit.json").read_text())
    redrock = json.loads((HERE / "nominal_independent_audit.json").read_text())
    if audit["status"] != "passed" or redrock["status"] != "passed":
        raise ValueError("required audits are not complete")
    for name, expected in audit["sha256"].items():
        if sha(HERE/name) != expected:
            raise ValueError(f"weighted model audit is stale: {name}")
    raw = pd.read_csv(HERE / "nominal_validated_thresholds.csv")
    curves = pd.read_csv(HERE / "nominal_success_curves.csv")
    m = WeightedDepthModel()
    rows = []
    for r in raw.itertuples():
        out = {"sample":r.sample, "criterion":r.criterion,
               "frozen_nominal_test_s":r.nominal_s,
               "weighted_nominal_eff_s":np.nan, "weighted_dark_s":np.nan,
               "weighted_bright_s":np.nan, "nominal_success":np.nan,
               "nominal_n":np.nan, "nominal_fraction":np.nan, "nominal_lower":np.nan,
               "ebv":m.default_ebv,
               "status":"unconfirmed in two nominal SUM-coadd tests"}
        if np.isfinite(r.nominal_s):
            selected = curves[(curves['sample']==r.sample)&(curves.actual_s==r.nominal_s)]
            if len(selected)!=1:
                raise ValueError("anchor does not map to one nominal test")
            x = selected.iloc[0]
            dark = m.equivalent_to_frozen_nominal("dark", r.nominal_s)
            bright = m.equivalent_to_frozen_nominal("bright", r.nominal_s)
            if not np.isclose(dark['weighted_nominal_s'],bright['weighted_nominal_s'],rtol=1e-12):
                raise ValueError("nominal calibration depends on observed sky")
            out.update({"weighted_nominal_eff_s":dark['weighted_nominal_s'],
                        "weighted_dark_s":dark['weighted_scene_s'],
                        "weighted_bright_s":bright['weighted_scene_s'],
                        "nominal_success":int(x.k_success), "nominal_n":int(x.n),
                        "nominal_fraction":x.success_fraction, "nominal_lower":x.lower,
                        "status":"proxy-converted from first passing nominal SUM-coadd test; weighted coadd unvalidated"})
        rows.append(out)
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "weighted_thresholds.csv", index=False)
    print(result.to_string(index=False))
    # Retain an explicitly indirect comparison based on the previous dark-sky
    # Redrock anchors, even if a new nominal test does not meet the criterion.
    old = pd.read_csv(HERE / "estimated_thresholds.csv")
    indirect = []
    for x in old.itertuples():
        goal = m.frozen.information("dark", x.dark_anchor_s)
        indirect.append({"sample":x.sample, "criterion":x.criterion,
                         "dark_redrock_anchor_s":x.dark_anchor_s,
                         "dark_success":x.dark_success, "n":x.n,
                         "dark_success_fraction":x.dark_success_fraction,
                         "dark_adjusted_lower":x.dark_lower,
                         "weighted_nominal_eff_s":m.time_for_information("nominal", goal),
                         "weighted_dark_s":m.time_for_information("dark", goal),
                         "weighted_bright_s":m.time_for_information("bright", goal),
                         "ebv":m.default_ebv,
                         "status":"indirect proxy from prior dark SUM-coadd test; weighted coadd unvalidated"})
    table = pd.DataFrame(indirect)
    table.to_csv(HERE / "indirect_weighted_thresholds.csv", index=False)
    print("\nIndirect prior-dark anchors:\n"+table.to_string(index=False))


if __name__ == "__main__":
    main()
