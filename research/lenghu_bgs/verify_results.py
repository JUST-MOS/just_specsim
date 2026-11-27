"""Read-only recomputation of the delivered old/new redshift-success curves."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
STUDY = HERE/"results"
NEW = STUDY/"tractor_redrock_20261121/analysis_extended"
OLD = STUDY/"leader_195_205_20261116/extended_20p5/analysis_20261119"
DATA = STUDY/"leader_195_205_20261116/extended_20p5/input"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    delivery = json.loads((NEW/"delivery_manifest.json").read_text())
    for name, expected in delivery["outputs"].items():
        if sha(NEW/name) != expected:
            raise ValueError(f"Delivered result changed: {name}")
    sample = pd.read_csv(DATA/"frozen_sample.csv")
    sample = sample[sample.split == "test"]
    if len(sample) != 1332 or sample.TARGETID.nunique() != 1332:
        raise ValueError("Unexpected frozen test sample")
    sources = {"mixture":pd.read_csv(NEW/"individual_results.csv", dtype={"exposures_s":str}),
               "original":pd.read_csv(OLD/"individual_results.csv", dtype={"exposures_s":str})}
    curves = pd.read_csv(NEW/"success_curves.csv")
    max_delta = 0.
    for c in curves.itertuples(index=False):
        ids = set(sample.loc[sample.r_dered < (19.5 if c.group == "r_lt_19p5" else 20.5), "TARGETID"])
        d = sources[c.model]
        d = d[d.TARGETID.isin(ids) & (d.sky == c.sky) & (d.total_s == c.time_s)]
        if len(d) != len(ids) or d.TARGETID.nunique() != len(ids):
            raise ValueError("Changed denominator or repeated object")
        finite = np.isfinite(d.z_fit.to_numpy(float))
        ok = (finite & (d.ZWARN.to_numpy() == 0) & (d.DELTACHI2.to_numpy() > 40) &
              (np.abs(d.z_fit.to_numpy(float)-d.z_true_vi.to_numpy(float))/(1+d.z_true_vi.to_numpy(float)) < .0033))
        k, n = int(ok.sum()), len(d)
        lower = 0. if k == 0 else float(beta.ppf(.05/8, k, n-k+1))
        delta = abs(lower-c.cp95_lower_bonferroni_8)
        if k != c.success or n != c.n or delta > 2e-11:
            raise ValueError("Saved success rate or adjusted lower bound differs")
        max_delta = max(max_delta, delta)
    thresholds = pd.read_csv(NEW/"exposure_thresholds.csv")
    for r in thresholds.itertuples(index=False):
        d = curves[(curves.group == r.group)&(curves.sky == r.sky)&(curves.model == r.model)].sort_values("time_s")
        selected = d[d.cp95_lower_bonferroni_8 >= .95]
        expected = None if selected.empty else int(selected.time_s.iloc[0])
        saved = None if pd.isna(r.time_s_adjusted_lower_95) else int(r.time_s_adjusted_lower_95)
        if saved != expected:
            raise ValueError("Saved first passing time differs")
    result = {"status":"passed", "curves_checked":len(curves), "thresholds_checked":len(thresholds),
              "new_individual_rows":len(sources["mixture"]), "max_lower_bound_difference":max_delta,
              "delivery_files_checked":len(delivery["outputs"])}
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
