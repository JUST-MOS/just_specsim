"""Independently regroup raw per-object results and check exact binomial bounds."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE / "extended_20p5"
OUT = ROOT / "analysis_20261119"
RUN = ROOT / "run"
DATA = ROOT / "input"
OLD = HERE.parent / "population_4200_camera_overlap/output/test_individual_results.csv"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = json.loads((OUT / "analysis_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("main analysis is incomplete")
    for name, digest in manifest["outputs"].items():
        if sha(OUT / name) != digest:
            raise ValueError(f"analysis output hash mismatch: {name}")
    sample = pd.read_csv(DATA / "frozen_sample.csv")
    sample = sample[sample.split == "test"].copy()
    if sha(DATA / "frozen_sample.csv") != manifest["sample_sha256"] or len(sample) != 1332:
        raise ValueError("sample mismatch")
    plan = json.loads((RUN / "run_plan.json").read_text())
    cols = ["TARGETID", "sky", "total_s", "z_fit", "z_true_vi", "ZWARN", "DELTACHI2"]
    raw = [pd.read_csv(OLD, usecols=cols)]
    for batch in plan["batches"]:
        raw.append(pd.read_csv(RUN / f"test_{batch['start']:04d}_redrock.csv", usecols=cols))
    raw = pd.concat(raw, ignore_index=True)
    if len(raw) != 13320 or raw.duplicated(["TARGETID", "sky", "total_s"]).any():
        raise ValueError("raw scenario rows are incomplete or duplicated")
    raw["returned_new"] = np.isfinite(raw.z_fit)
    raw["quality_new"] = raw.returned_new & (raw.ZWARN == 0) & (raw.DELTACHI2 > 40)
    raw["correct_new"] = raw.returned_new & (np.abs(raw.z_fit - raw.z_true_vi) / (1 + raw.z_true_vi) < .0033)
    raw["success_new"] = raw.quality_new & raw.correct_new
    curves = pd.read_csv(OUT / "success_curves.csv")
    results = pd.read_csv(OUT / "individual_results.csv")
    merged = raw.merge(results[["TARGETID", "sky", "total_s", "returned", "quality_pass", "truth_pass", "success"]],
                       on=["TARGETID", "sky", "total_s"], validate="one_to_one")
    for a, b in (("returned_new", "returned"), ("quality_new", "quality_pass"),
                 ("correct_new", "truth_pass"), ("success_new", "success")):
        if not np.array_equal(merged[a], merged[b]):
            raise ValueError(f"raw result flag mismatch: {a}")
    sample["fiber_r"] = 22.5 - 2.5 * np.log10(sample.FIBERFLUX_R / sample.MW_TRANSMISSION_R)
    expected_groups = {("all", "all"): sample}
    for field, kind in (("r_bin", "magnitude"), ("z_bin", "redshift"), ("spectral_type", "spectral_type")):
        for val in sample[field].unique():
            expected_groups[(kind, str(val))] = sample[sample[field] == val]
    for r in sample.r_bin.unique():
        for z in sample.z_bin.unique():
            for spec in sample.spectral_type.unique():
                sub = sample[(sample.r_bin == r) & (sample.z_bin == z) & (sample.spectral_type == spec)]
                if len(sub):
                    expected_groups[("joint", "|".join((r, z, spec)))] = sub
        for field, cut, kind in (("gr_dered", 1.009976148605347, "observable_color"),
                                 ("fiber_r", 20.5899125097701, "observable_fiber_r")):
            for label, condition in (("low_or_equal", sample[field] <= cut), ("high", sample[field] > cut)):
                sub = sample[(sample.r_bin == r) & condition]
                if len(sub):
                    expected_groups[(kind, f"{r}|{label}")] = sub
    if set(zip(curves.group_type, curves.group)) != set(expected_groups):
        raise ValueError("reported groups differ from independently constructed groups")
    max_difference = 0.0
    for row in curves.itertuples(index=False):
        ids = set(expected_groups[(row.group_type, row.group)].TARGETID)
        part = raw[(raw.TARGETID.isin(ids)) & (raw.sky == row.sky) & (raw.total_s == row.time_s)]
        k, n, q = int(part.success_new.sum()), len(part), int(part.quality_new.sum())
        no_result = int((~part.returned_new).sum())
        quality_fail = int((part.returned_new & ~part.quality_new).sum())
        wrong = int((part.quality_new & ~part.correct_new).sum())
        if (set(part.TARGETID) != ids or (n, k, q, no_result, quality_fail, wrong) !=
            (row.n, row.k_success, row.quality_pass, row.no_result, row.quality_fail,
             row.wrong_redshift_quality_pass)):
            raise ValueError(f"count mismatch for {row.group_type} {row.group} {row.sky} {row.time_s}")
        if not math.isclose(row.success_fraction, k / n, rel_tol=0, abs_tol=1e-14):
            raise ValueError("success fraction mismatch")
        for confidence, bound in ((.95, row.cp95_lower_pointwise),
                                  (.99, row.cp95_lower_bonferroni_times)):
            check = binomtest(k, n, alternative="greater").proportion_ci(
                confidence_level=confidence, method="exact").low
            max_difference = max(max_difference, abs(check - bound))
    if max_difference > 1e-8:
        raise ValueError("exact-binomial lower bound mismatch")
    thresholds = pd.read_csv(OUT / "exposure_thresholds.csv")
    for row in thresholds.itertuples(index=False):
        part = curves[(curves.group_type == row.group_type) & (curves.group == row.group) &
                      (curves.sky == row.sky)].sort_values("time_s")
        if len(part) != 5:
            raise ValueError("threshold missing time points")
        for field, column in (("time_s_point", "success_fraction"),
                              ("time_s_pointwise_lower", "cp95_lower_pointwise"),
                              ("time_s_adjusted_lower", "cp95_lower_bonferroni_times")):
            hits = part[part[column] >= row.target]
            expected = int(hits.time_s.iloc[0]) if len(hits) else None
            got = getattr(row, field)
            if (expected is None and not pd.isna(got)) or (expected is not None and got != expected):
                raise ValueError(f"threshold mismatch {row.group_type} {row.group} {row.sky} {field}")
    summary = {"status": "passed", "n_galaxies": len(sample), "n_scenario_rows": len(raw),
               "n_groups": len(expected_groups), "n_curve_rows": len(curves),
               "n_threshold_rows": len(thresholds), "max_cp_lower_abs_difference": max_difference,
               "scope": "Recomputed raw classifications, target membership, group counts, first sampled thresholds and independent scipy.stats.binomtest exact one-sided intervals; this is not validation of physical model assumptions.",
               "analysis_manifest_sha256": sha(OUT / "analysis_manifest.json"),
               "audit_script_sha256": sha(Path(__file__))}
    (OUT / "independent_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
