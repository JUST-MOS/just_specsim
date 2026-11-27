"""Compare the full original and extended Tractor scans on an eight-time family.

The three shorter times were selected after inspecting the original 750-s
bright-subset outcome. The original model has only the five original times;
both models use alpha=.05/8 for the comparison, conservatively accounting
for the full set of tested times. Neither scan estimates untested minima.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

import analyze_scan as original_analysis
import short_scan as short
from simulate_tractor import DATA, HERE, sha

OUT = HERE / "analysis_extended"
TIMES = sorted(short.TIMES + short.base.TIMES)
SKIES = short.SKIES
ALPHA = .05 / len(TIMES)


def minimum_k(n, threshold):
    return next(k for k in range(n + 1) if binom.sf(k - 1, n, threshold) <= ALPHA)


def summarize(old, new, sample):
    """Summaries for the frozen sample; counts include failures and missing fits."""
    groups = {"r_lt_19p5": sample[sample.r_dered < 19.5],
              "r_lt_20p5": sample[sample.r_dered < 20.5]}
    rows, thresholds = [], []
    for group, subset in groups.items():
        ids = set(subset.TARGETID.astype(int))
        n = len(ids)
        for sky in SKIES:
            for name, df, grid in (("original", old, short.base.TIMES), ("mixture", new, TIMES)):
                series = []
                for t in grid:
                    part = df[df.TARGETID.isin(ids) & (df.sky == sky) & (df.total_s == t)]
                    if len(part) != n or set(part.TARGETID.astype(int)) != ids:
                        raise ValueError(f"missing/duplicate scenario: {group}, {sky}, {name}, {t}")
                    k = int(part.success.sum())
                    q = int(part.quality_pass.sum())
                    lo, hi = original_analysis.purity_interval(k, q)
                    record = {"group": group, "sky": sky, "model": name, "time_s": t,
                              "n": n, "success": k, "success_fraction": k / n,
                              "cp95_lower_pointwise": original_analysis.lower(k, n, .05),
                              "cp95_lower_bonferroni_8": original_analysis.lower(k, n, ALPHA),
                              "quality_pass": q, "purity": k / q if q else np.nan,
                              "purity_cp95_lo": lo, "purity_cp95_hi": hi,
                              "no_result": int((part.outcome == "no_result").sum()),
                              "quality_fail": int((part.outcome == "quality_fail").sum()),
                              "wrong_redshift_quality_pass": int((part.outcome == "wrong_redshift").sum())}
                    if sum(record[x] for x in ("success", "no_result", "quality_fail", "wrong_redshift_quality_pass")) != n:
                        raise ValueError("outcome counts do not partition all test targets")
                    series.append(record)
                    rows.append(record)
                def first(column, target):
                    return next((v["time_s"] for v in series if v[column] >= target), None)
                t95 = first("cp95_lower_bonferroni_8", .95)
                thresholds.append({"group": group, "sky": sky, "model": name, "n": n,
                                   "time_s_point_95": first("success_fraction", .95),
                                   "time_s_adjusted_lower_95": t95,
                                   "time_s_point_90": first("success_fraction", .90),
                                   "time_s_adjusted_lower_90": first("cp95_lower_bonferroni_8", .90),
                                   "required_successes_95": minimum_k(n, .95),
                                   "status_95": "met_on_tested_grid" if t95 is not None else "not_confirmed_through_3600s",
                                   "minimum_tested_s": min(grid)})
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def main():
    progress_path = short.RUN / "progress.json"
    progress = json.loads(progress_path.read_text())
    expected = [(int(s), int(c)) for s, c in short.PLAN]
    completed = [(v["start"], v["galaxies"]) for v in progress["completed"]]
    if progress["status"] != "complete" or completed != expected:
        raise RuntimeError("short scan is incomplete")
    sample = short.base.preflight()
    oldfile = original_analysis.OLD_DIR / "individual_results.csv"
    oldmeta = json.loads((original_analysis.OLD_DIR / "analysis_manifest.json").read_text())
    if oldmeta["status"] != "complete" or oldmeta["outputs"][oldfile.name] != sha(oldfile):
        raise ValueError("old historical results changed")
    fivefile = HERE / "analysis/individual_results.csv"
    fivemeta = json.loads((HERE / "analysis/analysis_manifest.json").read_text())
    if fivemeta["status"] != "complete" or fivemeta["outputs"][fivefile.name] != sha(fivefile):
        raise ValueError("original five-time Tractor analysis changed")
    old = pd.read_csv(oldfile, float_precision="round_trip", dtype={"exposures_s": str})
    five = pd.read_csv(fivefile, float_precision="round_trip", dtype={"exposures_s": str})
    short.base.paths = short.paths
    short.base.RUN = short.RUN
    parts, files = [], []
    for start, count in expected:
        spectra = short.validate_spectra(start, count, sample)
        fitmeta = short.base.validate_fit(start, spectra)
        _, prefix = short.paths(start)
        csvfile = prefix.with_suffix(".csv")
        parts.append(pd.read_csv(csvfile, float_precision="round_trip", dtype={"exposures_s": str}))
        files.append({"start": start, "n": count, "spectra_sha256": sha(spectra),
                      "results_sha256": sha(csvfile), "fit_sha256": fitmeta["output_sha256"]})
    follow = pd.concat(parts, ignore_index=True)
    for label, df, expected_rows in (("old", old, 13320), ("five", five, 13320), ("short", follow, 7992)):
        if len(df) != expected_rows or df.TARGETID.nunique() != 1332:
            raise ValueError(f"{label} roster is incomplete")
        original_analysis.quality_flags(df)
    fields = ["TARGETID", "source_row", "VI_Z", "r_dered", "gr_dered", "r_bin", "z_bin",
              "spectral_type", "morphology", "SERSIC", "SHAPE_R", "SHAPE_E1", "SHAPE_E2",
              "FIBERFLUX_R", "MW_TRANSMISSION_R"]
    follow = follow.merge(sample[fields], on="TARGETID", validate="many_to_one", suffixes=("", "_catalog"))
    np.testing.assert_array_equal(follow.source_row, follow.source_row_catalog)
    np.testing.assert_allclose(follow.z_true_vi, follow.VI_Z, rtol=0, atol=1e-12)
    # Use reference TARGETID / sky / time: synthetic IDs can overlap across runs.
    keys = ["TARGETID", "sky", "total_s"]
    new = pd.concat([five, follow], ignore_index=True)
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("duplicate target/sky/time scenario")
    if set(new.total_s.unique()) != set(TIMES):
        raise ValueError("extended time grid mismatch")
    new["run_grid"] = np.where(new.total_s.isin(short.TIMES), "short", "original_five")
    new["scenario_key"] = (new.TARGETID.astype(str) + ":" + new.sky.astype(str) + ":" + new.total_s.astype(str))
    if not new.scenario_key.is_unique:
        raise ValueError("extended scenario key is not unique")
    curves, thresholds = summarize(old, new, sample)
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep new raw rows for later independent re-analysis, without changing
    # either parent run or the historical catalog.
    for name, table in (("individual_results.csv", new), ("success_curves.csv", curves),
                        ("exposure_thresholds.csv", thresholds)):
        table.to_csv(OUT / name, index=False)
    manifest = {"status": "complete", "n_test": len(sample), "n_old_rows": len(old),
                "n_five_rows": len(five), "n_short_rows": len(follow), "times_s": TIMES,
                "old_times_s": short.base.TIMES, "skies": SKIES, "alpha": ALPHA,
                "confidence": "one-sided exact Clopper-Pearson lower with Bonferroni over eight times per group and sky; no cross-group adjustment",
                "sequential_design": "125, 250 and 500 s were selected after the bright 750-s outcome; they were fixed before follow-up results",
                "join_key": "(TARGETID, sky, total_s), also scenario_key; SYNTHETIC_TARGETID is unique only within a run batch",
                "sample_sha256": sha(DATA / "frozen_sample.csv"), "old_results_sha256": sha(oldfile),
                "five_results_sha256": sha(fivefile), "short_plan_sha256": sha(short.RUN / "run_plan.json"),
                "short_progress_sha256": sha(progress_path), "files": files,
                "analysis_script_sha256": sha(Path(__file__)),
                "outputs": {p: sha(OUT / p) for p in ("individual_results.csv", "success_curves.csv", "exposure_thresholds.csv")}}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(thresholds.to_string(index=False), flush=True)
    print(curves.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
