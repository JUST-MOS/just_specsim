"""Analyze a complete paired old/new five-time Redrock scan.

The denominator is the frozen test roster, including missing Redrock results.
Thresholds are first tested time points; confidence is one-sided exact
Clopper-Pearson with Bonferroni alpha=.05/5 for each sample and sky.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

import run_scan as run
from simulate_tractor import DATA, EXT, HERE, sha

OLD_DIR = EXT / "analysis_20261119"
OUT = HERE / "analysis"


def lower(k, n, alpha):
    if not 0 <= k <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid binomial arguments")
    return 0. if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def purity_interval(k, quality):
    """Two-sided 95% exact interval among quality-selected redshifts."""
    if quality == 0:
        return np.nan, np.nan
    lo = 0. if k == 0 else float(beta.ppf(.025, k, quality-k+1))
    hi = 1. if k == quality else float(beta.ppf(.975, k+1, quality-k))
    return lo, hi


def quality_flags(rows):
    returned = np.isfinite(rows.z_fit.to_numpy(float))
    quality = returned & (rows.ZWARN == 0) & (rows.DELTACHI2 > 40)
    truth = returned & (np.abs(rows.z_fit - rows.z_true_vi) / (1 + rows.z_true_vi) < .0033)
    success = quality & truth
    for computed, column in ((returned, "returned"), (quality, "quality_pass"),
                             (truth, "truth_pass"), (success, "success")):
        np.testing.assert_array_equal(computed, rows[column])


def calculate(old, new, sample, times=run.TIMES, skies=run.SKIES):
    """Return curves, first-passing grid times and paired outcome counts."""
    groups = {"r_lt_19p5": sample[sample.r_dered < 19.5],
              "r_lt_20p5": sample[sample.r_dered < 20.5]}
    curves, thresholds, pairs = [], [], []
    keys = ["TARGETID", "sky", "total_s"]
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("duplicated target/sky/time scenario")
    pair = old[keys + ["success"]].merge(new[keys + ["success"]], on=keys,
                    how="outer", validate="one_to_one", suffixes=("_old", "_new"), indicator=True)
    if not (pair._merge == "both").all():
        raise ValueError("old/new scenario coverage differs")
    for group, subset in groups.items():
        ids = set(subset.TARGETID.astype(int))
        n = len(ids)
        if not n:
            raise ValueError("empty comparison group")
        for sky in skies:
            series = {"original": [], "mixture": []}
            for total in times:
                for label, frame in (("original", old), ("mixture", new)):
                    x = frame[(frame.TARGETID.isin(ids)) & (frame.sky == sky) & (frame.total_s == total)]
                    if len(x) != n or set(x.TARGETID.astype(int)) != ids:
                        raise ValueError(f"denominator/coverage mismatch: {group} {sky} {total} {label}")
                    k, q = int(x.success.sum()), int(x.quality_pass.sum())
                    purity_lo, purity_hi = purity_interval(k, q)
                    entry = {"group": group, "sky": sky, "time_s": int(total), "model": label,
                             "n": n, "success": k, "success_fraction": k/n,
                             "cp95_lower_pointwise": lower(k, n, .05),
                             "cp95_lower_bonferroni_5": lower(k, n, .01),
                             "quality_pass": q, "purity": k/q if q else np.nan,
                             "purity_cp95_lo": purity_lo, "purity_cp95_hi": purity_hi,
                             "no_result": int((x.outcome == "no_result").sum()),
                             "quality_fail": int((x.outcome == "quality_fail").sum()),
                             "wrong_redshift_quality_pass": int((x.outcome == "wrong_redshift").sum())}
                    if k + entry["no_result"] + entry["quality_fail"] + entry["wrong_redshift_quality_pass"] != n:
                        raise ValueError("outcomes do not partition the denominator")
                    curves.append(entry)
                    series[label].append(entry)
                pp = pair[(pair.TARGETID.isin(ids)) & (pair.sky == sky) & (pair.total_s == total)]
                old_ok, new_ok = pp.success_old.to_numpy(bool), pp.success_new.to_numpy(bool)
                pair_entry = {"group": group, "sky": sky, "time_s": int(total), "n": len(pp),
                    "old_fail_new_success": int((~old_ok & new_ok).sum()),
                    "old_success_new_fail": int((old_ok & ~new_ok).sum()),
                    "both_success": int((old_ok & new_ok).sum()), "both_fail": int((~old_ok & ~new_ok).sum())}
                if sum(pair_entry[k] for k in ("old_fail_new_success", "old_success_new_fail", "both_success", "both_fail")) != n:
                    raise ValueError("paired outcomes do not partition the denominator")
                pairs.append(pair_entry)
            for label, values in series.items():
                def first(column):
                    return next((v["time_s"] for v in values if v[column] >= .95), None)
                adjusted = first("cp95_lower_bonferroni_5")
                thresholds.append({"group": group, "sky": sky, "model": label, "n": n,
                    "time_s_point": first("success_fraction"),
                    "time_s_pointwise_lower": first("cp95_lower_pointwise"),
                    "time_s_adjusted_lower": adjusted,
                    "status_adjusted": "met_on_grid" if adjusted is not None else "not_confirmed_through_3600s"})
    return pd.DataFrame(curves), pd.DataFrame(thresholds), pd.DataFrame(pairs)


def main():
    sample = run.preflight()
    progress = json.loads((run.RUN / "progress.json").read_text())
    expected = [(int(start), int(count)) for start, count in run.PLAN]
    completed = [(v["start"], v["galaxies"]) for v in progress["completed"]]
    if progress["status"] != "complete" or completed != expected:
        raise RuntimeError("full scan is incomplete")
    old_path = OLD_DIR / "individual_results.csv"
    old_meta = json.loads((OLD_DIR / "analysis_manifest.json").read_text())
    if old_meta["status"] != "complete" or sha(old_path) != old_meta["outputs"]["individual_results.csv"]:
        raise ValueError("frozen original results changed")
    old = pd.read_csv(old_path, float_precision="round_trip")
    new_parts, files = [], []
    for start, count in expected:
        spectra = run.validate_spectra(start, count, sample)
        meta = run.validate_fit(start, spectra)
        _, prefix = run.paths(start)
        csvfile = prefix.with_suffix(".csv")
        new_parts.append(pd.read_csv(csvfile, float_precision="round_trip"))
        files.append({"start": start, "galaxies": count, "spectra_sha256": sha(spectra),
                      "redrock_fits_sha256": meta["output_sha256"], "redrock_csv_sha256": sha(csvfile)})
    new = pd.concat(new_parts, ignore_index=True)
    for label, rows in (("old", old), ("new", new)):
        if len(rows) != 13320 or rows.TARGETID.nunique() != 1332:
            raise ValueError(f"{label} result coverage incorrect")
        quality_flags(rows)
    # Keep the derived per-target table useful for future magnitude, shape and
    # spectral-type analyses without modifying the frozen source catalog.
    fields = ["TARGETID", "source_row", "VI_Z", "r_dered", "gr_dered", "r_bin", "z_bin",
              "spectral_type", "morphology", "SERSIC", "SHAPE_R", "SHAPE_E1", "SHAPE_E2",
              "FIBERFLUX_R", "MW_TRANSMISSION_R"]
    new = new.merge(sample[fields], on="TARGETID", validate="many_to_one", suffixes=("", "_catalog"))
    np.testing.assert_array_equal(new.source_row, new.source_row_catalog)
    np.testing.assert_allclose(new.z_true_vi, new.VI_Z, rtol=0, atol=1e-12)
    curves, thresholds, pairs = calculate(old, new, sample)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("individual_results.csv", new), ("success_curves.csv", curves),
                        ("exposure_thresholds.csv", thresholds), ("paired_outcomes.csv", pairs)):
        value.to_csv(OUT / name, index=False)
    result = {"status": "complete", "n_test": len(sample), "n_old_rows": len(old), "n_new_rows": len(new),
              "times_s": run.TIMES, "skies": run.SKIES, "files": files,
              "sample_sha256": sha(DATA / "frozen_sample.csv"), "old_results_sha256": sha(old_path),
              "run_plan_sha256": sha(run.RUN / "run_plan.json"), "progress_sha256": sha(run.RUN / "progress.json"),
              "analysis_script_sha256": sha(Path(__file__)),
              "outputs": {p: sha(OUT / p) for p in ("individual_results.csv", "success_curves.csv", "exposure_thresholds.csv", "paired_outcomes.csv")},
              "confidence": "one-sided exact 95% lower, Bonferroni over five times per group/sky; no cross-group adjustment",
              "limitations": "Selected field and model assumptions; sample true redshifts are reference measurements, not wholly independent of source modeling."}
    (OUT / "analysis_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(thresholds.to_string(index=False), flush=True)
    print(curves.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
