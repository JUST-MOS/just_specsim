"""Analyze the completed r < 20.5 extension and original held-out results.

All fractions and intervals are conditional on the selected deep-field sample
and frozen simulation. Exposure thresholds are sampled grid points, not
interpolated estimates. A complete checkpointed run is required.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

import run_extension as scan

HERE = Path(__file__).resolve().parent
ROOT = HERE / "extended_20p5"
OUT = ROOT / "analysis_20261119"
OLD = HERE.parent / "population_4200_camera_overlap"
COLS = ["SYNTHETIC_TARGETID", "TARGETID", "source_row", "split", "z_true_vi", "sky",
        "total_s", "exposures_s", "z_fit", "zerr", "ZWARN", "DELTACHI2", "SPECTYPE",
        "returned", "delta_z_norm", "quality_pass", "truth_pass", "success", "outcome"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lower(k, n, alpha):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def interval(k, n):
    if not n:
        return np.nan, np.nan
    lo = 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))
    return lo, hi


def verify_flags(rows):
    returned = np.isfinite(rows.z_fit.to_numpy(float))
    delta = (rows.z_fit - rows.z_true_vi) / (1.0 + rows.z_true_vi)
    quality = returned & (rows.ZWARN == 0) & (rows.DELTACHI2 > 40)
    correct = returned & (np.abs(delta) < .0033)
    success = quality & correct
    outcome = np.select([~returned, returned & ~quality, quality & ~correct, success],
                        ["no_result", "quality_fail", "wrong_redshift", "success"],
                        default="internal_error")
    np.testing.assert_allclose(delta, rows.delta_z_norm, rtol=1e-12, atol=1e-15, equal_nan=True)
    for actual, expected in ((rows.returned, returned), (rows.quality_pass, quality),
                             (rows.truth_pass, correct), (rows.success, success),
                             (rows.outcome, outcome)):
        np.testing.assert_array_equal(actual, expected)


def groups(sample, design):
    yield "all", "all", sample
    for field, kind in (("r_bin", "magnitude"), ("z_bin", "redshift"),
                        ("spectral_type", "spectral_type")):
        for name, group in sample.groupby(field, sort=True, dropna=False):
            yield kind, str(name), group
    for name, group in sample.groupby(["r_bin", "z_bin", "spectral_type"], sort=True, dropna=False):
        yield "joint", "|".join(name), group
    for field, cut, kind in (("gr_dered", design["color_split"], "observable_color"),
                             ("fiber_r_dered", design["fiber_r_split"], "observable_fiber_r")):
        for mag, sub in sample.groupby("r_bin", sort=True):
            for label, group in (("low_or_equal", sub[sub[field] <= cut]),
                                 ("high", sub[sub[field] > cut])):
                if len(group):
                    yield kind, f"{mag}|{label}", group


def load_complete():
    design = json.loads((OUT / "design.json").read_text())
    selected = scan.preflight()
    progress = json.loads((scan.OUT / "progress.json").read_text())
    expected = [(int(a), int(b)) for a, b in scan.PLAN]
    actual = [(x["start"], x["galaxies"]) for x in progress["completed"]]
    if progress["status"] != "complete" or actual != expected:
        raise RuntimeError("extension scan has not completed every planned batch")
    oldpath = OLD / "output/test_individual_results.csv"
    oldmanifest = json.loads((OLD / "output/analysis_manifest.json").read_text())
    audit = json.loads((OLD / "output/independent_final_audit.json").read_text())
    if (sha(oldpath) != oldmanifest["individual_sha256"] or
        oldmanifest["status"] != "complete" or audit["status"] != "passed"):
        raise ValueError("original sample results or independent audit do not match")
    old = pd.read_csv(oldpath, usecols=COLS, float_precision="round_trip")
    files = [{"kind": "original_individual", "path": str(oldpath), "sha256": sha(oldpath)}]
    new = []
    for start, count in expected:
        spectra = scan.validate_spectra(start, count, selected)
        meta = scan.validate_fit(start, spectra)
        if meta["attempted"] != 10 * count:
            raise ValueError("incorrect attempted count")
        p = scan.OUT / f"test_{start:04d}_redrock.csv"
        new.append(pd.read_csv(p, usecols=COLS, float_precision="round_trip"))
        files.append({"kind": "extension_batch", "start": start, "galaxies": count,
                      "csv_sha256": sha(p), "fit_fits_sha256": meta["output_sha256"],
                      "spectra_sha256": sha(spectra)})
    rows = pd.concat([old, *new], ignore_index=True)
    if (len(old) != 8910 or len(rows) != 13320 or rows.SYNTHETIC_TARGETID.duplicated().any()
        or rows.duplicated(["TARGETID", "sky", "total_s"]).any()):
        raise ValueError("unexpected complete dataset size or duplicate identifiers")
    expected_cases = {(int(tid), sky, t) for tid in selected.TARGETID for sky in scan.SKY for t in scan.TIMES}
    actual_cases = set(zip(rows.TARGETID.astype(int), rows.sky, rows.total_s.astype(int)))
    if actual_cases != expected_cases:
        raise ValueError("selected target/sky/time combinations differ from results")
    if set(old.TARGETID) != set(selected.iloc[:891].TARGETID) or set(rows.TARGETID) != set(selected.TARGETID):
        raise ValueError("old or combined target membership differs from frozen sample")
    if not np.isfinite(selected.FIBERFLUX_R).all() or not np.isfinite(selected.MW_TRANSMISSION_R).all():
        raise ValueError("missing imaging predictor")
    if not ((selected.FIBERFLUX_R > 0) & (selected.MW_TRANSMISSION_R > 0)).all():
        raise ValueError("fiber flux or dust transmission must be positive")
    selected["fiber_r_dered"] = 22.5 - 2.5 * np.log10(selected.FIBERFLUX_R / selected.MW_TRANSMISSION_R)
    if not np.isfinite(selected.gr_dered).all():
        raise ValueError("missing dereddened color")
    expected_r = np.select([selected.r_dered <= 19, selected.r_dered <= 20, selected.r_dered < 20.5],
                           ["r_le_19", "19_lt_r_le_20", "20_lt_r_lt_20p5"], default="invalid")
    expected_z = np.where(selected.VI_Z <= .25, "z_0p1_0p25", "z_0p25_0p4")
    if (not ((selected.r_dered < 20.5) & (selected.VI_Z > .1) & (selected.VI_Z < .4)).all() or
        not np.array_equal(expected_r, selected.r_bin) or not np.array_equal(expected_z, selected.z_bin) or
        set(selected.spectral_type) != {"bpt_star_forming", "weak_line_old", "other_unclassified"}):
        raise ValueError("frozen grouping metadata inconsistent with values")
    info = selected[["TARGETID", "VI_Z", "r_dered", "gr_dered", "fiber_r_dered", "r_bin",
                     "z_bin", "spectral_type", "source_row"]]
    rows = rows.merge(info, on="TARGETID", validate="many_to_one", suffixes=("", "_catalog"))
    if (not np.allclose(rows.z_true_vi, rows.VI_Z, rtol=0, atol=1e-12) or
        not np.array_equal(rows.source_row, rows.source_row_catalog)):
        raise ValueError("catalog redshift or source row mismatch")
    verify_flags(rows)
    return rows, selected, design, files


def calculate(rows, sample, design):
    curves = []
    thresholds = []
    for kind, name, group in groups(sample, design):
        ids = set(group.TARGETID)
        n = len(ids)
        for sky in scan.SKY:
            choices = []
            for t in scan.TIMES:
                data = rows[(rows.TARGETID.isin(ids)) & (rows.sky == sky) & (rows.total_s == t)]
                if len(data) != n or set(data.TARGETID) != ids:
                    raise ValueError(f"group denominator mismatch {kind} {name} {sky} {t}")
                k = int(data.success.sum())
                q = int(data.quality_pass.sum())
                purity_lo, purity_hi = interval(k, q)
                entry = {"group_type": kind, "group": name, "sky": sky, "time_s": t,
                         "n": n, "k_success": k, "success_fraction": k / n,
                         "cp95_lower_pointwise": lower(k, n, .05),
                         "cp95_lower_bonferroni_times": lower(k, n, .01),
                         "quality_pass": q, "purity": k / q if q else np.nan,
                         "purity_cp95_lo": purity_lo, "purity_cp95_hi": purity_hi,
                         "no_result": int((data.outcome == "no_result").sum()),
                         "quality_fail": int((data.outcome == "quality_fail").sum()),
                         "wrong_redshift_quality_pass": int((data.outcome == "wrong_redshift").sum())}
                if k + entry["no_result"] + entry["quality_fail"] + entry["wrong_redshift_quality_pass"] != n:
                    raise ValueError("failure categories do not partition group")
                curves.append(entry)
                choices.append(entry)
            for target in (.9, .95):
                minimum_n = math.ceil(math.log(.01) / math.log(target))
                def first(column):
                    return next((x["time_s"] for x in choices if x[column] >= target), None)
                adjusted = first("cp95_lower_bonferroni_times")
                thresholds.append({"group_type": kind, "group": name, "sky": sky, "target": target,
                    "n": n, "min_n_all_success_adjusted": minimum_n,
                    "time_s_point": first("success_fraction"),
                    "time_s_pointwise_lower": first("cp95_lower_pointwise"),
                    "time_s_adjusted_lower": adjusted,
                    "status_adjusted": "met" if adjusted is not None else
                        ("insufficient_even_if_all_success" if n < minimum_n else "not_met_at_tested_times")})
    return pd.DataFrame(curves), pd.DataFrame(thresholds)


def main():
    rows, sample, design, files = load_complete()
    curves, thresholds = calculate(rows, sample, design)
    for label, data in (("individual_results.csv", rows), ("success_curves.csv", curves),
                        ("exposure_thresholds.csv", thresholds)):
        data.to_csv(OUT / label, index=False)
    result = {"status": "complete", "n_test": len(sample), "n_rows": len(rows),
              "n_original": 891, "n_new": 441, "times_s": scan.TIMES, "skies": scan.SKY,
              "input_files": files, "sample_sha256": sha(scan.DATA / "frozen_sample.csv"),
              "design_sha256": sha(OUT / "design.json"), "run_progress_sha256": sha(scan.OUT / "progress.json"),
              "analysis_script_sha256": sha(Path(__file__)), "python": sys.version,
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "astropy")},
              "outputs": {p: sha(OUT / p) for p in ("individual_results.csv", "success_curves.csv", "exposure_thresholds.csv")},
              "interval_scope": design["caveats"]}
    (OUT / "analysis_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(curves.query("group_type == 'all'").to_string(index=False), flush=True)
    print(thresholds.query("group_type == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
