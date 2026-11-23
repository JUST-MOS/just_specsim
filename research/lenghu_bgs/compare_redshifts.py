"""Compare pilot Redrock rows using physical target, sky, and shutter time."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(reference, candidate, z_tolerance=1e-6, chi_rtol=1e-4, chi_atol=1e-3,
            expected_rows=None):
    keys = ["TARGETID", "sky", "total_s"]
    old, new = pd.read_csv(reference), pd.read_csv(candidate)
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("Duplicate target/sky/time rows")
    merged = new.merge(old, on=keys, how="left", validate="one_to_one", suffixes=("_new", "_old"), indicator=True)
    if not len(merged) or not (merged._merge == "both").all():
        raise ValueError("Candidate rows do not all match archived scenarios")
    if expected_rows is not None and len(merged) != expected_rows:
        raise ValueError(f"Expected {expected_rows} candidate rows, got {len(merged)}")
    z_old = pd.to_numeric(merged.z_fit_old, errors="coerce").to_numpy(float)
    z_new = pd.to_numeric(merged.z_fit_new, errors="coerce").to_numpy(float)
    truth = pd.to_numeric(merged.z_true_vi_old, errors="coerce").to_numpy(float)
    new_truth = pd.to_numeric(merged.z_true_vi_new, errors="coerce").to_numpy(float)
    same_truth = bool(np.isfinite(truth).all() and (truth > -1).all() and
                      np.array_equal(truth, new_truth))
    same_returned = np.array_equal(np.isfinite(z_old), np.isfinite(z_new))
    matched = np.isfinite(z_old) & np.isfinite(z_new)
    normalized = np.abs(z_new[matched] - z_old[matched]) / (1. + truth[matched])
    old_chi = pd.to_numeric(merged.DELTACHI2_old, errors="coerce").to_numpy(float)
    new_chi = pd.to_numeric(merged.DELTACHI2_new, errors="coerce").to_numpy(float)
    old_warn = pd.to_numeric(merged.ZWARN_old, errors="coerce").to_numpy(float)
    new_warn = pd.to_numeric(merged.ZWARN_new, errors="coerce").to_numpy(float)
    finite_returned_fields = bool(all(np.isfinite(arr[matched]).all()
                                      for arr in (old_chi, new_chi, old_warn, new_warn)))
    same_warn = bool(np.array_equal(old_warn, new_warn, equal_nan=True))
    same_type = bool(np.array_equal(merged.SPECTYPE_old.fillna(""), merged.SPECTYPE_new.fillna("")))
    old_success = (np.isfinite(z_old) & (old_warn == 0) & (old_chi > 40) &
                   (np.abs(z_old-truth)/(1+truth) < .0033))
    new_success = (np.isfinite(z_new) & (new_warn == 0) & (new_chi > 40) &
                   (np.abs(z_new-truth)/(1+truth) < .0033))
    same_success = bool(np.array_equal(old_success, new_success))
    delta_z = float(np.max(normalized)) if len(normalized) else None
    same_chi = bool(np.allclose(old_chi, new_chi, rtol=chi_rtol, atol=chi_atol, equal_nan=True))
    z_ok = delta_z is not None and delta_z <= z_tolerance
    return {"status": "passed" if same_truth and same_returned and finite_returned_fields and same_warn and same_type and same_success and same_chi and z_ok else "failed",
            "matched_scenarios": len(merged), "both_returned": int(matched.sum()),
            "same_input_truth": same_truth, "finite_returned_fields": finite_returned_fields,
            "same_returned": bool(same_returned), "same_zwarn": same_warn, "same_spectype": same_type,
            "same_success": same_success, "max_abs_dz_over_1plusz": delta_z,
            "max_abs_deltachi2_difference": float(np.nanmax(np.abs(old_chi-new_chi))) if matched.any() else None,
            "deltachi2_within_tolerance": same_chi, "z_tolerance": z_tolerance,
            "deltachi2_rtol": chi_rtol, "deltachi2_atol": chi_atol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-rows", type=int,
                        help="Require this many candidate scenarios (e.g. 10 for five targets in two skies).")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, expected_rows=args.expected_rows)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
