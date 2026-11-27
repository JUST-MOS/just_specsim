"""Small tests for the relocation and the success-rate denominator."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bgs_workflow", HERE / "workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
compare_spec = importlib.util.spec_from_file_location("bgs_compare", HERE / "compare_redshifts.py")
comparison = importlib.util.module_from_spec(compare_spec)
compare_spec.loader.exec_module(comparison)


def _write(root, relative, text="placeholder"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_layout_rebuilds_historical_absolute_links(tmp_path):
    data = tmp_path / "inputs"
    for name in ("sky/sky.fits", "pilot_4200/extinction/table.csv",
                 str(workflow.EXT / "input/frozen_sample.csv"),
                 str(workflow.HIGH / "mixture_verification.json"),
                 "sersic_fiber_model_20261121/fiber_coupling_per_target.csv",
                 "sersic_fiber_model_20261121/summary.json",
                 str(workflow.POP / "output/redrock_cache_validation.json"),
                 str(workflow.EXT / "shared/desi_camera_mapping.json")):
        _write(data, name)
    for name in ("desi_camera_mapping.npz", "instrument_reference_4200.npz", "instrument_validation.json"):
        _write(data, workflow.POP / "output" / name, text=name)
    etc = tmp_path / "just_etc"
    _write(etc, "mixture_fiber_coupling.py")
    root, study = workflow.prepare_layout(tmp_path / "work", data, etc)
    assert (root / "py/just_etc").resolve() == etc.resolve()
    assert (study / workflow.EXT / "shared/instrument_reference_4200.npz").read_text() == "instrument_reference_4200.npz"
    assert workflow.sha(study / workflow.TRACTOR / "simulate_tractor.py") == workflow.sha(
        workflow.FROZEN / workflow.TRACTOR / "simulate_tractor.py")


def test_rejects_output_in_input_tree_and_existing_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError):
        workflow._check_output_outside_input(data / "new.fits", data, [])
    existing = tmp_path / "result.fits"
    existing.write_text("old data")
    with pytest.raises(FileExistsError):
        workflow._check_output_outside_input(existing, data, [existing])


def test_summary_counts_missing_and_bad_redshifts_in_denominator(tmp_path):
    data = tmp_path / "input"
    sample_path = data / workflow.EXT / "input/frozen_sample.csv"
    sample_path.parent.mkdir(parents=True)
    pd.DataFrame({"TARGETID": [1, 2, 3], "r_dered": [19., 19.2, 20.]}).to_csv(sample_path, index=False)
    results = pd.DataFrame({"TARGETID": [1, 2, 3], "sky": ["dark"] * 3, "total_s": [750] * 3,
                            "z_fit": [.2, .3, np.nan], "z_true_vi": [.2, .2, .2],
                            "ZWARN": [0, 0, np.nan], "DELTACHI2": [50, 50, np.nan]})
    results_path = tmp_path / "results.csv"
    output_path = tmp_path / "summary.csv"
    results.to_csv(results_path, index=False)
    args = argparse.Namespace(data_root=data, results=results_path, output=output_path,
                              confidence=.95, scan_count=1)
    workflow.summarize(args)
    summary = pd.read_csv(output_path).set_index("r_dered_lt")
    assert summary.loc[20.5, "attempted"] == 3
    assert summary.loc[20.5, "returned"] == 2
    assert summary.loc[20.5, "quality_pass"] == 2
    assert summary.loc[20.5, "success"] == 1
    assert summary.loc[20.5, "purity"] == .5
    assert summary.loc[20.5, "one_sided_lower"] == pytest.approx(beta.ppf(.05, 1, 3))
    assert summary.loc[19.5, "attempted"] == 2

    extra = results.iloc[[0]].copy()
    extra["total_s"] = 1500
    pd.concat([results, extra]).to_csv(results_path, index=False)
    args.output = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="scan-count"):
        workflow.summarize(args)


def test_redshift_comparison_requires_complete_and_valid_returned_rows(tmp_path):
    rows = pd.DataFrame({"TARGETID": [1, 2], "sky": ["dark", "bright"],
                         "total_s": [750, 750], "z_fit": [.2, .3],
                         "z_true_vi": [.2, .3], "ZWARN": [0, 0],
                         "DELTACHI2": [50., 60.], "SPECTYPE": ["GALAXY", "GALAXY"]})
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    rows.to_csv(reference, index=False)
    rows.to_csv(candidate, index=False)
    assert comparison.compare(reference, candidate, expected_rows=2)["status"] == "passed"
    rows.iloc[:1].to_csv(candidate, index=False)
    with pytest.raises(ValueError, match="Expected 2"):
        comparison.compare(reference, candidate, expected_rows=2)
    rows.loc[0, "DELTACHI2"] = np.nan
    rows.to_csv(candidate, index=False)
    assert comparison.compare(reference, candidate, expected_rows=2)["status"] == "failed"
