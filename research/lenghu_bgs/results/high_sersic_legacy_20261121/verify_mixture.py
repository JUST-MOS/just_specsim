"""Verify the opt-in JUST mixture against independent reference calculations.

Uses saved deterministic inputs and expected source counts only.  No mock
noise is drawn and Redrock is not called.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from desispec.resolution import Resolution

from compare_tractor import HERE, PRIOR, ROOT, STUDY, TractorDR94Profiles, continuous_fiber_fraction, sha
from just_etc import JUSTExposureTimeCalculator, ETC_py_optimized as engine
from just_etc.mixture_fiber_coupling import GaussianMixtureProfile, gaussian_mixture_fiber_fraction, etc_mixture_fiber_fraction

EXT = STUDY / "leader_195_205_20261116/extended_20p5/input"
POP = STUDY / "population_4200_camera_overlap"
PREV = STUDY / "sersic_fiber_model_20261121"
sys.path.insert(0, str(POP))
sys.path.insert(0, str(STUDY / "pilot_4200"))
from lenghu_etc import LenghuPilotETC
from simulate_batch import pixel_average
from tractor_adapter import make_source_arm_mixture


def main():
    prior_manifest = json.loads((PREV / "summary.json").read_text())
    for relative, digest in prior_manifest["inputs"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("previously frozen input changed: " + relative)
    design = json.loads((EXT / "frozen_design.json").read_text())
    instrument_meta = json.loads((POP / "output/instrument_validation.json").read_text())
    camera_meta = json.loads((POP / "output/desi_camera_mapping.json").read_text())
    lsf_meta = json.loads((EXT / "combined_resolution_manifest.json").read_text())
    run_meta = json.loads((POP / "output/test_0000_spectra_manifest.json").read_text())
    checks = {
        EXT / "normalized_sources.fits": design["normalized_sources_sha256"],
        EXT / "combined_resolution_nodes.npz": lsf_meta["combined_nodes_sha256"],
        POP / "output/instrument_reference_4200.npz": instrument_meta["instrument_sha256"],
        POP / "output/desi_camera_mapping.npz": camera_meta["mapping_sha256"],
        POP / "simulate_batch.py": run_meta["simulation_script_sha256"],
        STUDY / "pilot_4200/lenghu_etc.py": run_meta["research_adapter_sha256"],
    }
    bricks_meta = json.loads((HERE / "matched_legacy_bricks.json").read_text())
    checks.update({ROOT / entry["path"]: entry["sha256"] for entry in bricks_meta["files"]})
    for path, digest in checks.items():
        if sha(path) != digest:
            raise ValueError("frozen adapter or Legacy brick input changed: " + str(path))
    reconstruction = pd.read_csv(HERE / "catalog_reconstruction_per_target.csv")
    test_reconstruction = reconstruction[reconstruction.split == "test"]
    direct_pixel = pd.read_csv(HERE / "pixel_diagnosis_per_target.csv")
    assert len(test_reconstruction) == 1332 and len(direct_pixel) == 67
    assert np.max(np.abs(test_reconstruction.pixel_over_catalog - 1.)) < .001
    assert np.max(np.abs(direct_pixel.pixel_over_cat - 1.)) < 2e-7
    data = pd.read_csv(PRIOR)
    direct_cat = pd.read_csv(HERE / "matched_legacy_bricks.csv")
    model = TractorDR94Profiles()
    direct = data.merge(direct_cat[["TARGETID"]], on="TARGETID", validate="one_to_one")
    reference_errors = []
    for row in direct.itertuples(index=False):
        amp, var = model.profile(row.morphology, row.SERSIC)
        profile = GaussianMixtureProfile(row.SHAPE_R, row.SHAPE_E1, row.SHAPE_E2, amp, var)
        # Legacy defines seeing sigma=FWHM/2.35, so represent that as the
        # equivalent standard-Gaussian FWHM for the independent calculation.
        value = gaussian_mixture_fiber_fraction(profile, 1.5,
                    2 * np.sqrt(2 * np.log(2)) / 2.35)
        reference = continuous_fiber_fraction(amp, var, row.SHAPE_R, row.axis_ratio)
        reference_errors.append(abs(value / reference - 1))
    assert max(reference_errors) < 1e-8

    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, field_angle=0., decenter=.03)
    calc._set_mode()
    calc._configure_spectro(1)
    sp, obs = calc._spectro, calc._make_obs()
    point = GaussianMixtureProfile(1., 0., 0., [1.], [0.])
    point_waves = np.array([400., 500., 623.1, 800., 900.])
    point_new = etc_mixture_fiber_fraction(point, sp, obs, point_waves)
    point_original = np.array([engine.gsGeometricThroughput(sp, obs, w, 0., .03, 0., 0)
                               for w in point_waves])
    point_error = float(np.max(np.abs(point_new / point_original - 1.)))
    assert point_error < .01
    nodes = np.linspace(min(sp.lmin[:sp.N_arms]), max(sp.lmax[:sp.N_arms]), 21)
    off_grid = np.array([401.3, 577.7, 884.2])
    selected = pd.concat([data.nsmallest(2, "axis_ratio"), data.nlargest(2, "SHAPE_R"),
                          data.nlargest(2, "SERSIC"), data.iloc[[200, 700, 1200]]]).drop_duplicates("TARGETID")
    interpolation_errors = []
    for row in selected.itertuples(index=False):
        amp, var = model.profile(row.morphology, row.SERSIC)
        profile = GaussianMixtureProfile(row.SHAPE_R, row.SHAPE_E1, row.SHAPE_E2, amp, var)
        anchored = etc_mixture_fiber_fraction(profile, sp, obs, nodes)
        evaluated = etc_mixture_fiber_fraction(profile, sp, obs, off_grid)
        interpolation_errors.extend(np.abs(np.interp(off_grid, nodes, anchored) / evaluated - 1.))
    assert max(interpolation_errors) < .001

    sample = pd.read_csv(EXT / "frozen_sample.csv")
    calc2 = LenghuPilotETC("dark", calc_mode="accurate")
    calc2.set_obs_conditions(seeing_fwhm_800=.8, ebv=0., r_eff=0., decenter=.03)
    calc2._set_mode()
    calc2._configure_spectro(1)
    with np.load(POP / "output/instrument_reference_4200.npz") as z:
        inst = {k: z[k].copy() for k in z.files}
    with np.load(POP / "output/desi_camera_mapping.npz") as z:
        mapping = {k: z[k].copy() for k in z.files}
    with np.load(EXT / "combined_resolution_nodes.npz") as z:
        ids, lsf_wave, lsf_sigma = z["targetid"].copy(), z["wave"].copy(), z["sigma"].copy()
    index = {int(t): i for i, t in enumerate(ids)}
    fallback = np.nanmedian(lsf_sigma, axis=(0, 2))
    chosen = []
    for morph in ("REX", "EXP", "DEV", "SER"):
        rows = sample[(sample.split == "test") & (sample.morphology == morph)]
        chosen.append(rows.sort_values("SERSIC").iloc[-1])
    response_errors, adapter_results = [], []
    with fits.open(EXT / "normalized_sources.fits", memmap=True, checksum=True) as hdus:
        wave = np.array(hdus["WAVELENGTH"].data, dtype=float)
        for series in chosen:
            row = next(pd.DataFrame([series]).itertuples(index=False))
            assert int(hdus["TARGETID"].data[int(row.source_row)]) == int(row.TARGETID)
            flux = np.array(hdus["FLUX_1E17"].data[int(row.source_row)], dtype=float)
            ii = index[int(row.TARGETID)]
            amp, var = model.profile(row.morphology, row.SERSIC)
            profile = GaussianMixtureProfile(row.SHAPE_R, row.SHAPE_E1, row.SHAPE_E2, amp, var)
            for arm, band in enumerate("BRZ"):
                response, counts, resolution, mask = make_source_arm_mixture(
                    row, flux, wave, band, arm, inst, calc2, lsf_wave,
                    lsf_sigma[ii], mapping, profile, fallback[arm])
                nodes = inst[f"{band}_geo_wave_nodes_nm"]
                w = inst[f"{band}_wave_nm"]
                new_geo = etc_mixture_fiber_fraction(profile, calc2._spectro, calc2._make_obs(), nodes)
                dust = 10**(-.4 * inst[f"{band}_kgal_per_ebv"] * row.EBV)
                expected = inst[f"{band}_flat_response_750_e"] * dust * \
                           np.interp(w, nodes, new_geo) / np.interp(w, nodes, inst[f"{band}_geo0_nodes"])
                delta = float(np.max(np.abs(response / expected - 1.)))
                response_errors.append(delta)
                assert delta < 1e-12
                bins = pixel_average(wave, np.maximum(flux, 0), inst[f"{band}_edges_aa"])
                expected_counts = Resolution(inst[f"{band}_lsf_just"]).dot(expected * bins)
                np.testing.assert_allclose(counts, np.maximum(expected_counts, 0), rtol=1e-12, atol=1e-12)
                assert resolution.shape[1] == len(w) and np.all(np.isfinite(counts)) and np.all(counts >= 0)
                assert mask.size == len(w)
                adapter_results.append({"targetid": int(row.TARGETID), "morphology": row.morphology, "band": band})
    outcome = {
        "status": "passed", "prior_frozen_input_hashes_match": True,
        "frozen_adapter_and_brick_hashes_match": True,
        "n_adapter_and_brick_inputs_verified": len(checks),
        "test_pixel_max_relative_difference": float(np.max(np.abs(test_reconstruction.pixel_over_catalog - 1.))),
        "direct_pixel_max_relative_difference": float(np.max(np.abs(direct_pixel.pixel_over_cat - 1.))),
        "n_direct_brick_profiles": len(reference_errors),
        "continuous_fraction_max_relative_difference": float(max(reference_errors)),
        "n_interpolation_profiles": len(selected),
        "interpolation_max_relative_difference": float(max(interpolation_errors)),
        "point_check_wavelengths_nm": point_waves.tolist(),
        "etc_psf_mixture_point_max_relative_difference": point_error,
        "adapter_n_galaxies": len(chosen), "adapter_n_target_arm": len(adapter_results),
        "adapter_max_response_relative_difference": max(response_errors),
        "adapter_sources": adapter_results,
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                           (PRIOR, PREV / "summary.json", Path(__file__), HERE / "tractor_adapter.py",
                            ROOT / "py/just_etc/mixture_fiber_coupling.py")},
        "redrock_run": False,
    }
    (HERE / "mixture_verification.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == "__main__":
    main()
