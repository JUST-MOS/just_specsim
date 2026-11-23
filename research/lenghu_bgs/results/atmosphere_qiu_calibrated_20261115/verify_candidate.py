#!/usr/bin/env python3
"""Verify candidate physics, provenance, and an opt-in JUST ETC interface."""
from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from fit_candidate import (AIRMASS, KNOTS, ESO_ATMOSPHERE, QIU_CSV, atmosphere_components, evaluate_full,
                           QIU_AIRMASS, HERE, BinnedSlopeModel, FIT_X, X_WEIGHTS)
from etc_adapter import QiuConstrainedAtmosphere
from lenghu_etc import LenghuPilotETC


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    report = json.loads((HERE / "fit_report.json").read_text())
    if sha(QIU_CSV) != report["qiu_csv_sha256"] or sha(ESO_ATMOSPHERE) != report["eso_fits_sha256"]:
        raise AssertionError("Source hash mismatch")
    for name, expected in report["code_sha256"].items():
        if sha(HERE/name) != expected:
            raise AssertionError(f"Code hash mismatch: {name}")
    for name, expected in report["output_sha256"].items():
        if sha(HERE/name) != expected:
            raise AssertionError(f"Output hash mismatch: {name}")
    original, molecular, rayleigh, adjustable = atmosphere_components()
    w = original.reference_wave_nm
    bins = BinnedSlopeModel(original, molecular, rayleigh, adjustable, "center")

    # Independent analytic limit: a wavelength-constant optical depth produces
    # a linear extinction law regardless of bins or spectral resolution.
    tau = 0.2
    flat = np.exp(-tau * FIT_X / AIRMASS)
    analytic = (2.5/np.log(10)) * tau/AIRMASS
    np.testing.assert_allclose((-2.5*np.log10(flat)) @ X_WEIGHTS, analytic, atol=1e-14)

    # With zero correction, the candidate must be exactly the pressure-
    # adjusted ESO product, including the molecular features.
    baseline, scale0 = evaluate_full(w, molecular, rayleigh, adjustable, np.zeros(len(KNOTS)))
    np.testing.assert_allclose(scale0, 1.0, atol=1e-14)
    np.testing.assert_allclose(baseline, original.reference_trans, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(bins.predict(np.zeros(len(KNOTS))),
                               np.array(__import__("pandas").read_csv(HERE/"extinction_fit_comparison.csv")["baseline_center"]),
                               atol=1e-12)

    # Check the stored components independently and confirm the first
    # derivative of the fitted smooth correction vanishes at both joins.
    nodes = np.array(report["fit_nodes"]["center"])
    spline = CubicSpline(KNOTS, nodes, bc_type="clamped")
    assert abs(float(spline(400, 1))) < 1e-12 and abs(float(spline(790, 1))) < 1e-12
    candidate, scale = evaluate_full(w, molecular, rayleigh, adjustable, nodes)
    with fits.open(HERE/"atmosphere_center_constant.fits", memmap=False, checksum=True) as h:
        if h["ATMOSPHERE"].columns["WAVELENGTH"].unit != "nm":
            raise AssertionError("Unexpected wavelength unit")
        np.testing.assert_array_equal(h["ATMOSPHERE"].data["WAVELENGTH"], w)
        np.testing.assert_allclose(h["ATMOSPHERE"].data["TRANSMISSION"], candidate, atol=1e-14)
        np.testing.assert_allclose(h["ATMOSPHERE"].data["O3_MIE_SCALE"], scale, atol=1e-14)
    assert np.all((candidate >= 0) & (candidate <= 1))

    # Independently convolve the full transmission grid before integrating
    # four bins, rather than reusing the fitting code's integration kernels.
    with (HERE/"extinction_fit_comparison.csv").open(newline="", encoding="utf-8") as f:
        table = {float(row["wavelength_label_nm"]): row for row in csv.DictReader(f)}
    independent_errors = {}
    for label in (400., 650., 760., 790.):
        width = 10. if label < 650. else 5.
        inside = (w >= label-width/2) & (w <= label+width/2)
        lsf_pixels = label / (600 * 2.354820045 * np.median(np.diff(w)))
        means = []
        for x in FIT_X:
            smoothed = gaussian_filter1d(candidate ** (x/AIRMASS), lsf_pixels,
                                         mode="nearest", truncate=6)
            means.append(np.trapz(smoothed[inside], w[inside]) / width)
        k_independent = float(np.polyfit(FIT_X, -2.5*np.log10(means), 1)[0])
        delta = k_independent - float(table[label]["candidate_center"])
        if abs(delta) >= 0.002:
            raise AssertionError(f"Independent convolution failed at {label} nm: {delta}")
        independent_errors[str(label)] = delta

    adapter = QiuConstrainedAtmosphere()
    np.testing.assert_allclose(adapter.transmission(np.array([400., 790., 925.]))[0],
                               np.interp([400., 790., 925.], w, candidate), atol=1e-14)
    try:
        adapter.transmission([500], airmass=1.3)
        raise AssertionError("Incorrect airmass accepted")
    except ValueError:
        pass

    instrument_path = HERE.parent / "population_4200_camera_overlap/output/instrument_reference_4200.npz"
    with np.load(instrument_path) as z:
        saved = {key: z[key] for key in z.files}
    instrument_checks = {}
    for sky in ("dark", "bright"):
        calc = LenghuPilotETC(sky, calc_mode="accurate")
        calc.atmosphere = QiuConstrainedAtmosphere()
        calc.set_obs_conditions(seeing_fwhm_800=.8, ebv=0., r_eff=0., decenter=.03,
                                sysfrac=.01, diffuse_stray=.02)
        x = np.linspace(3600., 9900., 200)
        response = calc.compute_snr(x, np.full_like(x, 1e-17), t_exp=750., n_exp=1)
        for arm, band in enumerate("BRZ"):
            old_sky = saved[f"{sky}_{band}_sky_e"]
            new_sky = calc.sky_diagnostics[arm]["sky_e"]
            delta = float(np.max(np.abs(new_sky-old_sky)) / np.max(old_sky))
            if delta > 1e-12:
                raise AssertionError(f"Sky radiance unexpectedly changed: {sky} {band}: {delta}")
            signal = np.asarray(response[arm]["signal"])
            previous = saved[f"{band}_flat_response_750_e"]
            if np.any(~np.isfinite(signal)) or np.any(signal <= 0):
                raise AssertionError(f"Invalid source response: {sky} {band}")
            instrument_checks[f"{sky}_{band}"] = {"sky_max_relative_difference": delta,
                "median_source_response_ratio_to_previous_hybrid": float(np.median(signal/previous))}

    results = {"status": "passed", "checks": ["output hashes", "analytic constant-optical-depth slope",
               "zero correction recovers ESO", "FITS units and reconstruction", "bounded transmission",
               "C1 continuum at joins", "independent Gaussian-convolution spot checks",
               "adapter guards fixed airmass", "actual ETC bright/dark smoke"],
               "airmass_qiu_same_zenith": QIU_AIRMASS,
               "independent_convolution_k_differences_mag_per_airmass": independent_errors,
               "etc_smoke": instrument_checks,
               "limitations": "Numerical/interface checks only; no independent physical calibration or Redrock rerun."}
    (HERE/"verification.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
