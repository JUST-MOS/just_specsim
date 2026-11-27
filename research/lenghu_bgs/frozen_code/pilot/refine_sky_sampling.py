"""Recalculate the SkyCalc comparison with linear subpixel deposition.

Reuses native ETC sky counts from compare_sky_models.py, so the costly native
ETC calculation need not be repeated. High-resolution sky photons are assigned
linearly to neighboring native pixel centers before applying the same 33-diagonal
theoretical ETC LSF. This approximates (rather than exactly evaluates) the
continuous response of a line at arbitrary subpixel phase. No spectrum is
normalized to Lenghu or corrected with a second atmospheric transmission.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from compare_sky_models import (
    AIRMASS, CONDITIONS, EXPOSURE_S, FIELD_ANGLE_DEG, SEEING_800_ARCSEC,
    STUDY, ZENITH_DEG, engine, make_resolution, read_sky, sha256,
    trace_fraction, JUSTExposureTimeCalculator,
)


def subpixel_deposit(wavelength_nm, flux_per_micron, centers_nm, sensitivity):
    """Deposit trapezoid-integrated photons to adjacent detector centers.

    wavelength_nm and flux_per_micron are positive sampled ground radiance;
    sensitivity at samples is effective area * trace fraction * fiber area *
    time. The supplied center grid is uniform. Samples outside arm edges are
    discarded; an explicit LSF-edge mask is required downstream.
    """
    x = np.asarray(wavelength_nm, dtype=np.float64)
    f = np.asarray(flux_per_micron, dtype=np.float64)
    centers = np.asarray(centers_nm, dtype=np.float64)
    sens = np.asarray(sensitivity, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or f.shape != x.shape or sens.shape != x.shape:
        raise ValueError("invalid sky wavelength, flux or sensitivity shapes")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(sens)):
        raise ValueError("nonfinite sky input")
    if np.any(np.diff(x) <= 0) or np.any(f < 0) or np.any(sens < 0):
        raise ValueError("sky wavelength must increase; flux and sensitivity must be nonnegative")
    delta = np.diff(centers)
    if len(centers) < 2 or not np.allclose(delta, delta[0], rtol=0, atol=1e-10):
        raise ValueError("native detector centers must be linear")
    dw = np.diff(x)
    quadrature = np.concatenate(([dw[0] / 2], (dw[:-1] + dw[1:]) / 2, [dw[-1] / 2]))
    photons = f / 1000.0 * quadrature * sens
    position = (x - centers[0]) / delta[0]
    lower = np.floor(position).astype(np.int64)
    frac = position - lower
    deposited = np.zeros(len(centers), dtype=np.float64)
    for index, weight in ((lower, 1 - frac), (lower + 1, frac)):
        valid = (index >= 0) & (index < len(centers))
        deposited += np.bincount(index[valid], weights=(photons * weight)[valid], minlength=len(centers))
    return deposited, float(photons.sum()), float(deposited.sum())


def crop_arm(wave_nm, flux, start_nm, end_nm):
    """Crop on existing samples; require exact wavelength range endpoints."""
    use = (wave_nm >= start_nm - 1e-8) & (wave_nm <= end_nm + 1e-8)
    x = np.asarray(wave_nm[use], dtype=np.float64)
    y = np.asarray(flux[use], dtype=np.float64)
    if len(x) < 2 or not np.isclose(x[0], start_nm, rtol=0, atol=1e-8) or not np.isclose(x[-1], end_nm, rtol=0, atol=1e-8):
        raise ValueError(f"sky samples do not exactly cover {start_nm:g}–{end_nm:g} nm")
    return x, y


def setup_arm(calc, name, arm):
    calc.set_obs_conditions(
        seeing_fwhm_800=SEEING_800_ARCSEC, zenith_angle=ZENITH_DEG,
        field_angle=FIELD_ANGLE_DEG, lunar_za=CONDITIONS[name]["moon_za"],
        lunar_angle=60.0, lunar_phase=0.25,
    )
    calc._set_mode()
    sp = calc._configure_spectro(1)
    obs = calc._make_obs()
    n = int(sp.npix[arm])
    edges = sp.lmin[arm] + sp.dl[arm] * np.arange(n + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    aeff = np.array([engine.gsAeff(sp, obs, arm, w, FIELD_ANGLE_DEG) for w in centers])
    trace, trace_error = trace_fraction(sp, obs, arm, centers)
    radius = sp.fiber_ent_rad / sp.EFL[0] * engine.ARCSEC_PER_URAD
    sens = aeff * trace * math.pi * radius**2 * EXPOSURE_S
    resolution, valid, meta = make_resolution(sp, obs, arm, 10 * centers)
    return edges, centers, sens, resolution, valid, trace_error, meta


def apply_sky(wave_nm, flux, arm_setup, start_nm=None, end_nm=None):
    edges, centers, sens, resolution, valid, _, _ = arm_setup
    lo = float(edges[0] if start_nm is None else start_nm)
    hi = float(edges[-1] if end_nm is None else end_nm)
    wave, values = crop_arm(wave_nm, flux, lo, hi)
    sensitivity = np.interp(wave, centers, sens)
    raw, incoming, deposited = subpixel_deposit(wave, values, centers, sensitivity)
    return np.asarray(resolution.dot(raw)), valid, incoming, deposited


def self_check():
    centers = np.arange(5, dtype=float)
    # A symmetric line centered at 1.25 pixel distributes its integral 75/25.
    x = np.array([1.2, 1.25, 1.3])
    flux = np.array([0.0, 1000.0, 0.0])
    deposited, incoming, total = subpixel_deposit(x, flux, centers, np.ones(3))
    if not np.isclose(incoming, 0.05, atol=1e-12) or not np.isclose(total, incoming, atol=1e-12):
        raise AssertionError("subpixel triangular-line integral not conserved")
    if not np.allclose(deposited, [0, 0.0375, 0.0125, 0, 0], atol=1e-12):
        raise AssertionError("subpixel flux distribution/first moment failed")


def main():
    started = time.monotonic()
    self_check()
    output = Path(__file__).resolve().parent / "output"
    original = np.load(output / "sky_comparison_arrays.npz")
    revised = {}
    summaries = []
    data_rows = []
    setups = {}
    for name, condition in CONDITIONS.items():
        calc = JUSTExposureTimeCalculator(calc_mode="accurate")
        sky_path = STUDY / "sky" / condition["sky"]
        wave_nm, sky_flux, _ = read_sky(sky_path)
        for arm in range(3):
            setup = setup_arm(calc, name, arm)
            setups[(name, arm)] = setup
            result, valid, incoming, deposited = apply_sky(wave_nm, sky_flux, setup)
            key = f"{name}_arm{arm}"
            old = original[f"{key}_skycalc_e"]
            native = original[f"{key}_native_e"]
            centers = setup[1]
            if not np.array_equal(valid, original[f"{key}_valid"]):
                raise ValueError("old and updated LSF masks differ")
            if not np.allclose(centers, original[f"{key}_wave_nm"], rtol=0, atol=1e-9):
                raise ValueError("old and updated detector grids differ")
            diff = np.abs(result[valid] - old[valid]) / np.maximum(old[valid], 1e-30)
            summaries.append({
                "condition": name, "arm": arm, "valid_pixels": int(valid.sum()),
                "skycalc_to_native_integrated_ratio": float(result[valid].sum() / native[valid].sum()),
                "skycalc_mean_electrons_per_pixel": float(result[valid].mean()),
                "native_mean_electrons_per_pixel": float(native[valid].mean()),
                "subpixel_vs_bin_center_median_abs_relative": float(np.median(diff)),
                "subpixel_vs_bin_center_p95_abs_relative": float(np.percentile(diff, 95)),
                "subpixel_vs_bin_center_max_abs_relative": float(np.max(diff)),
                "edge_deposit_fraction_lost": float(1 - deposited / incoming),
                "trace_midpoint_max_relative_error": setup[5],
            })
            revised[f"{key}_wave_nm"] = centers
            revised[f"{key}_skycalc_e"] = result
            revised[f"{key}_native_e"] = native
            revised[f"{key}_valid"] = valid
            for w, a, b, ok in zip(centers, result, native, valid):
                data_rows.append((name, arm, w, a, b, int(ok)))

    sampling = []
    full_wave, full_flux, _ = read_sky(STUDY / "sky" / "skycalc_dark_001nm.fits")
    setup = setups[("dark", 2)]
    for lo, hi in ((750, 770), (825, 845)):
        path = STUDY / "sky" / f"skycalc_dark_0002nm_{lo}_{hi}.fits"
        fine_wave, fine_flux, _ = read_sky(path)
        a, _, _, _ = apply_sky(full_wave, full_flux, setup, lo, hi)
        b, _, _, _ = apply_sky(fine_wave, fine_flux, setup, lo, hi)
        guard = 16 * (setup[0][1] - setup[0][0])
        use = (setup[1] > lo + guard) & (setup[1] < hi - guard) & (b > 0)
        rel = np.abs(a[use] - b[use]) / b[use]
        sampling.append({
            "window_nm": [lo, hi], "valid_pixels": int(use.sum()),
            "median_abs_relative_difference": float(np.median(rel)),
            "p95_abs_relative_difference": float(np.percentile(rel, 95)),
            "max_abs_relative_difference": float(np.max(rel)),
            "reference_file_sha256": sha256(path),
        })

    with (output / "sky_comparison_subpixel_pixels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "arm", "wavelength_nm", "skycalc_e_per_pixel", "native_e_per_pixel", "interior_valid"])
        writer.writerows(data_rows)
    np.savez_compressed(output / "sky_comparison_subpixel_arrays.npz", **revised)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True, layout="constrained")
        for row, name in enumerate(("dark", "bright")):
            for arm in range(3):
                ax = axes[row, arm]
                key = f"{name}_arm{arm}"
                valid = revised[f"{key}_valid"]
                wave = revised[f"{key}_wave_nm"]
                ax.plot(wave[valid], revised[f"{key}_skycalc_e"][valid], color="#0072B2", lw=0.65,
                        label="ESO SkyCalc reference")
                ax.plot(wave[valid], revised[f"{key}_native_e"][valid], color="#D55E00", lw=0.65,
                        linestyle="--", label="ETC built-in sky")
                ax.set_yscale("log")
                ax.set_xlabel("Vacuum wavelength (nm)")
                ax.set_title(f"{name.capitalize()} — arm {arm}")
                ax.grid(alpha=0.18)
                if arm == 0:
                    ax.set_ylabel("Sky electrons / spectral pixel / 750 s")
                if row == 0 and arm == 0:
                    ax.legend(fontsize=7)
        fig.suptitle("Reference model comparison — neither is a calibrated Lenghu sky")
        fig.savefig(output / "sky_comparison_subpixel.png", dpi=200, facecolor="white")
        fig.savefig(output / "sky_comparison_subpixel.pdf", facecolor="white")
        plt.close(fig)
    report = {
        "purpose": "Method sensitivity, NOT a Lenghu sky calibration or a correction",
        "exposure_s": EXPOSURE_S, "airmass_source_secant": AIRMASS, "field_angle_deg": FIELD_ANGLE_DEG,
        "method": "trapezoidal sky photon quadrature; linearly deposit at adjacent detector centers; apply theoretical ETC LSF; mask 16 edge pixels",
        "limitation": "linear subpixel deposition approximates continuous instrumental LSF; native ETC sky and ESO sky use different atmosphere/site models; 0.01nm vs 0.002nm tested only in two narrow windows",
        "summaries": summaries, "dark_sampling_convergence": sampling,
        "self_checks": "triangular line total and first moment passed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (output / "sky_comparison_subpixel_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
