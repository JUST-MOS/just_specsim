"""Compare ETC-native sky counts with unscaled ESO SkyCalc reference spectra.

This is a phase-1 diagnostic, NOT a Lenghu-calibrated sky spectrum or a redshift
forecast. SkyCalc spectra are ground-level sky radiance, in photons s-1 m-2
micron-1 arcsec-2 on a vacuum nm grid. For each arm we integrate incident sky
photons over the detector's pixel wavelength bounds, multiply by JUST model
collecting area, fiber solid angle, exposure and sky-trace fraction, and apply
the ETC's theoretical LSF. Sky binning before discrete LSF application places
each bin's photons at its center, so unresolved subpixel line structure is an
approximation. The instrument LSF is theoretical and has not been measured.

The native ETC sky output includes its own line, continuum and moon model; it is
not added to SkyCalc. We compare sky electrons before detector/read/systematic
terms. The models differ in site and atmospheric assumptions; a ratio is NOT a
calibration or correction. Both paths use the same JUST model and field angle.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(STUDY / "instrument"))

from just_etc import JUSTExposureTimeCalculator  # noqa: E402
from just_etc import ETC_py_optimized as engine  # noqa: E402
from theoretical_lsf import make_resolution  # noqa: E402

EXPOSURE_S = 750.0
FIELD_ANGLE_DEG = 0.0  # Explicit on-axis diagnostic, not an FOV average.
AIRMASS = 1.2
ZENITH_DEG = math.degrees(math.acos(1.0 / AIRMASS))
SEEING_800_ARCSEC = 0.8
CONDITIONS = {
    "dark": {"sky": "skycalc_dark_001nm.fits", "moon_za": 135.0},
    "bright": {"sky": "skycalc_bright_001nm.fits", "moon_za": 45.0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrate_linear(wavelength_nm, flux_per_micron, bin_edges_nm):
    """Integrate nonnegative linearly interpolated photon radiance in nm bins.

    Input flux: photons s-1 m-2 micron-1 arcsec-2. Output per bin:
    photons s-1 m-2 arcsec-2. No extrapolation is allowed.
    """
    x = np.asarray(wavelength_nm, dtype=np.float64)
    y = np.asarray(flux_per_micron, dtype=np.float64) / 1000.0
    edges = np.asarray(bin_edges_nm, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2 or edges.ndim != 1:
        raise ValueError("sky wavelength, flux and bin edges must be 1D")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite sky wavelength or flux")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(edges) <= 0) or np.any(y < 0):
        raise ValueError("wavelengths and edges must increase and flux be nonnegative")
    if edges[0] < x[0] - 1e-9 or edges[-1] > x[-1] + 1e-9:
        raise ValueError("sky spectrum does not cover all requested bin edges")
    edges = np.clip(edges, x[0], x[-1])
    dx = np.diff(x)
    cumulative = np.concatenate(([0.0], np.cumsum(dx * (y[:-1] + y[1:]) / 2.0)))
    index = np.clip(np.searchsorted(x, edges, side="right") - 1, 0, len(x) - 2)
    delta = edges - x[index]
    slope = (y[index + 1] - y[index]) / dx[index]
    at_edge = cumulative[index] + y[index] * delta + 0.5 * slope * delta**2
    integral = np.diff(at_edge)
    if np.any(integral < -1e-10):
        raise ValueError("negative integrated sky photons")
    return np.maximum(integral, 0.0)


def self_check():
    """Check physical units and conservation independently of real spectra."""
    x = np.array([0.0, 0.1, 0.2, 0.3])
    constant = integrate_linear(x, np.full(4, 1000.0), np.array([0, 0.15, 0.3]))
    if not np.allclose(constant, [0.15, 0.15], rtol=0, atol=1e-13):
        raise AssertionError("flat-spectrum per-micron to per-nm conversion failed")
    triangle = integrate_linear(x, np.array([0, 1000, 1000, 0]), np.array([0, 0.1, 0.2, 0.3]))
    if not np.allclose(triangle, [0.05, 0.1, 0.05], rtol=0, atol=1e-13):
        raise AssertionError("linear-spectrum integration/flux conservation failed")


def read_sky(path: Path):
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        columns = hdus[1].columns
        if columns["lam"].unit != "nm" or columns["flux"].unit != "ph/s/m2/micron/arcsec2":
            raise ValueError(f"unexpected SkyCalc units in {path}")
        wave = np.array(table["lam"], dtype=np.float64, copy=True)
        flux = np.array(table["flux"], dtype=np.float64, copy=True)
        moon = np.array(table["flux_sml"], dtype=np.float64, copy=True)
    return wave, flux, moon


def trace_fraction(sp, obs, arm, wave_nm):
    """Interpolate an explicit grid; check midpoint interpolation error."""
    nodes = np.linspace(wave_nm[0], wave_nm[-1], 33)
    node_values = np.array([engine.gsFracTrace(sp, obs, arm, w, 1) for w in nodes])
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    exact = np.array([engine.gsFracTrace(sp, obs, arm, w, 1) for w in mids])
    interpolated = np.interp(mids, nodes, node_values)
    max_rel = float(np.max(np.abs(interpolated - exact) / np.maximum(np.abs(exact), 1e-12)))
    if max_rel > 1e-3:
        raise ValueError(f"sky trace interpolation error {max_rel:g} exceeds 0.1%")
    return np.interp(wave_nm, nodes, node_values), max_rel


def main():
    start = time.monotonic()
    self_check()
    sky_paths = {name: STUDY / "sky" / config["sky"] for name, config in CONDITIONS.items()}
    sky_data = {name: read_sky(path) for name, path in sky_paths.items()}
    if not np.array_equal(sky_data["dark"][0], sky_data["bright"][0]):
        raise ValueError("dark and bright SkyCalc wavelength grids differ")
    if np.any(sky_data["dark"][2] != 0):
        raise ValueError("dark reference unexpectedly has a moon component")

    output = Path(__file__).resolve().parent / "output"
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    npz_arrays = {}
    for name, config in CONDITIONS.items():
        calc = JUSTExposureTimeCalculator(calc_mode="accurate")
        calc.set_obs_conditions(
            seeing_fwhm_800=SEEING_800_ARCSEC,
            zenith_angle=ZENITH_DEG,
            field_angle=FIELD_ANGLE_DEG,
            lunar_za=config["moon_za"],
            lunar_angle=60.0,
            lunar_phase=0.25,
        )
        calc._set_mode()
        sp = calc._configure_spectro(n_exp=1)
        obs = calc._make_obs()
        if not math.isclose(FIELD_ANGLE_DEG, 0.0):
            raise ValueError("this diagnostic currently assumes on-axis EFL")
        radius_arcsec = sp.fiber_ent_rad / sp.EFL[0] * engine.ARCSEC_PER_URAD
        solid_angle_arcsec2 = math.pi * radius_arcsec**2

        for arm in range(sp.N_arms):
            n = int(sp.npix[arm])
            edges_nm = sp.lmin[arm] + sp.dl[arm] * np.arange(n + 1)
            centers_nm = 0.5 * (edges_nm[:-1] + edges_nm[1:])
            wave_aa = 10.0 * centers_nm
            native_noise, native_sky, sample_factor = engine.gsGetNoise(
                sp, obs, arm, FIELD_ANGLE_DEG, EXPOSURE_S, 0x0
            )
            del native_noise
            per_bin = integrate_linear(sky_data[name][0], sky_data[name][1], edges_nm)
            aeff = np.array([engine.gsAeff(sp, obs, arm, w, FIELD_ANGLE_DEG) for w in centers_nm])
            trace, trace_error = trace_fraction(sp, obs, arm, centers_nm)
            before_lsf = per_bin * aeff * solid_angle_arcsec2 * trace * EXPOSURE_S
            resolution, valid, lsf_metadata = make_resolution(sp, obs, arm, wave_aa)
            external_sky = np.asarray(resolution.dot(before_lsf))
            if np.any(~np.isfinite(external_sky)) or np.any(external_sky < -1e-8):
                raise ValueError("non-finite or negative model sky counts")
            native_sky = np.asarray(native_sky)
            if np.any(~np.isfinite(native_sky)) or np.any(native_sky[valid] <= 0):
                raise ValueError("non-finite or nonpositive native sky counts")
            ratio = float(np.sum(external_sky[valid]) / np.sum(native_sky[valid]))
            summaries.append({
                "condition": name, "arm": int(arm), "wavelength_min_nm": float(edges_nm[0]),
                "wavelength_max_nm": float(edges_nm[-1]), "valid_pixels": int(valid.sum()),
                "pixel_width_angstrom": float(10 * sp.dl[arm]),
                "skycalc_to_native_integrated_ratio": ratio,
                "skycalc_mean_electrons_per_pixel": float(np.mean(external_sky[valid])),
                "native_mean_electrons_per_pixel": float(np.mean(native_sky[valid])),
                "trace_interpolation_max_relative_error": trace_error,
                "sample_factor": float(sample_factor), "lsf": lsf_metadata,
            })
            for wavelength, a, b, ok in zip(centers_nm, external_sky, native_sky, valid):
                all_rows.append((name, arm, wavelength, a, b, int(ok)))
            key = f"{name}_arm{arm}"
            npz_arrays[f"{key}_wave_nm"] = centers_nm
            npz_arrays[f"{key}_skycalc_e"] = external_sky
            npz_arrays[f"{key}_native_e"] = native_sky
            npz_arrays[f"{key}_valid"] = valid

    with (output / "sky_comparison_pixels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "arm", "wavelength_nm", "skycalc_e_per_pixel", "native_e_per_pixel", "interior_valid"])
        writer.writerows(all_rows)
    np.savez_compressed(output / "sky_comparison_arrays.npz", **npz_arrays)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True, layout="constrained")
        for row, name in enumerate(("dark", "bright")):
            for arm in range(3):
                ax = axes[row, arm]
                key = f"{name}_arm{arm}"
                valid = npz_arrays[f"{key}_valid"]
                wave = npz_arrays[f"{key}_wave_nm"]
                ax.plot(wave[valid], npz_arrays[f"{key}_skycalc_e"][valid],
                        color="#0072B2", lw=0.65, label="ESO SkyCalc reference")
                ax.plot(wave[valid], npz_arrays[f"{key}_native_e"][valid],
                        color="#D55E00", lw=0.65, linestyle="--", label="ETC built-in sky")
                ax.set_yscale("log")
                ax.set_xlabel("Vacuum wavelength (nm)")
                ax.set_title(f"{name.capitalize()} — arm {arm}")
                ax.grid(alpha=0.18)
                if arm == 0:
                    ax.set_ylabel("Sky electrons / spectral pixel / 750 s")
                if row == 0 and arm == 0:
                    ax.legend(fontsize=7)
        fig.suptitle("Reference model comparison — neither is a calibrated Lenghu sky")
        fig.savefig(output / "sky_comparison.png", dpi=200, facecolor="white")
        fig.savefig(output / "sky_comparison.pdf", facecolor="white")
        plt.close(fig)

    metadata = {
        "purpose": "Unscaled ESO 3060 m SkyCalc vs ETC native sky; neither calibrated to Lenghu",
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in sky_paths.items()},
        "spec_file": str(calc.spec_file), "spec_sha256": sha256(calc.spec_file),
        "exposure_s": EXPOSURE_S, "field_angle_deg": FIELD_ANGLE_DEG,
        "seeing_800_arcsec": SEEING_800_ARCSEC, "zenith_angle_deg": ZENITH_DEG,
        "airmass_source_secant": AIRMASS,
        "airmass_native_sky_spherical": float(1 / math.sqrt(1 - 0.96 * math.sin(math.radians(ZENITH_DEG))**2)),
        "fiber_radius_arcsec": float(radius_arcsec), "fiber_solid_angle_arcsec2": float(solid_angle_arcsec2),
        "moon_bright_illumination": 0.5, "moon_bright_altitude_deg": 45.0,
        "moon_target_separation_deg": 60.0, "moon_dark_altitude_deg": -45.0,
        "raw_sky_photons": "bin-integrated at ground, no repeat atmospheric extinction or extra moon",
        "lsf_approximation": "theoretical ETC 33-diagonal matrix; pixel-bin-center approximation for subpixel sky features; first and last 16 pixels masked",
        "counts_scope": "sky electrons only, including model neighboring traces; no detector/read/dark/systematic terms",
        "integration_self_checks": "constant and triangular spectra passed",
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "elapsed_seconds": round(time.monotonic() - start, 3), "summaries": summaries,
    }
    (output / "sky_comparison_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"elapsed_seconds": metadata["elapsed_seconds"], "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
