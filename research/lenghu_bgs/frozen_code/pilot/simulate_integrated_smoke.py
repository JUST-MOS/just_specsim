"""Small conditional JUST/Redrock interface trial using synthetic galaxies.

This creates six simulated spectra (two arbitrary BGS templates x three sky
scenarios), all at 2 x 750 s. It is NOT a representative sample, and cannot
measure a redshift success rate or a Lenghu exposure time. The foreground-free
status and instrumental resolution of the source basis need further audit.

SkyCalc is an ESO 3060 m model; gray scaling assumes SQM equals V_AB. JUST's
source atmospheric model is separately its native model, so the two are not a
single site-calibrated atmosphere. The modeled LSF and diagonal/no-Gaussian-
covariance noise are idealized. Source rebinning before the discrete LSF is an
approximation for unresolved lines. Only a guarded, isolated FITS is written.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, vstack
from desispec.resolution import Resolution
from speclite.filters import load_filter

from compare_sky_models import (
    ROOT, STUDY, FIELD_ANGLE_DEG, ZENITH_DEG, SEEING_800_ARCSEC,
    engine, make_resolution, JUSTExposureTimeCalculator,
)
from just_etc.redrock import _build_fibermap

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
SPEC_SIM = Path("/Users/rain/just_specsim")
BASIS = SPEC_SIM / "example/DESI_templates/bgs_templates_v2.3.fits"
NEXP = 2
SINGLE_EXPOSURE_S = 750.0
GAL_EBV = 0.03  # engineering assumption, not an observed target LOS
HALF_LIGHT_RADIUS_ARCSEC = 0.5  # circular exponential model assumption
R_DERED_AB = 20.0
TEMPLATE_SEED = 20261111
NOISE_SEED = 20261112
SOURCES = [
    {"redshift": 0.2, "Mr": -21.0, "gr": 0.5},
    {"redshift": 0.3, "Mr": -21.0, "gr": 0.9},
]
SKIES = ("dark_gray", "bright_gray_all", "bright_gray_base")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_average(wavelength, flux, edges):
    """Integrate a piecewise-linear f_lambda spectrum over wavelength bins."""
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(flux, dtype=float)
    e = np.asarray(edges, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or np.any(np.diff(x) <= 0):
        raise ValueError("invalid intrinsic spectrum")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("nonfinite or negative synthetic source flux; do not clip")
    if e[0] < x[0] or e[-1] > x[-1] or np.any(np.diff(e) <= 0):
        raise ValueError("source spectrum does not cover detector bin edges")
    dx = np.diff(x)
    cumulative = np.concatenate(([0.0], np.cumsum(dx * (y[:-1] + y[1:]) / 2)))
    i = np.clip(np.searchsorted(x, e, side="right") - 1, 0, len(x)-2)
    d = e - x[i]
    primitive = cumulative[i] + y[i]*d + (y[i+1]-y[i])/(2*dx[i])*d*d
    return np.diff(primitive) / np.diff(e)


def calibrated_resolution(resolution, response):
    """Build D(response)^-1 R D(response) in DESI diagonal convention."""
    t = np.asarray(response, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("calibrated resolution requires strictly positive response")
    n = len(t)
    cols = np.arange(n)
    arr = np.zeros_like(resolution.data, dtype=float)
    for k, offset in enumerate(resolution.offsets):
        rows = cols - int(offset)
        good = (rows >= 0) & (rows < n)
        arr[k, good] = resolution.data[k, good] * t[cols[good]] / t[rows[good]]
    return Resolution(arr)


def variance_terms(source_e, sky_e, *, n_exp, sample_factor, read_e, dark_e_s,
                   extraction_width, pixel_um, separation_um, sysfrac, diffuse):
    """Match ETC independent-pixel variance and its non-averaging sky floor."""
    source = np.asarray(source_e, dtype=float)
    sky = np.asarray(sky_e, dtype=float)
    if source.shape != sky.shape or np.any(source < 0) or np.any(sky < 0):
        raise ValueError("invalid per-exposure source/sky electron expectations")
    # The ETC uses max(sky[i-1:i+2]) as the systematic reference.
    sky_ref = np.maximum.reduce([np.r_[sky[0], sky[:-1]], sky, np.r_[sky[1:], sky[-1]]])
    sky_mean_stray = np.sum(sky) * extraction_width * pixel_um / separation_um / len(sky)
    source_poisson = n_exp * sample_factor * source
    sky_poisson = n_exp * sample_factor * sky
    diffuse_var = n_exp * diffuse * sky_mean_stray * sample_factor
    detector_var = n_exp * extraction_width * (dark_e_s * SINGLE_EXPOSURE_S * sample_factor + read_e**2)
    floor_var = (n_exp * sysfrac * sky_ref)**2
    total = source_poisson + sky_poisson + diffuse_var + detector_var + floor_var
    return total, {"source_poisson": source_poisson, "sky_poisson": sky_poisson,
                   "diffuse": diffuse_var, "detector": detector_var, "floor": floor_var}


def self_check():
    """Test photon binning, read variance, and systematic floor scaling."""
    a = pixel_average([0, 1, 2], [1, 3, 1], [0, 1, 2])
    if not np.allclose(a, [2, 2], atol=1e-14):
        raise AssertionError("linear source pixel integration failed")
    def v(n, floor):
        return variance_terms(np.zeros(3), np.full(3, 10.), n_exp=n, sample_factor=1,
                              read_e=2, dark_e_s=0, extraction_width=1, pixel_um=1,
                              separation_um=1, sysfrac=floor, diffuse=0)
    _, one = v(1, 0.1)
    _, two = v(2, 0.1)
    if not np.allclose(two["floor"], 4 * one["floor"]):
        raise AssertionError("systematic sky floor must not average down")
    if not np.allclose(two["detector"], 2 * one["detector"]):
        raise AssertionError("read variance must scale with exposure count")


class SourceOnlyCalculator(JUSTExposureTimeCalculator):
    """Reuse the ETC source response; supply no internal sky variance."""
    def _arm_noise(self, obs, t_exp):
        del obs, t_exp
        return [(np.zeros(int(self._spectro.npix[i])), 1.0)
                for i in range(self._spectro.N_arms)]


def main():
    started = time.monotonic()
    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    response_calc = SourceOnlyCalculator(calc_mode="accurate")
    response_calc.set_obs_conditions(
        seeing_fwhm_800=SEEING_800_ARCSEC, zenith_angle=ZENITH_DEG,
        field_angle=FIELD_ANGLE_DEG, r_eff=HALF_LIGHT_RADIUS_ARCSEC,
        ebv=GAL_EBV, decenter=0.03,
    )

    # The neighboring repository reads its basis via relative paths.  CWD is
    # restored immediately; no files in that checkout are modified.
    sys.path.insert(0, str(SPEC_SIM / "py"))
    old_cwd = Path.cwd()
    try:
        os.chdir(SPEC_SIM / "example")
        from just_specsim.maker import SpectrumMaker
        maker = SpectrumMaker()
        wave, fluxes, meta, objmeta = maker(
            z=np.array([s["redshift"] for s in SOURCES]),
            Mr=np.array([s["Mr"] for s in SOURCES]),
            color=np.array([s["gr"] for s in SOURCES]),
            seed=TEMPLATE_SEED, nocolorcuts=True,
        )
    finally:
        os.chdir(old_cwd)
    wave, fluxes = np.asarray(wave, dtype=float), np.asarray(fluxes, dtype=float)
    del maker
    gc.collect()
    if fluxes.shape != (len(SOURCES), len(wave)):
        raise ValueError("unexpected synthetic galaxy array shape")
    if np.any(~np.isfinite(fluxes)) or np.any(fluxes < 0):
        raise ValueError("synthetic galaxy contains invalid/negative flux")

    band = load_filter("decam2014-r")
    scaled = []
    source_metadata = []
    for i, source in enumerate(SOURCES):
        unit_flux = fluxes[i] * 1e-17 * u.erg / u.s / u.cm**2 / u.AA
        mags = {}
        for method in ("zero", "edge"):
            padded, padded_wave = band.pad_spectrum(unit_flux, wave * u.AA, method=method)
            mags[method] = float(band.get_ab_magnitude(padded, padded_wave))
        scale = 10**(-0.4 * (R_DERED_AB - mags["edge"]))
        true_flux = fluxes[i] * scale
        extinct = np.array([10**(-0.4 * engine.gsGalactic_Alambda__EBV(w/10) * GAL_EBV) for w in wave])
        pad_obs, pad_wave = band.pad_spectrum(true_flux * extinct * 1e-17 * u.erg/u.s/u.cm**2/u.AA,
                                               wave*u.AA, method="edge")
        r_obs = float(band.get_ab_magnitude(pad_obs, pad_wave))
        scaled.append(true_flux)
        source_metadata.append({
            **source, "source_index": i, "r0_ab_edge_padded": R_DERED_AB,
            "r_observed_ab_edge_padded": r_obs, "native_r_ab_edge_padded": mags["edge"],
            "native_r_ab_zero_padded": mags["zero"], "filter_padding_difference_mag": mags["edge"]-mags["zero"],
            "meta_mag": float(meta[i]["MAG"]), "template_id": int(meta[i]["TEMPLATEID"]),
            "stellar_vdisp_km_s": float(objmeta[i]["VDISP"]), "normalization_factor": float(scale),
        })
    scaled = np.asarray(scaled)
    source_hdus = fits.HDUList([
        fits.PrimaryHDU(), fits.ImageHDU(wave, name="WAVELENGTH"),
        fits.ImageHDU(scaled, name="FLUX_1E17"),
    ])
    source_hdus["WAVELENGTH"].header["BUNIT"] = "Angstrom"
    source_hdus["WAVELENGTH"].header.add_history("Observer-frame wavelength; vacuum assumed by the generator.")
    source_hdus["FLUX_1E17"].header["BUNIT"] = "10**-17 erg/(s cm**2 Angstrom)"
    source_hdus.writeto(OUTPUT / "synthetic_input_sources.fits", overwrite=True, checksum=True)

    # T_i is the modeled per-exposure electron count for input f_lambda=1e-17.
    response = response_calc.compute_snr(wave, np.full_like(wave, 1e-17),
                                         t_exp=SINGLE_EXPOSURE_S, n_exp=1)
    sp = response_calc._spectro
    obs = response_calc._make_obs()
    if any(int(sp.Dtype[a]) != 0 for a in range(sp.N_arms)):
        raise ValueError("first-stage noise implementation supports CCD arms only")
    sky = np.load(OUTPUT / "sky_comparison_subpixel_arrays.npz")
    sky_meta = json.loads((OUTPUT / "assumed_sky_sensitivity.json").read_text())
    k = float(sky_meta["gray_scale"])

    trial_info = []
    fibermaps = []
    scores = {"TARGETID": []}
    arm_data = []
    for arm, band_name in enumerate(("B", "R", "Z")):
        w = np.asarray(response[arm]["wave_aa"])
        t = np.asarray(response[arm]["signal"], dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t <= 0):
            raise ValueError(f"nonpositive constant-flux response in arm {arm}")
        edges = 10 * (sp.lmin[arm] + sp.dl[arm] * np.arange(sp.npix[arm]+1))
        resolution, edge_mask, lsf_meta = make_resolution(sp, obs, arm, w)
        calibrated = calibrated_resolution(resolution, t)
        # Verify the forward linear operator for the exact response used here.
        test = np.linspace(0.8, 1.2, len(t))
        lhs = calibrated.dot(test)
        rhs = resolution.dot(t * test) / t
        residual = float(np.max(np.abs(lhs[edge_mask]-rhs[edge_mask]) / np.maximum(np.abs(rhs[edge_mask]), 1e-30)))
        if residual > 1e-10:
            raise AssertionError("response-calibrated resolution matrix failed forward-model check")
        response_valid = edge_mask & (t > 1e-6 * np.max(t))
        dark = sky[f"dark_arm{arm}_skycalc_e"]
        bright = sky[f"bright_arm{arm}_skycalc_e"]
        moon = bright - dark
        if np.min(moon[response_valid]) < -1e-8:
            raise ValueError("negative modeled moon counts")
        sky_choices = {"dark_gray": k*dark, "bright_gray_all": k*bright,
                       "bright_gray_base": k*dark + moon}
        arm_flux, arm_ivar, arm_masks, arm_res = [], [], [], []
        for si, spectrum in enumerate(scaled):
            f = pixel_average(wave, spectrum, edges)
            source_e = np.maximum(np.asarray(resolution.dot(t*f)), 0.0)
            for scenario_index, label in enumerate(SKIES):
                idx = si*len(SKIES) + scenario_index
                var, components = variance_terms(
                    source_e, sky_choices[label], n_exp=NEXP, sample_factor=1.0,
                    read_e=float(sp.read[arm]), dark_e_s=float(sp.dark[arm]),
                    extraction_width=int(sp.width[arm]), pixel_um=float(sp.pix[arm]),
                    separation_um=float(sp.sep[arm]), sysfrac=float(response_calc._obs_params["sysfrac"]),
                    diffuse=float(response_calc._obs_params["diffuse_stray"]),
                )
                valid = response_valid & np.isfinite(var) & (var > 0)
                rng = np.random.default_rng(NOISE_SEED + idx*10 + arm)
                signal_total = NEXP*source_e
                mock = np.zeros(len(t), dtype=float)
                ivar = np.zeros(len(t), dtype=float)
                mock[valid] = (signal_total[valid] + rng.normal(size=valid.sum()) * np.sqrt(var[valid])) / (NEXP*t[valid])
                ivar[valid] = (NEXP*t[valid])**2 / var[valid]
                mask = (~valid).astype(np.uint32)
                arm_flux.append(mock.astype(np.float32))
                arm_ivar.append(ivar.astype(np.float32))
                arm_masks.append(mask)
                arm_res.append(calibrated.data.astype(np.float32))
                if arm == 0:
                    target_id = 910000 + idx
                    fm = _build_fibermap(target_id, 150., 2.)
                    fm["FIBER"] = [idx]
                    fibermaps.append(fm)
                    scores["TARGETID"].append(target_id)
                    trial_info.append({"targetid": target_id, "source_index": si, "sky": label,
                                       "z_true": source_metadata[si]["redshift"],
                                       "r0_ab": R_DERED_AB, "noise_seed_base": NOISE_SEED + idx*10})
        arm_data.append({"name": band_name, "wave": w, "flux": np.asarray(arm_flux),
                         "ivar": np.asarray(arm_ivar), "mask": np.asarray(arm_masks),
                         "resolution": np.asarray(arm_res), "valid": response_valid,
                         "forward_identity_max_rel": residual, "lsf": lsf_meta})
        scores[f"SNR_{band_name}"] = [float(np.median(f[v] * np.sqrt(i[v])))
                                      for f, i, v in zip(arm_flux, arm_ivar, [response_valid]*len(arm_flux))]
        scores[f"INTEG_RAW_FLUX_{band_name}"] = [float(np.mean(f[response_valid])) for f in arm_flux]
        scores[f"MEDIAN_RAW_FLUX_{band_name}"] = [float(np.median(f[response_valid])) for f in arm_flux]

    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "JUST ETC conditional interface trial"
    primary.header["NOTSCI"] = (True, "not a Lenghu exposure/redshift forecast")
    primary.header["SKYREF"] = "ESO3060 SQM-as-V"
    primary.header["LSF"] = "ETC theoretical"
    primary.header["TEXP"] = SINGLE_EXPOSURE_S
    primary.header["NEXP"] = NEXP
    primary.header["EBV"] = GAL_EBV
    primary.header["REFF"] = HALF_LIGHT_RADIUS_ARCSEC
    primary.header.add_history("Input galaxy templates and sky are conditional models, not measured Lenghu data.")
    hdus = [primary, fits.BinTableHDU(vstack(fibermaps), name="FIBERMAP")]
    for arm in arm_data:
        name = arm["name"]
        hdus.extend([
            fits.ImageHDU(arm["wave"], name=f"{name}_WAVELENGTH"),
            fits.ImageHDU(arm["flux"], name=f"{name}_FLUX"),
            fits.ImageHDU(arm["ivar"], name=f"{name}_IVAR"),
            fits.ImageHDU(arm["mask"], name=f"{name}_MASK"),
            fits.ImageHDU(arm["resolution"], name=f"{name}_RESOLUTION"),
        ])
        hdus[-5].header["BUNIT"] = "Angstrom"
        hdus[-4].header["BUNIT"] = "10**-17 erg/(s cm**2 Angstrom)"
        hdus[-3].header["BUNIT"] = "10**34 (s**2 cm**4 Angstrom**2)/erg**2"
    hdus.append(fits.BinTableHDU(Table(scores), name="SCORES"))
    dest = OUTPUT / "integrated_synthetic_smoke.fits"
    fits.HDUList(hdus).writeto(dest, overwrite=True, checksum=True)
    with (OUTPUT / "integrated_truth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_info[0]))
        writer.writeheader()
        writer.writerows(trial_info)
    report = {
        "status": "conditional interface test, not a success-rate or exposure-time estimate",
        "n_targets": len(trial_info), "source_models": source_metadata,
        "spectra": trial_info, "single_exposure_s": SINGLE_EXPOSURE_S, "n_exp": NEXP,
        "total_s": SINGLE_EXPOSURE_S*NEXP, "mw_ebv_assumed": GAL_EBV,
        "half_light_radius_arcsec_assumed": HALF_LIGHT_RADIUS_ARCSEC,
        "sky_status": sky_meta["status"], "sky_gray_scale": k,
        "source_atmosphere": "native ETC model; not site matched to ESO sky",
        "lsf": "ETC theoretical; not measured; source sampled before discrete LSF",
        "source_basis_status": "local just_specsim BGS basis; effective starting LSF and overlap with fitting templates unverified",
        "noise": "ETC modeled independent Gaussian pixel noise, source+sky+read+dark+diffuse+nonaveraging sky floor; no covariance",
        "input_basis": str(BASIS), "input_basis_sha256": sha256(BASIS),
        "input_sources_fits": str(OUTPUT / "synthetic_input_sources.fits"),
        "input_sources_sha256": sha256(OUTPUT / "synthetic_input_sources.fits"),
        "mock_fits": str(dest), "mock_sha256": sha256(dest),
        "forward_operator_max_rel_by_arm": [a["forward_identity_max_rel"] for a in arm_data],
        "valid_response_pixels_by_arm": [int(a["valid"].sum()) for a in arm_data],
        "test": "source pixel integral, read-variance scaling, nonaveraging sky floor, calibrated forward operator passed",
        "elapsed_seconds": round(time.monotonic()-started, 3),
    }
    (OUTPUT / "integrated_synthetic_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
