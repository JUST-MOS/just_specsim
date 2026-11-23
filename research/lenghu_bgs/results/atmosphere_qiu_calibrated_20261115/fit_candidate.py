#!/usr/bin/env python3
"""Fit conditional ESO-derived extinction candidates to Qiu Table 4.

Products are research-only alternatives; no previous simulation is modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import astropy
import scipy
from astropy.io import fits
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "pilot_4200"))
from lenghu_etc import (  # noqa: E402
    AIRMASS, ESO_3060_PRESSURE_HPA, ESO_ATMOSPHERE, LENGHU_PRESSURE_HPA,
    QIU_AIRMASS, QIU_CSV, QiuObservedAtmosphere,
)

KNOTS = np.arange(400.0, 791.0, 30.0)
FIT_X = np.linspace(1.0, 2.0, 7)
X_WEIGHTS = (FIT_X - FIT_X.mean()) / np.sum((FIT_X - FIT_X.mean()) ** 2)
R_QIU = 600.0
SIGMA_FLOOR = 0.005  # mag/airmass; chosen regularization, not a measured error
SMOOTH_SCALE = 0.5
PRIOR_SCALE = 3.0
LOWER, UPPER = -5.0, 2.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def knot_basis(wave: np.ndarray) -> np.ndarray:
    """Clamped cubic basis, with zero derivative and constant continuation."""
    spline = CubicSpline(KNOTS, np.eye(len(KNOTS)), axis=0, bc_type="clamped")
    return spline(np.clip(wave, KNOTS[0], KNOTS[-1]))


def atmosphere_components():
    original = QiuObservedAtmosphere()
    with fits.open(ESO_ATMOSPHERE, memmap=False) as h:
        data = h[1].data
        molecular = np.array(data["trans_ma"], dtype=float)
        ozone = np.array(data["trans_o3"], dtype=float)
        mie = np.array(data["trans_ms"], dtype=float)
        rayleigh = np.array(data["trans_rs"], dtype=float) ** (
            LENGHU_PRESSURE_HPA / ESO_3060_PRESSURE_HPA)
    for part in (molecular, ozone, mie, rayleigh):
        if np.any(~np.isfinite(part)) or np.any(part <= 0) or np.any(part > 1):
            raise ValueError("Invalid ESO transmission component")
    np.testing.assert_allclose(molecular * ozone * mie * rayleigh, original.reference_trans,
                               rtol=1e-12, atol=1e-14)
    return original, molecular, rayleigh, ozone * mie


class BinnedSlopeModel:
    """Approximate Qiu k by fitting b+kX to modelled bin means at seven X.

    R~600 is approximated by a Gaussian line spread function. Within each
    bin the stellar spectrum is taken as locally flat. The reference optical
    depths are scaled linearly with X, which is a model assumption.
    """
    def __init__(self, original, molecular, rayleigh, adjustable, convention):
        if convention not in {"center", "edge"}:
            raise ValueError(convention)
        wave = original.reference_wave_nm
        step = np.median(np.diff(wave))
        if not np.allclose(np.diff(wave), step, rtol=1e-8, atol=1e-9):
            raise ValueError("ESO wavelength grid is not uniform")
        grid, weight, starts = [], [], []
        for label in original.wave_nm:
            width = 10.0 if label < 650 else 5.0
            lo = label if convention == "edge" else label - width / 2
            hi = lo + width
            margin = 6 * hi / (R_QIU * 2.354820045)
            xx = wave[(wave > lo - margin) & (wave < hi + margin)]
            sigma = xx / (R_QIU * 2.354820045)
            weights = ndtr((hi - xx) / sigma) - ndtr((lo - xx) / sigma)
            if np.sum(weights) <= 0:
                raise ValueError("empty line-spread kernel")
            weights /= np.sum(weights)
            starts.append(len(grid))
            grid.extend(xx)
            weight.extend(weights)
        self.grid = np.array(grid)
        self.weight = np.array(weight)
        self.starts = np.array(starts)
        self.tau_fixed = -np.log(np.interp(self.grid, wave, molecular * rayleigh))
        self.tau_adjustable = -np.log(np.interp(self.grid, wave, adjustable))
        self.basis = knot_basis(self.grid)
        self.k = original.k
        self.sigma = original.sigma_k
        self.effective_sigma = np.hypot(self.sigma, SIGMA_FLOOR)

    def predict(self, nodes: np.ndarray) -> np.ndarray:
        scale = np.exp(self.basis @ nodes)
        tau = self.tau_fixed + self.tau_adjustable * scale
        trans = np.exp(-tau[:, None] * (FIT_X / AIRMASS)[None, :])
        means = np.add.reduceat(trans * self.weight[:, None], self.starts, axis=0)
        if np.any(means <= 0) or np.any(means > 1 + 1e-10):
            raise ValueError("Invalid model bin-averaged transmission")
        extinction_mag = -2.5 * np.log10(means)
        return extinction_mag @ X_WEIGHTS

    def residual(self, nodes: np.ndarray, train: np.ndarray) -> np.ndarray:
        data_residual = (self.predict(nodes) - self.k) / self.effective_sigma
        return np.r_[data_residual[train], np.diff(nodes, 2) / SMOOTH_SCALE, nodes / PRIOR_SCALE]

    def fit(self, train: np.ndarray | None = None):
        if train is None:
            train = np.ones(len(self.k), dtype=bool)
        result = least_squares(self.residual, np.zeros(len(KNOTS)), args=(train,),
                               bounds=(LOWER, UPPER), xtol=1e-8, ftol=1e-8,
                               gtol=1e-8, max_nfev=250)
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Fit failed: {result.message}")
        return result


def evaluate_full(wave, molecular, rayleigh, adjustable, nodes, red_tail="constant"):
    spline = CubicSpline(KNOTS, nodes, bc_type="clamped")
    s = spline(np.clip(wave, KNOTS[0], KNOTS[-1]))
    if red_tail == "relax_870":
        u = np.clip((wave - 790.0) / 80.0, 0, 1)
        factor = 1.0 - (3 * u**2 - 2 * u**3)
        s = np.where(wave > 790, nodes[-1] * factor, s)
    elif red_tail != "constant":
        raise ValueError(red_tail)
    scale = np.exp(s)
    transmission = molecular * rayleigh * adjustable ** scale
    if np.any(~np.isfinite(transmission)) or np.any(transmission < 0) or np.any(transmission > 1):
        raise ValueError("Candidate violates 0 <= T <= 1")
    return transmission, scale


def write_fits(name, original, baseline, candidate, scale, convention, red_tail):
    hdr = fits.Header()
    hdr["ORIGIN"] = "JUST research"
    hdr["AIRMASS"] = (AIRMASS, "fixed geometry; sec(z)")
    hdr["HEIGHT"] = (4200, "target nominal altitude, m")
    hdr["PRESSURE"] = (LENGHU_PRESSURE_HPA, "assumed Lenghu pressure, hPa")
    hdr["QBIN"] = (convention, "unverified Qiu wavelength-bin convention")
    hdr["R_QIU"] = (R_QIU, "Gaussian approximation for Qiu instrument")
    hdr["REDTAIL"] = (red_tail, "assumption for >790 nm")
    hdr["NOTCAL"] = (True, "conditional candidate, not site-validated")
    hdr.add_history("ESO molecular absorption and pressure-scaled Rayleigh fixed.")
    hdr.add_history("Positive smooth multiplier applied to ESO ozone+Mie optical depth.")
    cols = [fits.Column(name="WAVELENGTH", array=original.reference_wave_nm, format="D", unit="nm"),
            fits.Column(name="TRANSMISSION", array=candidate, format="D", unit="1"),
            fits.Column(name="ESO_BASELINE", array=baseline, format="D", unit="1"),
            fits.Column(name="O3_MIE_SCALE", array=scale, format="D", unit="1"),
            fits.Column(name="WITHIN_QIU_LABELS", array=(original.reference_wave_nm >= 400) &
                        (original.reference_wave_nm <= 790), format="L")]
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU.from_columns(cols, name="ATMOSPHERE")]).writeto(
        HERE / name, overwrite=True, checksum=True)


def residual_stats(reference, model, mask=None):
    diff = np.asarray(model) - np.asarray(reference)
    if mask is not None:
        diff = diff[mask]
    return {"n": int(len(diff)), "mae_mag_per_airmass": float(np.mean(np.abs(diff))),
            "median_abs_mag_per_airmass": float(np.median(np.abs(diff))),
            "rmse_mag_per_airmass": float(np.sqrt(np.mean(diff**2))),
            "max_abs_mag_per_airmass": float(np.max(np.abs(diff)))}


def plot_diagnostics(original, models, base_center, fit_center, fit_edge, wave,
                     t_base, t_center, t_edge, t_relax):
    qiu, sigma, labels = original.k, original.sigma_k, original.wave_nm
    qiu_t = 10 ** (-0.4 * qiu * QIU_AIRMASS)
    qiu_error = np.array([qiu_t - 10 ** (-0.4 * (qiu + sigma) * QIU_AIRMASS),
                          10 ** (-0.4 * (qiu - sigma) * QIU_AIRMASS) - qiu_t])
    with plt.rc_context({"font.family": "serif", "font.size": 11.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"}):
        fig, (ax, diff) = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True,
                                       gridspec_kw={"height_ratios": [2.1, 1]})
        fig.subplots_adjust(left=.105, right=.98, top=.85, bottom=.14, hspace=.09)
        fig.suptitle("Qiu-constrained atmospheric extinction: fit diagnostics", fontsize=17, y=.965)
        fig.text(.5, .91, "Qiu bin slopes vs. modelled bin slopes; ESO Rayleigh and molecular absorption held fixed",
                 ha="center", fontsize=11)
        ax.errorbar(labels, qiu, yerr=sigma, fmt="o", color="#0072B2", ms=4, capsize=2,
                    lw=.8, label="Qiu Table 4, ± published σk", zorder=5)
        ax.scatter(labels, base_center, color="0.6", marker="x", s=19, linewidths=.9,
                   label="Uncalibrated ESO, centre-bin assumption")
        ax.scatter(labels, fit_center, facecolors="none", edgecolors="#D55E00", marker="s", s=26,
                   linewidths=1, label="Candidate, centre-bin assumption")
        ax.scatter(labels, fit_edge, color="#009E73", marker="+", s=25, linewidths=1,
                   label="Alternative fit, edge-bin assumption")
        ax.set_ylabel("Extinction slope k (mag / airmass)")
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=.97)
        ax.set_ylim(bottom=-0.015)
        diff.axhline(0, color=".35", lw=.9, ls="--")
        diff.errorbar(labels, fit_center-qiu, yerr=sigma, fmt="s", color="#D55E00", ms=3.2,
                      lw=.7, capsize=1.4, label="Candidate − Qiu (centre)")
        diff.scatter(labels, fit_edge-qiu, color="#009E73", marker="+", s=22, linewidths=1,
                     label="Alternative − Qiu (edge)")
        diff.set_ylabel("Model − Qiu\n(mag / airmass)")
        diff.set_xlabel("Qiu wavelength label (nm)")
        diff.grid(axis="y", alpha=.18)
        diff.legend(loc="upper left", fontsize=8.5, framealpha=.97)
        fig.text(.5, .045, "Both bin conventions and the R≈600 Gaussian response are assumptions;"
                 " error bars show only Qiu σk.\nModel k uses seven approximate airmasses in 1–2;"
                 " finite-bin slope is not direct high-resolution transmission.",
                 ha="center", fontsize=9.4, color=".25")
        for ext in ("png", "pdf"):
            fig.savefig(HERE / f"extinction_fit.{ext}", dpi=200)
        plt.close(fig)

        fig, (full, seam) = plt.subplots(2, 1, figsize=(11.5, 8.0),
                                         gridspec_kw={"height_ratios": [1.3, 1]})
        fig.subplots_adjust(left=.105, right=.98, top=.87, bottom=.13, hspace=.29)
        fig.suptitle("Continuous Qiu-constrained source atmosphere", fontsize=17, y=.975)
        fig.text(.5, .925, "Conditional ESO-derived candidates at airmass 1.2; not measured Lenghu spectra",
                 ha="center", fontsize=11)
        full.plot(wave, t_base, color=".65", lw=.5, label="ESO baseline; Rayleigh pressure-adjusted")
        full.plot(wave, t_center, color="#D55E00", lw=.65, label="Candidate: centre bins, constant red tail")
        full.plot(wave, t_relax, color="#009E73", lw=.65, ls="--", label="Alternative: return to ESO by 870 nm")
        full.errorbar(labels, qiu_t, yerr=qiu_error, fmt="o", ms=3, lw=.7, capsize=1.5,
                      color="#0072B2", label="Qiu effective transmission ± σk", zorder=5)
        full.set_xlim(365, 925)
        full.set_ylim(0, 1.075)
        full.set_ylabel("Source transmission")
        full.set_xlabel("Wavelength (nm)")
        full.grid(axis="y", alpha=.16)
        full.legend(loc="lower left", fontsize=8.5, framealpha=.98)
        sub = (wave >= 775) & (wave <= 810)
        seam.plot(wave[sub], t_base[sub], color=".50", lw=1, label="ESO baseline")
        seam.plot(wave[sub], t_center[sub], color="#D55E00", lw=1.6, label="Candidate: centre bins")
        seam.plot(wave[sub], t_edge[sub], color="#0072B2", lw=1.2, ls="--", label="Alternative: edge bins")
        seam.plot(wave[sub], t_relax[sub], color="#009E73", lw=1.4, ls=":", label="Alternative: red-tail relaxation")
        qsel = (labels >= 775) & (labels <= 790)
        seam.errorbar(labels[qsel], qiu_t[qsel], yerr=qiu_error[:, qsel], fmt="o", ms=4,
                      color="#333333", capsize=2, lw=.8, label="Qiu effective T ± σk")
        seam.axvline(790, color=".5", ls=":", lw=1)
        seam.set_xlim(775, 810)
        seam.set_ylim(.90, 1.03)
        seam.set_xlabel("Wavelength (nm)")
        seam.set_ylabel("Source transmission\n(zoomed scale)")
        seam.grid(alpha=.16)
        seam.legend(loc="lower left", ncol=2, fontsize=8.2, framealpha=.97)
        fig.text(.5, .035, "Vertical line: last Qiu table label, not a model switch."
                 "  Beyond 790 nm the red-tail behaviour is assumed.\n"
                 "Qiu effective T uses 10$^{-0.4kX}$; the published k alone does not give absolute transmission.",
                 ha="center", fontsize=9.3, color=".25")
        for ext in ("png", "pdf"):
            fig.savefig(HERE / f"continuous_atmosphere.{ext}", dpi=200)
        plt.close(fig)


def main():
    original, molecular, rayleigh, adjustable = atmosphere_components()
    wave = original.reference_wave_nm
    baseline = original.reference_trans
    models = {name: BinnedSlopeModel(original, molecular, rayleigh, adjustable, name)
              for name in ("center", "edge")}
    results = {name: model.fit() for name, model in models.items()}
    base = {name: model.predict(np.zeros(len(KNOTS))) for name, model in models.items()}
    pred = {name: models[name].predict(results[name].x) for name in models}
    primary, primary_scale = evaluate_full(wave, molecular, rayleigh, adjustable, results["center"].x)
    edge, edge_scale = evaluate_full(wave, molecular, rayleigh, adjustable, results["edge"].x)
    relax, relax_scale = evaluate_full(wave, molecular, rayleigh, adjustable, results["center"].x, "relax_870")
    for filename, tr, scale, conv, tail in [
        ("atmosphere_center_constant.fits", primary, primary_scale, "center", "constant"),
        ("atmosphere_edge_constant.fits", edge, edge_scale, "edge", "constant"),
        ("atmosphere_center_relax870.fits", relax, relax_scale, "center", "relax870"),
    ]:
        write_fits(filename, original, baseline, tr, scale, conv, tail)

    holdout = np.arange(len(original.k)) % 5 == 2
    cv = models["center"].fit(~holdout)
    cv_pred = models["center"].predict(cv.x)
    names = ["wavelength_label_nm", "qiu_k", "qiu_sigma_k", "baseline_center", "candidate_center",
             "baseline_edge", "candidate_edge", "heldout_cv", "cv_prediction_center"]
    table = HERE / "extinction_fit_comparison.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in zip(original.wave_nm, original.k, original.sigma_k, base["center"], pred["center"],
                       base["edge"], pred["edge"], holdout, cv_pred):
            writer.writerow(row)

    plot_diagnostics(original, models, base["center"], pred["center"], pred["edge"], wave,
                     baseline, primary, edge, relax)
    idx790 = int(np.argmin(np.abs(wave-790)))
    smooth_center = CubicSpline(KNOTS, results["center"].x, bc_type="clamped")
    report = {
        "status": "conditional candidate, not measured or calibrated site truth",
        "source_qiu": "https://arxiv.org/html/2606.18804v1#S4.T4",
        "qiu_csv_sha256": digest(QIU_CSV), "eso_fits_sha256": digest(ESO_ATMOSPHERE),
        "code_sha256": {name: digest(HERE/name) for name in ("fit_candidate.py", "etc_adapter.py", "verify_candidate.py")},
        "software_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "scipy": scipy.__version__, "astropy": astropy.__version__,
                              "matplotlib": matplotlib.__version__},
        "pressure_lenghu_hpa": LENGHU_PRESSURE_HPA, "pressure_eso_hpa": ESO_3060_PRESSURE_HPA,
        "airmass_reference": AIRMASS, "airmass_qiu_same_zenith": QIU_AIRMASS,
        "slope_forward_airmasses": FIT_X.tolist(), "assumed_qiu_resolving_power": R_QIU,
        "sigma_floor_mag_per_airmass": SIGMA_FLOOR,
        "knot_wavelengths_nm": KNOTS.tolist(), "smooth_second_difference_scale": SMOOTH_SCALE,
        "weak_log_scale_prior": PRIOR_SCALE, "log_scale_bounds": [LOWER, UPPER],
        "parameterization": "T = T_molecular * T_rayleigh_pressure * (T_ozone*T_mie)^exp(spline)",
        "center_assumption": "Qiu labels are bin centers; wavelengths <650 use 10 nm, >=650 use 5 nm; 650 is chosen",
        "edge_assumption": "Qiu labels are left edges; wavelengths <650 use 10 nm, >=650 use 5 nm; 650 is chosen",
        "primary_extrapolation": "clamped spline evaluated at 400 and 790 outside those labels; red correction remains constant",
        "red_sensitivity": "log correction smoothly relaxed to ESO baseline by 870 nm",
        "fit_nodes": {key: value.x.tolist() for key, value in results.items()},
        "fit_optimizer": {key: {"success": bool(value.success), "nfev": int(value.nfev),
                                 "cost": float(value.cost), "message": str(value.message)}
                          for key, value in results.items()},
        "full_sample_stats": {name: {"baseline": residual_stats(original.k, base[name]),
                                      "candidate": residual_stats(original.k, pred[name])} for name in models},
        "single_interleaved_holdout": {
            "note": "Every fifth table row held out, no independent nights; diagnostic only",
            "baseline": residual_stats(original.k, base["center"], holdout),
            "refit": residual_stats(original.k, cv_pred, holdout)},
        "at_790_nm": {"qiu_k": float(original.k[-1]), "qiu_sigma_k": float(original.sigma_k[-1]),
                       "baseline_T_at_label": float(baseline[idx790]), "candidate_T_at_label": float(primary[idx790]),
                       "alternative_edge_T_at_label": float(edge[idx790]),
                       "rayleigh_only_k_at_label": float(-2.5*np.log10(rayleigh[idx790])/AIRMASS),
                       "continuum_scale_spline_derivative": float(smooth_center(790, 1))},
        "limitations": ["Qiu k values are fitted slopes; b and cross-bin covariance are unavailable",
                        "label convention and wavelength air/vacuum convention not established",
                        "fixed 3060m molecular absorption is not calibrated to Lenghu; airmass powers and R=600 Gaussian are approximations",
                        "ozone and aerosol are scaled together, not separately identified",
                        "Qiu data do not constrain >790nm; red continuation is a model choice",
                        "no propagation of model, wavelength, extinction covariance or observational systematics",
                        "no Redrock exposure scan rerun and no 4200m sky radiance recalibration"],
    }
    output_names = ["atmosphere_center_constant.fits", "atmosphere_edge_constant.fits",
                    "atmosphere_center_relax870.fits", table.name,
                    "extinction_fit.png", "extinction_fit.pdf", "continuous_atmosphere.png", "continuous_atmosphere.pdf"]
    report["output_sha256"] = {name: digest(HERE/name) for name in output_names}
    (HERE / "fit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"stats": report["full_sample_stats"], "holdout": report["single_interleaved_holdout"],
                      "at_790_nm": report["at_790_nm"]}, indent=2))


if __name__ == "__main__":
    main()
