"""Build the JUST BGS effective-depth research inputs and summarize results.

Run using the project's verified desi environment from the repository root.
No previously frozen input or Redrock result is overwritten or altered.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import astropy
from astropy.io import fits
from astropy.table import Table
from desispec.resolution import Resolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import binary_dilation, median_filter
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
POP = STUDY / "population_4200_camera_overlap"
LEADER = STUDY / "leader_195_205_20261116"
sys.path[:0] = [str(ROOT / "py"), str(STUDY / "pilot"), str(STUDY / "pilot_4200"),
                str(STUDY / "sky/height_4200"), str(POP), str(HERE)]
from just_etc import JUSTExposureTimeCalculator, ETC_py_optimized as eng
from extrapolate_4200 import COMPONENTS, make_model
from lenghu_etc import QiuObservedAtmosphere, LENGHU_PRESSURE_HPA, ESO_3060_PRESSURE_HPA
from refine_sky_sampling import apply_sky, setup_arm
from simulate_batch import pixel_average, source_negative_mask, background_variance
from effective_model import DepthModel, T0

H = 6.62607015e-27  # erg s
C_A = 2.99792458e18  # Angstrom / s
SEEING_NOMINAL_800 = 1.1 * (800. / 635.5)**(-.2)
FILTER = ROOT / "doc/validation/desi_bgs_comparison/cache/decam2014-r.ecsv"
SKYFILES = [HERE / (tag + "_zenith_0p01nm.fits")
            for tag in ("lasilla_2400", "paranal_2640", "anchor_3060")]
INSTRUMENT = POP / "output/instrument_reference_4200.npz"
SAMPLE = POP / "input/frozen_sample.csv"
SPECTRA = POP / "input/normalized_sources.fits"
CURVES19 = LEADER / "bgs_like_r_lt_19p5_curves.csv"
CURVES205 = LEADER / "extended_20p5/analysis_20261119/success_curves.csv"
RESULTS205 = LEADER / "extended_20p5/analysis_20261119/individual_results.csv"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sky_r_ab(wave_nm, photon_flux):
    """Synthetic DECam-r AB surface brightness, input photon/nm/m2/s/arcsec2."""
    tab = Table.read(FILTER)
    fw = np.asarray(tab["wavelength"], dtype=float)*10.
    fr = np.asarray(tab["response"], dtype=float)
    w = np.asarray(wave_nm)*10.
    rr = np.interp(w, fw, fr, left=0., right=0.)
    coverage = np.trapz(rr/w, w)/np.trapz(fr/fw, fw)
    if coverage < .999:
        raise ValueError(f"incomplete r filter coverage {coverage}")
    # SkyCalc: photons/s/m2/micron/arcsec2 -> erg/s/cm2/A/arcsec2.
    flam = np.asarray(photon_flux)*H*C_A/w*1e-8
    fnu = np.trapz(flam*w*rr, w)/(C_A*np.trapz(rr/w, w))
    return float(-2.5*np.log10(fnu/(3631e-23)))


def read_eso(path):
    with fits.open(path, memmap=False) as h:
        h.verify("exception")
        t = h[1].data
        names = ("lam", "flux", "trans", "trans_ma", "trans_o3", "trans_rs", "trans_ms", *COMPONENTS)
        return {k: np.asarray(t[k], dtype=float).copy() for k in names}


def sky_reference():
    sources = [read_eso(p) for p in SKYFILES]
    for source in sources:
        np.testing.assert_allclose(source["flux"], sum(source[c] for c in COMPONENTS), rtol=1e-10, atol=1e-9)
    for source in sources[1:]:
        np.testing.assert_array_equal(source["lam"], sources[0]["lam"])
    sky = make_model(sources, "log")
    raw_mag = sky_r_ab(sky["lam"], sky["flux"])
    scale = 10**(-.4*(21.07-raw_mag))
    scaled = sky["flux"]*scale
    check_mag = sky_r_ab(sky["lam"], scaled)
    if abs(check_mag-21.07) > 1e-10:
        raise ValueError("reference sky normalization failed")
    primary = fits.PrimaryHDU()
    primary.header["ALT_TGT"] = (4200, "m; conditional altitude extrapolation")
    primary.header["AIRMASS"] = 1.0
    primary.header["METHOD"] = "log"
    primary.header["NOTMEAS"] = (True, "model, not measured Lenghu sky")
    primary.header["R_AB"] = (21.07, "DECam-r mag arcsec^-2, imposed")
    primary.header["SCALE"] = (scale, "grey normalization to imposed r sky")
    primary.header["R_RAW"] = (raw_mag, "unscaled extrapolated zenith r sky")
    primary.header["GUARDT"] = .01
    primary.header["NGUARD"] = int(sky["guard_mask"].sum())
    primary.header.add_history("Three ESO sky components extrapolated to 4200 m then grey scaled to DESI r=21.07.")
    primary.header.add_history("Deep-transmission guard retains 3060 m model in flagged bins; this is not a Lenghu measurement.")
    cols = [fits.Column(name="lam", format="D", unit="nm", array=sky["lam"]),
            fits.Column(name="flux", format="D", unit="ph/s/m2/micron/arcsec2", array=scaled),
            fits.Column(name="GUARD_MASK", format="B", array=sky["guard_mask"].astype(np.uint8))]
    fits.HDUList([primary, fits.BinTableHDU.from_columns(cols, name="SKY")]).writeto(
        HERE / "zenith_4200m_desi_r21p07.fits", overwrite=True, checksum=True)
    return sky, sources[-1], scale, {"raw_r_ab": raw_mag, "nominal_r_ab": check_mag,
                                     "grey_scale_raw_to_nominal": scale,
                                     "guarded_sky_samples": int(sky["guard_mask"].sum())}


def atm_x1(wave_nm, qiu, zenith_source):
    """Qiu X=1; outside its range use ESO X=1 pressure-adjusted Rayleigh."""
    w = np.asarray(wave_nm)
    src = zenith_source
    trans = src["trans_ma"]*src["trans_o3"]*src["trans_ms"]*src["trans_rs"]**(
        LENGHU_PRESSURE_HPA/ESO_3060_PRESSURE_HPA)
    fallback = np.interp(w, src["lam"], trans)
    k = np.interp(w, qiu.wave_nm, qiu.k)
    # Same curved-Earth airmass approximation used in the frozen adapter.
    x_qiu = 1./(1.+.5057*(6.0799+90.)**(-1.6364))
    return np.where((w >= qiu.wave_nm[0]) & (w <= qiu.wave_nm[-1]),
                    10**(-.4*k*x_qiu), fallback)


def training_weights(inst, training):
    with fits.open(SPECTRA, memmap=True, checksum=True) as h:
        wave = np.asarray(h["WAVELENGTH"].data, dtype=float)
        indices = training.source_row.to_numpy(int)
        if not np.array_equal(h["TARGETID"].data[indices], training.TARGETID.to_numpy()):
            raise ValueError("calibration target IDs do not match model spectra")
        spectra = np.asarray(h["FLUX_1E17"].data[indices], dtype=float)
    arrays, detail = {}, {}
    for arm in "BRZ":
        w = 10.*inst[f"{arm}_wave_nm"]
        edges = inst[f"{arm}_edges_aa"]
        res = Resolution(inst[f"{arm}_lsf_just"])
        step = float(np.median(np.diff(w)))
        window = max(3, int(round(100./step)))
        if window % 2 == 0:
            window += 1
        weight_sum, flux_sum = np.zeros(len(w)), np.zeros(len(w))
        valid_count = np.zeros(len(w), dtype=int)
        for row, f in zip(training.itertuples(index=False), spectra):
            scale = 10**(-.4*(19.5-row.r_dered))
            bins = pixel_average(wave, np.maximum(f, 0.)*scale, edges)
            signal = np.asarray(res.dot(bins))
            smooth = median_filter(signal, size=window, mode="nearest")
            # Negative reference flux is never used to define features. The
            # mask is dilated to cover both LSF support and the smoothing window.
            neg = source_negative_mask(wave, f, w)
            neg = binary_dilation(neg, iterations=window//2) if neg.any() else neg
            valid = ~neg & np.isfinite(signal) & (signal >= 0)
            weight_sum[valid] += (signal[valid]-smooth[valid])**2
            flux_sum[valid] += signal[valid]
            valid_count[valid] += 1
        enough = valid_count >= math.ceil(.5*len(training))
        feature = np.divide(weight_sum, valid_count, out=np.zeros_like(weight_sum), where=enough)
        meanflux = np.divide(flux_sum, valid_count, out=np.zeros_like(flux_sum), where=enough)
        mask = (inst[f"dark_{arm}_mask"] | inst[f"bright_{arm}_mask"] |
                (inst[f"{arm}_flat_response_750_e"] <= 1e-6 * np.max(inst[f"{arm}_flat_response_750_e"])) |
                ~enough)
        if not np.any(feature[~mask] > 0) or np.any(meanflux[~mask] <= 0):
            raise ValueError(f"invalid fiducial flux or feature weights in {arm}")
        arrays.update({f"{arm}_feature_weight": feature, f"{arm}_continuum": meanflux,
                       f"{arm}_valid": (~mask).astype(np.uint8),
                       f"{arm}_training_count": valid_count})
        detail[arm] = {"pixels": len(w), "valid_pixels": int((~mask).sum()),
                       "median_filter_width_A": window*step,
                       "minimum_training_count_valid": int(valid_count[~mask].min()),
                       "maximum_training_count_valid": int(valid_count[~mask].max())}
    return arrays, detail


def geometry(calc, inst, arm, seeing, radius=1.5):
    calc.set_obs_conditions(seeing_fwhm_800=seeing)
    obs, sp = calc._make_obs(), calc._spectro
    nodes = inst[f"{arm}_geo_wave_nodes_nm"]
    y = np.array([eng.gsGeometricThroughput(sp, obs, float(w), radius, .03, 0., 0)
                  for w in nodes])
    if np.any(y <= 0) or np.any(~np.isfinite(y)):
        raise ValueError("invalid geometric response")
    w = inst[f"{arm}_wave_nm"]
    return np.interp(w, nodes, y)/np.interp(w, nodes, inst[f"{arm}_geo0_nodes"])


def build_inputs():
    sky, eso_zenith, scale, sky_meta = sky_reference()
    design = json.loads((POP / "input/frozen_design.json").read_text())
    if sha(SAMPLE) != design["sample_sha256"] or sha(SPECTRA) != design["normalized_sources_sha256"]:
        raise ValueError("frozen sample or source spectra changed")
    validation = json.loads((POP / "output/instrument_validation.json").read_text())
    if validation["status"] != "passed" or sha(INSTRUMENT) != validation["instrument_sha256"]:
        raise ValueError("frozen instrument cache changed")
    with np.load(INSTRUMENT, allow_pickle=False) as f:
        inst = {k: f[k].copy() for k in f.files}
    cat = pd.read_csv(SAMPLE)
    train = cat[(cat.split == "calibration") & (cat.r_dered < 19.5) &
                (cat.VI_Z > .13) & (cat.VI_Z < .37)].copy()
    test = cat[(cat.split == "test") & (cat.r_dered < 19.5)].copy()
    if len(train) != 110 or len(test) != 578 or set(train.TARGETID) & set(test.TARGETID):
        raise ValueError("unexpected training/test selection")
    arrays, training_meta = training_weights(inst, train)
    ebv = float(test.EBV.median())
    arrays["scenario_median_ebv"] = np.array(ebv)
    qiu = QiuObservedAtmosphere()
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, ebv=0., r_eff=0., decenter=.03,
                            sysfrac=.01, diffuse_stray=.02)
    calc._set_mode()
    calc._configure_spectro(1)
    # The sky setup depends on telescope and extraction but not source seeing.
    sky_reconstruction = {}
    for index, arm in enumerate("BRZ"):
        setup = setup_arm(calc, "dark", index)
        wave = inst[f"{arm}_wave_nm"]
        if not np.allclose(setup[1], wave, rtol=0, atol=1e-10):
            raise ValueError("camera wavelength grids disagree")
        nominal_sky, _, incoming, deposited = apply_sky(sky["lam"], sky["flux"]*scale, setup)
        sky_ref = np.maximum.reduce([np.r_[nominal_sky[0],nominal_sky[:-1]], nominal_sky,
                                     np.r_[nominal_sky[1:],nominal_sky[-1]]])
        sp = calc._spectro
        stray = float(np.sum(nominal_sky)*sp.width[index]*sp.pix[index]/sp.sep[index]/len(wave))
        old_atm, _ = qiu.transmission(wave)
        new_atm = atm_x1(wave, qiu, eso_zenith)
        if np.any(new_atm <= 0) or np.any(old_atm <= 0):
            raise ValueError("zero transmission on a detector pixel")
        ratio_scen = geometry(calc, inst, arm, .8)
        ratio_nom = geometry(calc, inst, arm, SEEING_NOMINAL_800)
        flat = inst[f"{arm}_flat_response_750_e"]
        arrays[f"nominal_{arm}_response_no_dust"] = flat*(new_atm/old_atm)*ratio_nom
        for scene in ("dark", "bright"):
            arrays[f"{scene}_{arm}_response_no_dust"] = flat*ratio_scen
            arrays[f"{scene}_{arm}_sky"] = inst[f"{scene}_{arm}_sky_e"]
            arrays[f"{scene}_{arm}_sky_ref"] = inst[f"{scene}_{arm}_sky_ref_e"]
            arrays[f"{scene}_{arm}_stray"] = inst[f"{scene}_{arm}_stray_e"]
        arrays[f"nominal_{arm}_sky"] = nominal_sky
        arrays[f"nominal_{arm}_sky_ref"] = sky_ref
        arrays[f"nominal_{arm}_stray"] = np.array(stray)
        arrays[f"{arm}_wave_nm"] = wave
        arrays[f"{arm}_kgal"] = inst[f"{arm}_kgal_per_ebv"]
        arrays[f"{arm}_read"] = inst[f"{arm}_read_var_per_exp"]
        arrays[f"{arm}_dark"] = inst[f"{arm}_dark_e_per_750"]
        arrays[f"{arm}_nominal_atm"] = new_atm
        arrays[f"{arm}_scenario_atm"] = old_atm
        sky_reconstruction[arm] = {"reference_incoming_e": incoming,
                                   "reference_deposited_e": deposited,
                                   "nominal_sky_median_e_per_750": float(np.median(nominal_sky)),
                                   "nominal_atm_min": float(new_atm.min()),
                                   "nominal_atm_max": float(new_atm.max())}
    np.savez_compressed(HERE / "depth_inputs.npz", **arrays)
    train[["TARGETID", "VI_Z", "r_dered", "SHAPE_R", "EBV", "source_row"]].to_csv(
        HERE / "weight_training_targets.csv", index=False)
    return inst, train, test, {"sky": sky_meta, "training": training_meta,
                              "sky_deposition": sky_reconstruction,
                              "scenario_median_ebv": ebv,
                              "nominal_seeing_800_arcsec": SEEING_NOMINAL_800}


def read_curves():
    a = pd.read_csv(CURVES19).rename(columns={"success":"k_success",
        "six_times_bonferroni_lower":"lower", "time_s":"actual_s"})
    a["sample"] = "r<19.5"
    b = pd.read_csv(CURVES205)
    b = b[(b.group_type == "all") & (b.group == "all")].copy()
    b = b.rename(columns={"cp95_lower_bonferroni_times":"lower", "time_s":"actual_s"})
    b["sample"] = "r<20.5"
    cols = ["sample", "sky", "actual_s", "n", "k_success", "success_fraction", "lower"]
    return pd.concat([a[cols], b[cols]], ignore_index=True)


def analyze(model):
    curves = read_curves()
    curves["effective_s"] = [model.effective_time(sky, float(t))
                              for sky,t in zip(curves.sky,curves.actual_s)]
    curves.to_csv(HERE / "mapped_recovery_curves.csv", index=False)
    goals = []
    for sample in ("r<19.5", "r<20.5"):
        c = curves[(curves['sample'] == sample) & (curves.sky == "dark")].sort_values("actual_s")
        for criterion, col in (("point", "success_fraction"), ("bonferroni_lower", "lower")):
            passing = c[c[col] >= .95]
            if len(passing) == 0:
                raise ValueError(f"no dark threshold found for {sample} {criterion}")
            row = passing.iloc[0]
            t_nom = float(row.effective_s)
            pred_bright = model.exposure_for_depth("bright", t_nom)
            goals.append({"sample":sample, "criterion":criterion, "dark_anchor_s":int(row.actual_s),
                          "dark_success":int(row.k_success), "n":int(row.n),
                          "dark_success_fraction":float(row.success_fraction), "dark_lower":float(row.lower),
                          "estimated_nominal_s":t_nom, "dark_actual_s":float(row.actual_s),
                          "predicted_bright_s":pred_bright})
    goals = pd.DataFrame(goals)
    goals.to_csv(HERE / "estimated_thresholds.csv",index=False)
    grid = np.arange(0., 3601., 50.)
    mapping = []
    for scene in ("dark", "bright"):
        for t in grid:
            mapping.append({"sky":scene, "actual_s":t,
                            "effective_s":model.effective_time(scene,t)})
    pd.DataFrame(mapping).to_csv(HERE / "time_mapping.csv",index=False)
    return curves, goals


def verification(model, inst, curves, meta):
    checks = {}
    # Nominal = actual at the standard scene for the same split.
    errors = [abs(model.effective_time("nominal",t)-t) for t in (550.,750.,1500.,2250.,3000.,3600.)]
    checks["nominal_identity_max_abs_s"] = max(errors)
    if max(errors) > 1e-3:
        raise ValueError("reference effective-time identity failed")
    for scene in ("dark","bright"):
        for t in (550.,750.,1500.,2250.,3000.,3600.):
            eff = model.effective_time(scene,t)
            predicted = model.exposure_for_depth(scene,eff)
            if abs(predicted-t) > .001:
                raise ValueError(f"time conversion roundtrip failed: {scene} {t} {predicted}")
    # Frozen variance algebra: source varies; background must exactly reproduce
    # the independent sky/detector part saved by the prior simulator.
    max_rel = 0.
    for arm in "BRZ":
        for sky in ("dark","bright"):
            for t,n in ((750,1),(1500,2),(3600,5)):
                x = (t/T0)*(model.d[f"{sky}_{arm}_sky"]+.02*float(model.d[f"{sky}_{arm}_stray"])+
                            float(model.d[f"{arm}_dark"]))+n*float(model.d[f"{arm}_read"])+(
                            .01*(t/T0)*model.d[f"{sky}_{arm}_sky_ref"])**2
                y = background_variance(inst,arm,sky,t,n)
                max_rel = max(max_rel,float(np.max(np.abs(x-y)/np.maximum(y,1e-30))))
    checks["background_variance_max_relative_difference"] = max_rel
    if max_rel > 1e-12:
        raise ValueError("frozen background variance mismatch")
    # Independently recompute the selected one-sided Bonferroni bounds.
    max_ci = 0.
    for row in curves.itertuples():
        ntest = 6 if row.sample == "r<19.5" else 5
        expected = 0. if row.k_success == 0 else beta.ppf(.05/ntest, row.k_success, row.n-row.k_success+1)
        max_ci = max(max_ci, abs(row.lower-expected))
    checks["adjusted_lower_max_abs_difference"] = max_ci
    if max_ci > 1e-10:
        raise ValueError("confidence bound mismatch")
    # Feature positive, finite Q and monotone inside each fixed-read interval.
    for s in ("nominal","dark","bright"):
        for a in (0.,750.,1500.,2250.,3000.):
            times = np.linspace(a+1e-4, a+750., 8)
            vals = np.array([model.information(s,float(t)) for t in times])
            if np.any(~np.isfinite(vals)) or np.any(np.diff(vals) <= 0):
                raise ValueError(f"information not increasing within read interval {s} {a}")
    checks["within_split_monotonicity"] = "passed"
    checks["nominal_sky_decam_r_ab"] = meta["sky"]["nominal_r_ab"]
    checks["status"] = "passed; engineering and numerical checks, not physical calibration"
    (HERE / "verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    return checks


def plot_results(curves):
    plt.rcParams.update({"font.family":"serif","font.size":10,"pdf.fonttype":42})
    fig, axes = plt.subplots(1,2,figsize=(11.0,4.25),sharey=True,layout="constrained")
    for ax, sample in zip(axes,("r<19.5","r<20.5")):
        for sky,color,marker in (("dark","#0072B2","o"),("bright","#D55E00","s")):
            d = curves[(curves['sample']==sample)&(curves.sky==sky)].sort_values("actual_s")
            ax.plot(d.effective_s,d.success_fraction*100,color=color,marker=marker,
                    lw=1.2,ms=4,label=f"{sky.title()} point estimate")
            ax.vlines(d.effective_s,d.lower*100,d.success_fraction*100,color=color,lw=1,alpha=.75)
            ax.plot(d.effective_s,d.lower*100,marker="_",linestyle="none",color=color,ms=6)
        ax.axhline(95,color="#333333",linestyle="--",lw=1)
        ax.set_title(f"{sample}, n={int(d.n.iloc[0])}")
        ax.set_xlabel("Model-mapped nominal effective time [s]")
        ax.set_xlim(left=0)
        ax.set_ylim(75,100.7)
        ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Redshift recovery [%]")
    axes[0].legend(frameon=False,fontsize=9,loc="lower right")
    fig.suptitle("JUST BGS-like recovery on a provisional effective-time scale",fontsize=12)
    fig.supxlabel("Vertical marks: one-sided 95% lower bounds adjusted over tested times; lines connect tested grid points.",fontsize=8)
    fig.savefig(HERE / "effective_recovery.png",dpi=200,facecolor="white")
    fig.savefig(HERE / "effective_recovery.pdf",facecolor="white")
    plt.close(fig)


def main():
    start = time.monotonic()
    inst, train, test, meta = build_inputs()
    model = DepthModel()
    curves, goals = analyze(model)
    checks = verification(model, inst, curves, meta)
    plot_results(curves)
    meta.update({"status":"conditional model; transfer to equal Redrock success not yet validated",
      "training_n":len(train),"training_selection":"calibration split, r0<19.5, 0.13<VI_Z<0.37",
      "fiducial":"r0=19.5 BASS-r, circular exponential half-light radius 1.5 arcsec",
      "nominal_conditions":"X=1, photometric, EBV=0, DECam-r 21.07 AB arcsec^-2, seeing 1.1 arcsec at 6355 A",
      "scenario_conditions":"X=1.2, seeing 0.8 arcsec at 800 nm, field median EBV, 4200m dark and half-Moon skies",
      "time_rule":"750s maximum subexposure; total electron sum; one read per subexposure; 1% fully correlated sky residual",
      "weight_rule":"mean squared residual from 100A median-filtered calibration model spectra renormalized to r0=19.5; convolved with JUST LSF; per-wavelength negative-model neighborhoods masked",
      "input_sha256":{str(p):sha(p) for p in [*SKYFILES,FILTER,INSTRUMENT,SAMPLE,SPECTRA,CURVES19,CURVES205,RESULTS205,
                       STUDY/"pilot_4200/lenghu_etc.py",HERE/"effective_model.py",Path(__file__)]},
      "software":{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
                  "numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,
                  "astropy":astropy.__version__,"matplotlib":matplotlib.__version__},
      "verification":checks,"elapsed_seconds":time.monotonic()-start})
    (HERE/"model_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(goals.to_string(index=False),flush=True)
    print(json.dumps({"sky":meta["sky"],"training":meta["training"],"verification":checks,
                      "elapsed_seconds":meta["elapsed_seconds"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
