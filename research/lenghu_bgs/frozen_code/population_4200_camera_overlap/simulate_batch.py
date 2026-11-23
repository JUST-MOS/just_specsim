"""Simulate one frozen subset of independent galaxies for Redrock.

Output contains the requested exposure/sky combinations together, each with
its own TARGETID and independent noise draw. FastSpecFit spectra retain the
DESI LSF; a per-object Gaussian DESI moment approximation is included in the
Redrock resolution operator in addition to the theoretical JUST LSF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from desispec.resolution import Resolution
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
DATA = HERE / "input"
OUT = HERE / "output"
sys.path.insert(0, str(STUDY / "pilot_4200"))
sys.path.insert(0, str(STUDY / "pilot"))
from lenghu_etc import LenghuPilotETC
from simulate_integrated_smoke import calibrated_resolution, pixel_average
from just_etc import ETC_py_optimized as engine

TIMES = {750: [750], 1500: [750,750], 2250: [750,750,750],
         3000: [750,750,750,750], 3600: [750,750,750,750,600]}
SKY = ("dark", "bright")
TREF = 750.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(tid, sky, total, arm):
    return int(hashlib.sha256(f"20261114:{tid}:{sky}:{total}:{arm}".encode()).hexdigest()[:16], 16)


def background_variance(inst, band, sky, total, n_exposures):
    """Independent sky/dark/read variance plus fully correlated sky floor."""
    factor = total/TREF
    sky_e = inst[f"{sky}_{band}_sky_e"]
    sky_ref = inst[f"{sky}_{band}_sky_ref_e"]
    stray = float(inst[f"{sky}_{band}_stray_e"])
    dark = float(inst[f"{band}_dark_e_per_750"])
    read = float(inst[f"{band}_read_var_per_exp"])
    return factor*(sky_e + .02*stray + dark) + n_exposures*read + (.01*factor*sky_ref)**2


def dilate(mask, width):
    return np.convolve(np.asarray(mask, dtype=int), np.ones(2*width+1, dtype=int), mode="same") > 0


def source_negative_mask(wave, flux, centers):
    # Piecewise-linear intervals adjacent to a nonpositive native sample can
    # be affected. Include LSF support and one extra detector pixel. Only these
    # pixels are masked; noisy observed negative flux elsewhere is retained.
    native = (flux < 0).astype(float)
    support = np.interp(centers, wave, native, left=1., right=1.) > 0
    return dilate(support, 17)


def camera_moment_on_model_grid(lsf_wave, sigma, mapping, fallback_sigma):
    """Effective DESI sigma^2 [Å²] and invalid mask on the FastSpecFit grid.

    Reproduce the bracketing cameras and weights of FastSpecFit's sorted
    interpolation, approximating any mixture by its second moment. Missing
    LSF nodes remain invalid; a numerical placeholder is never used in fits.
    """
    model_wave = mapping["model_wave"]
    if lsf_wave.shape != (3,41) or sigma.shape != (3,41):
        raise ValueError("expected three sets of 41 DESI LSF nodes")
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        raise ValueError("invalid finite LSF placeholder")
    var = np.empty((3,len(model_wave)))
    bad = np.empty((3,len(model_wave)), dtype=bool)
    for camera in range(3):
        valid = np.isfinite(sigma[camera]) & (sigma[camera]>0) & (sigma[camera]<=3)
        if not np.any(valid):
            var[camera] = fallback_sigma**2
            bad[camera] = True
        else:
            var[camera] = np.interp(model_wave, lsf_wave[camera,valid], sigma[camera,valid]**2)
            bad[camera] = np.interp(model_wave, lsf_wave[camera], (~valid).astype(float),
                                    left=1., right=1.) > 0
    j = np.arange(len(model_wave))
    left, right, w = mapping["left_camera"], mapping["right_camera"], mapping["right_weight"]
    if np.any((left<0) | (left>2) | (right<0) | (right>2) | (w<0) | (w>1)):
        raise ValueError("invalid FastSpecFit camera mapping")
    variance = (1-w)*var[left,j] + w*var[right,j]
    invalid = ((1-w)>1e-10) & bad[left,j] | (w>1e-10) & bad[right,j]
    return variance, invalid


def resolution_and_mask(just, response, lsf_wave, sigma, mapping, wave_aa, fallback_sigma):
    model_wave = mapping["model_wave"]
    variance, invalid_native = camera_moment_on_model_grid(lsf_wave, sigma, mapping, fallback_sigma)
    sig = np.sqrt(np.interp(wave_aa, model_wave, variance))
    invalid = np.interp(wave_aa, model_wave, invalid_native.astype(float), left=1., right=1.) > 0
    invalid = dilate(invalid, 26)
    invalid[:26] = invalid[-26:] = True
    dl = np.median(np.diff(wave_aa))
    sigma_pix = sig/dl
    offsets = np.arange(10, -11, -1, dtype=float)[:, None]
    gauss = ndtr((offsets + .5)/sigma_pix[None,:]) - ndtr((offsets - .5)/sigma_pix[None,:])
    desi_approx = Resolution(gauss)
    calibrated = calibrated_resolution(just, response)
    product = (calibrated.tocsr() @ desi_approx.tocsr()).todia()
    padded = np.zeros((53, len(wave_aa)), dtype=float)
    for diagonal, offset in zip(product.data, product.offsets):
        if abs(int(offset)) > 26:
            raise ValueError("combined resolution exceeds 53-diagonal support")
        padded[26-int(offset)] = diagonal
    full = Resolution(padded)
    if np.any(~np.isfinite(full.data)):
        raise ValueError("non-finite combined resolution")
    # Composed operator must equal the ordered chain before float32 storage.
    test = 1. + .2*np.sin(wave_aa/73.)
    a = full.dot(test)
    b = calibrated.dot(desi_approx.dot(test))
    valid = ~invalid
    if valid.any() and np.max(np.abs(a[valid]-b[valid])) > 2e-12:
        raise ValueError("resolution composition mismatch")
    # Keep 33 central diagonals only after explicitly bounding the discarded
    # weight, both by source column and output row. This is substantially
    # smaller on disk and computationally, with a verified sub-1e-6 tail.
    combined = Resolution(padded[10:43])
    if valid.any():
        column_full = np.sum(np.abs(padded),axis=0)
        column_cut = np.sum(np.abs(padded[10:43]),axis=0)
        row_full = full.dot(np.ones(len(wave_aa)))
        row_cut = combined.dot(np.ones(len(wave_aa)))
        col_loss = np.max((column_full[valid]-column_cut[valid])/np.maximum(column_full[valid],1e-30))
        row_loss = np.max(np.abs(row_full[valid]-row_cut[valid])/np.maximum(np.abs(row_full[valid]),1e-30))
        if max(col_loss,row_loss) > 1e-6:
            raise ValueError(f"combined-resolution truncation exceeds tolerance: {col_loss}, {row_loss}")
    return combined.data.astype(np.float32), invalid


def make_source_arm(row, flux, source_wave, band, arm, inst, calc, lsf_wave, lsf_sigma, mapping,
                    fallback_sigma=0.7):
    wave_nm = inst[f"{band}_wave_nm"]
    centers = 10*wave_nm
    nodes = inst[f"{band}_geo_wave_nodes_nm"]
    geo = np.array([engine.gsGeometricThroughput(calc._spectro, calc._make_obs(),
                    float(w), float(row.SHAPE_R), .03, 0., 0) for w in nodes])
    if np.any(~np.isfinite(geo)) or np.any(geo <= 0):
        raise ValueError(f"invalid geometric throughput TARGETID={row.TARGETID}")
    ratio = np.interp(wave_nm, nodes, geo)/np.interp(wave_nm, nodes, inst[f"{band}_geo0_nodes"])
    dust = 10**(-.4*inst[f"{band}_kgal_per_ebv"]*row.EBV)
    response = inst[f"{band}_flat_response_750_e"] * ratio * dust
    if np.any(~np.isfinite(response)) or np.any(response <= 0):
        raise ValueError(f"invalid source response TARGETID={row.TARGETID}")
    just = Resolution(inst[f"{band}_lsf_just"])
    negmask = source_negative_mask(source_wave, flux, centers)
    # Source-model samples <0 are not physically usable; only their affected
    # neighborhoods are masked. Clipping here cannot influence unmasked pixels.
    bins = pixel_average(source_wave, np.maximum(flux, 0), inst[f"{band}_edges_aa"])
    source = just.dot(response*bins)
    resdata, resmask = resolution_and_mask(just, response, lsf_wave, lsf_sigma, mapping, centers, fallback_sigma)
    if np.any(~np.isfinite(source)) or np.any(source < -1e-9):
        raise ValueError(f"invalid source photons TARGETID={row.TARGETID}")
    return response, np.maximum(source, 0), resdata, negmask | resmask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["calibration", "test"], required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--times", type=int, nargs="+", choices=list(TIMES), required=True)
    p.add_argument("--resolution-file", type=Path, default=DATA / "desi_resolution_nodes.npz")
    p.add_argument("--petal", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.count <= 0 or args.start < 0 or len(set(args.times)) != len(args.times):
        p.error("count must be positive, start nonnegative and times unique")
    config = json.loads((DATA / "frozen_design.json").read_text())
    if digest(DATA / "frozen_sample.csv") != config["sample_sha256"] or \
       digest(DATA / "normalized_sources.fits") != config["normalized_sources_sha256"]:
        raise ValueError("frozen inputs have changed")
    meta = json.loads((OUT / "instrument_validation.json").read_text())
    if meta["status"] != "passed" or digest(OUT / "instrument_reference_4200.npz") != meta["instrument_sha256"]:
        raise ValueError("instrument reference changed or did not pass validation")
    cat = pd.read_csv(DATA / "frozen_sample.csv")
    selected = cat[cat.split == args.split].copy()
    if args.petal is not None:
        selected = selected[(selected.FIBER.astype(int)//500) == args.petal]
    selected = selected.iloc[args.start:args.start+args.count].reset_index(drop=True)
    if not len(selected):
        raise ValueError("empty batch")
    with np.load(args.resolution_file, allow_pickle=False) as z:
        lsf_ids, lsf_wave, lsf_sigma = z["targetid"], z["wave"], z["sigma"]
    fallback_sigma = np.nanmedian(lsf_sigma, axis=(0,2))
    if np.any(~np.isfinite(fallback_sigma)):
        raise ValueError("no median reference LSF available for a masked-arm placeholder")
    lsf_index = {int(t): i for i,t in enumerate(lsf_ids)}
    if not all(int(t) in lsf_index for t in selected.TARGETID):
        raise ValueError("missing independent DESI resolution for selected target")
    with fits.open(DATA / "normalized_sources.fits", memmap=True, checksum=True) as h:
        source_wave = np.array(h["WAVELENGTH"].data, dtype=float)
        ids = np.asarray(h["TARGETID"].data)
        ii = selected.source_row.to_numpy(int)
        assert np.array_equal(ids[ii], selected.TARGETID.to_numpy())
        sources = np.array(h["FLUX_1E17"].data[ii], dtype=float)
    camera_meta = json.loads((OUT / "desi_camera_mapping.json").read_text())
    mapping_path = OUT / "desi_camera_mapping.npz"
    if (camera_meta["status"] != "passed" or camera_meta["mapping_sha256"] != digest(mapping_path) or
        camera_meta["normalized_sources_sha256"] != config["normalized_sources_sha256"]):
        raise ValueError("FastSpecFit camera mapping changed or was not validated")
    with np.load(mapping_path,allow_pickle=False) as z:
        mapping = {key:z[key] for key in z.files}
    if not np.array_equal(mapping["model_wave"], source_wave):
        raise ValueError("camera mapping wavelength does not match source models")
    with np.load(OUT / "instrument_reference_4200.npz", allow_pickle=False) as z:
        inst = {k:z[k] for k in z.files}
    calc = LenghuPilotETC("dark", calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, ebv=0., r_eff=0., decenter=.03)
    calc._set_mode(); calc._configure_spectro(1)
    cases = [(total, sky) for total in args.times for sky in SKY]
    nrows = len(selected)*len(cases)
    offsets = [(row, total, sky) for row in selected.itertuples() for total,sky in cases]
    target = np.array([900000000 + int(row.source_row)*10 + list(TIMES).index(total)*2 + SKY.index(sky)
                       for row,total,sky in offsets], dtype=np.int64)
    assert len(np.unique(target)) == nrows
    fibermap = Table({"TARGETID": target, "TARGET_RA": [r.TARGET_RA for r,t,s in offsets],
        "TARGET_DEC": [r.TARGET_DEC for r,t,s in offsets], "RA": [r.TARGET_RA for r,t,s in offsets],
        "DEC": [r.TARGET_DEC for r,t,s in offsets], "OBJTYPE": ["TGT"]*nrows,
        "DESI_TARGET": np.ones(nrows, dtype=np.int64), "COADD_FIBERSTATUS": np.zeros(nrows,dtype=np.int32),
        "TILEID": np.zeros(nrows,dtype=np.int32), "PETAL_LOC": np.zeros(nrows,dtype=np.int16),
        "FIBER": np.arange(nrows,dtype=np.int32)%500, "SPECTROGRAPH": np.zeros(nrows,dtype=np.int16)})
    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "JUST ETC 4200m conditional galaxy scan"
    primary.header["ATMREF"] = "QIU+ESO-RAY"
    primary.header["SKYREF"] = "ESO-4200-EXTRAP"
    primary.header["NMODEL"] = len(selected)
    primary.header["SPLIT"] = args.split
    primary.header["NOTMEAS"] = True
    hdus = [primary, fits.BinTableHDU(fibermap, name="FIBERMAP")]
    scores = {"TARGETID": target}
    mask_stats = []
    for arm, band in enumerate("BRZ"):
        n = len(inst[f"{band}_wave_nm"])
        all_flux = np.zeros((nrows,n),dtype=np.float32)
        all_ivar = np.zeros((nrows,n),dtype=np.float32)
        all_mask = np.ones((nrows,n),dtype=np.uint32)
        all_res = np.zeros((nrows,33,n),dtype=np.float32)
        arm_snr = np.zeros(nrows, dtype=np.float32)
        for i,row in enumerate(selected.itertuples()):
            il = lsf_index[int(row.TARGETID)]
            d, source, rdata, inputmask = make_source_arm(row, sources[i], source_wave,
                band, arm, inst, calc, lsf_wave, lsf_sigma[il], mapping, fallback_sigma[arm])
            for j,(total,sky) in enumerate(cases):
                k = i*len(cases)+j
                n_exp = len(TIMES[total])
                factor = total/TREF
                variance = factor*source + background_variance(inst,band,sky,total,n_exp)
                mask = inputmask | inst[f"{sky}_{band}_mask"] | (d < 1e-6*np.max(d))
                valid = ~mask & np.isfinite(variance) & (variance > 0)
                all_res[k] = rdata
                if np.any(valid):
                    rng = np.random.default_rng(seed_for(int(row.TARGETID),sky,total,band))
                    scale = factor*d
                    all_flux[k,valid] = ((factor*source[valid] + rng.normal(size=valid.sum())*
                                         np.sqrt(variance[valid]))/scale[valid]).astype(np.float32)
                    all_ivar[k,valid] = (scale[valid]**2/variance[valid]).astype(np.float32)
                    all_mask[k,valid] = 0
                    arm_snr[k] = np.median(all_flux[k,valid]*np.sqrt(all_ivar[k,valid]))
                mask_stats.append({"synthetic_targetid": int(target[k]), "reference_targetid": int(row.TARGETID),
                                   "arm": band, "valid_pixels": int(valid.sum()),
                                   "source_or_resolution_masked": int(inputmask.sum())})
        scores[f"SNR_{band}"] = arm_snr
        scores[f"INTEG_RAW_FLUX_{band}"] = np.sum(all_flux,axis=1)/np.maximum(np.sum(all_mask==0,axis=1),1)
        scores[f"MEDIAN_RAW_FLUX_{band}"] = np.median(all_flux,axis=1)
        wh = fits.ImageHDU(10*inst[f"{band}_wave_nm"],name=f"{band}_WAVELENGTH"); wh.header["BUNIT"]="Angstrom"
        fh = fits.ImageHDU(all_flux,name=f"{band}_FLUX"); fh.header["BUNIT"]="10**-17 erg/(s cm2 Angstrom)"
        ih = fits.ImageHDU(all_ivar,name=f"{band}_IVAR"); ih.header["BUNIT"]="10**34 (s2 cm4 Angstrom2)/erg2"
        hdus.extend([wh,fh,ih,fits.ImageHDU(all_mask,name=f"{band}_MASK"),
                     fits.ImageHDU(all_res,name=f"{band}_RESOLUTION")])
        print(f"prepared arm {band}: {len(selected)} galaxies, {nrows} spectra", flush=True)
    hdus.append(fits.BinTableHDU(Table(scores),name="SCORES"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(args.output, overwrite=True, checksum=True)
    truth = pd.DataFrame({"SYNTHETIC_TARGETID": target, "TARGETID": [int(r.TARGETID) for r,t,s in offsets],
        "source_row": [int(r.source_row) for r,t,s in offsets], "split": args.split,
        "z_true_vi": [r.VI_Z for r,t,s in offsets], "sky": [s for r,t,s in offsets],
        "total_s": [t for r,t,s in offsets], "exposures_s": ["+".join(map(str,TIMES[t])) for r,t,s in offsets]})
    truthfile = args.output.with_name(args.output.stem+"_truth.csv")
    maskfile = args.output.with_name(args.output.stem+"_masks.csv")
    truth.to_csv(truthfile,index=False); pd.DataFrame(mask_stats).to_csv(maskfile,index=False)
    manifest = {"spectra": str(args.output), "spectra_sha256": digest(args.output),
        "truth_sha256": digest(truthfile), "mask_sha256": digest(maskfile),
        "frozen_design_sha256": digest(DATA / "frozen_design.json"),
        "simulation_script_sha256": digest(Path(__file__)),
        "research_adapter_sha256": digest(STUDY / "pilot_4200/lenghu_etc.py"),
        "resolution_sha256": digest(args.resolution_file), "instrument_sha256": meta["instrument_sha256"],
        "camera_mapping_sha256": digest(mapping_path),
        "galaxies": len(selected), "spectra_rows": nrows, "split": args.split,
        "times_s": args.times, "sky_cases": list(SKY), "selection_start": args.start,
        "selection_petal": args.petal,
        "seed_rule": config["noise_seed_version"]}
    args.output.with_name(args.output.stem+"_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest, indent=2),flush=True)


if __name__ == "__main__":
    main()
