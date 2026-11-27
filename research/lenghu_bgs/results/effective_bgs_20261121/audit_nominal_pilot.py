"""Independently reconstruct every noise draw and camera of the nominal pilot."""
import json
from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
POP = STUDY/"population_4200_camera_overlap"
DATA = STUDY/"leader_195_205_20261116/extended_20p5/input"
SHARED = HERE/"nominal_run/shared"
PILOT = HERE/"nominal_run/v2/test_0000_spectra.fits"
sys.path[:0] = [str(HERE),str(ROOT/"py"),str(POP),str(STUDY/"pilot_4200")]
from build_analysis import sha, SEEING_NOMINAL_800
from lenghu_etc import LenghuPilotETC
from simulate_batch import make_source_arm, background_variance
from simulate_nominal_batch import seed_for


def main():
    truthpath = PILOT.with_name(PILOT.stem+"_truth.csv")
    maskpath = PILOT.with_name(PILOT.stem+"_masks.csv")
    manifestpath = PILOT.with_name(PILOT.stem+"_manifest.json")
    man = json.loads(manifestpath.read_text())
    for path,key in ((PILOT,"spectra_sha256"),(truthpath,"truth_sha256"),(maskpath,"mask_sha256"),
                     (HERE/"simulate_nominal_batch.py","wrapper_sha256"),
                     (SHARED/"instrument_reference_4200.npz","instrument_sha256"),
                     (DATA/"combined_resolution_nodes.npz","resolution_sha256")):
        if sha(path) != man[key]:
            raise ValueError(f"pilot input hash mismatch {path}")
    truth = pd.read_csv(truthpath)
    cat = pd.read_csv(DATA/"frozen_sample.csv")
    selected = cat[cat.split=="test"].iloc[:8].copy().reset_index(drop=True)
    if len(truth)!=16 or set(truth.TARGETID)!=set(selected.TARGETID) or not truth.SYNTHETIC_TARGETID.is_unique:
        raise ValueError("pilot target denominator differs")
    if set(truth.sky)!={"nominal"} or set(truth.total_s)!={1817,3436}:
        raise ValueError("pilot scenarios differ")
    with fits.open(DATA/"normalized_sources.fits",memmap=True,checksum=True) as h:
        wave = np.asarray(h["WAVELENGTH"].data,dtype=float)
        ii = selected.source_row.to_numpy(int)
        np.testing.assert_array_equal(h["TARGETID"].data[ii],selected.TARGETID.to_numpy())
        spectra = np.asarray(h["FLUX_1E17"].data[ii],dtype=float)
    with np.load(DATA/"combined_resolution_nodes.npz",allow_pickle=False) as z:
        ids,lsfw,lsfs=z["targetid"],z["wave"],z["sigma"]
    lookup={int(x):i for i,x in enumerate(ids)}
    fallback=np.nanmedian(lsfs,axis=(0,2))
    with np.load(SHARED/"instrument_reference_4200.npz",allow_pickle=False) as z:
        inst={k:z[k] for k in z.files}
    with np.load(SHARED/"desi_camera_mapping.npz",allow_pickle=False) as z:
        mapping={k:z[k] for k in z.files}
    calc=LenghuPilotETC("dark",calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=SEEING_NOMINAL_800,ebv=0.,r_eff=0.,decenter=.03)
    calc._set_mode();calc._configure_spectro(1)
    checked = 0
    max_flux=0.;max_ivar=0.
    with fits.open(PILOT,memmap=False,checksum=True) as h:
        np.testing.assert_array_equal(h["FIBERMAP"].data["TARGETID"],truth.SYNTHETIC_TARGETID.to_numpy())
        for arm,band in enumerate("BRZ"):
            for i,row in enumerate(selected.itertuples(index=False)):
                il=lookup[int(row.TARGETID)]
                response,source,rdata,inputmask=make_source_arm(row._replace(EBV=0.),spectra[i],wave,band,arm,
                      inst,calc,lsfw,lsfs[il],mapping,fallback[arm])
                for total in (1817,3436):
                    index=np.flatnonzero((truth.TARGETID==row.TARGETID)&(truth.total_s==total))
                    if len(index)!=1:
                        raise ValueError("expected exactly one scenario")
                    k=int(index[0]);factor=total/750.;nread=3 if total==1817 else 5
                    variance=factor*source+background_variance(inst,band,"nominal",total,nread)
                    mask=inputmask|inst[f"nominal_{band}_mask"]|(response<1e-6*response.max())
                    valid=~mask&np.isfinite(variance)&(variance>0)
                    rng=np.random.default_rng(seed_for(int(row.TARGETID),"nominal",total,band))
                    expflux=np.zeros(len(response),dtype=np.float32); expivar=np.zeros_like(expflux)
                    scale=factor*response
                    expflux[valid]=((factor*source[valid]+rng.normal(size=valid.sum())*
                                     np.sqrt(variance[valid]))/scale[valid]).astype(np.float32)
                    expivar[valid]=(scale[valid]**2/variance[valid]).astype(np.float32)
                    gotflux=h[f"{band}_FLUX"].data[k];gotivar=h[f"{band}_IVAR"].data[k]
                    np.testing.assert_array_equal(h[f"{band}_MASK"].data[k]==0,valid)
                    np.testing.assert_array_equal(h[f"{band}_RESOLUTION"].data[k],rdata)
                    np.testing.assert_allclose(gotflux,expflux,rtol=2e-7,atol=1e-7)
                    np.testing.assert_allclose(gotivar,expivar,rtol=2e-7,atol=1e-7)
                    max_flux=max(max_flux,float(np.max(np.abs(gotflux-expflux))))
                    max_ivar=max(max_ivar,float(np.max(np.abs(gotivar-expivar))))
                    checked+=1
    out={"status":"passed","scope":"every pilot source, nominal time and arm; numerical reconstruction only",
         "galaxies":len(selected),"spectra":len(truth),"arm_cases":checked,
         "max_abs_flux_difference_in_1e-17_units":max_flux,"max_abs_ivar_difference":max_ivar,
         "target_ebv_in_simulation":0.,"seeing_800_arcsec":SEEING_NOMINAL_800,
         "spectra_sha256":sha(PILOT),"simulator_sha256":sha(HERE/"simulate_nominal_batch.py")}
    (HERE/"nominal_run/v2/pilot_input_audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
