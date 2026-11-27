"""Run original 4200 m forward model under frozen DESI-like nominal conditions.

The adapter changes only the cached response/sky, seeing, Galactic reddening,
time grid and random seed. The original galaxy source, masks, LSF composition,
extraction and electron-noise algebra remain unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
POP = STUDY / "population_4200_camera_overlap"
DATA = STUDY / "leader_195_205_20261116/extended_20p5/input"
SHARED = HERE / "nominal_run/shared"
sys.path[:0] = [str(HERE), str(POP)]
from build_analysis import sha, SEEING_NOMINAL_800
from lenghu_etc import LenghuPilotETC


class NominalGeometryETC(LenghuPilotETC):
    """Adapter is used ONLY for aperture geometry, never its X=1.2 atmosphere."""
    def set_obs_conditions(self, **kwargs):
        if "seeing_fwhm_800" in kwargs:
            kwargs["seeing_fwhm_800"] = SEEING_NOMINAL_800
        return super().set_obs_conditions(**kwargs)


def seed_for(tid, sky, total, arm):
    return int(hashlib.sha256(f"20261121-nominal:{tid}:{total}:{arm}".encode()).hexdigest()[:16],16)


def main():
    if "--output" not in sys.argv:
        raise ValueError("output path required")
    out = Path(sys.argv[sys.argv.index("--output")+1])
    related = [out, out.with_name(out.stem+"_truth.csv"),
               out.with_name(out.stem+"_masks.csv"),out.with_name(out.stem+"_manifest.json")]
    if any(p.exists() for p in related):
        raise FileExistsError("refusing to overwrite any existing or partial batch")
    validation = json.loads((SHARED/"instrument_validation.json").read_text())
    if validation["status"] != "passed" or validation["depth_inputs_sha256"] != sha(HERE/"depth_inputs.npz"):
        raise ValueError("nominal instrument inputs not verified or changed")
    source = POP/"simulate_batch.py"
    spec = importlib.util.spec_from_file_location("original_forward_simulation", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.DATA, original.OUT = DATA, SHARED
    original.LenghuPilotETC = NominalGeometryETC
    original.SKY = ("nominal",)
    # Rounded down from the model mapping of the 1500/3000 s dark anchors;
    # preserve the read-boundary behavior instead of rounding across it.
    original.TIMES = {1817:[750,750,317], 3436:[750,750,750,750,436]}
    original.seed_for = seed_for
    make = original.make_source_arm
    def nominal_source(row,*args,**kwargs):
        # Stored spectra represent extinction-corrected total photometry. The
        # reference view has E(B-V)=0, irrespective of the original line of sight.
        return make(row._replace(EBV=0.),*args,**kwargs)
    original.make_source_arm = nominal_source
    original.main()
    p = related[-1]
    m = json.loads(p.read_text())
    m.update({"wrapper_sha256":sha(Path(__file__)),"nominal_instrument_validation_sha256":sha(SHARED/"instrument_validation.json"),
              "nominal_sky_sha256":sha(HERE/"zenith_4200m_desi_r21p07.fits"),
              "seed_rule":"sha256(20261121-nominal:TARGETID:total_seconds:arm), first 16 hex digits",
              "atmosphere":"Qiu X=1; pressure-adjusted ESO X=1 fallback outside 400--790 nm",
              "target_ebv_applied":0.,"seeing_fwhm_800_arcsec":SEEING_NOMINAL_800,
              "sky_cases":["nominal"],"altitude_m":4200,
              "note":"NominalGeometryETC is only used by geometry; X=1 source transmission and sky are in the cache."})
    p.write_text(json.dumps(m,indent=2)+"\n")


if __name__ == "__main__":
    main()
