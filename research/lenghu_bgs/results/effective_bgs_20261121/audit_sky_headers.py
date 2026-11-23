"""Verify returned SkyCalc headers match the three archived requests."""
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
from build_analysis import SKYFILES,sha,sky_r_ab,H

HERE=Path(__file__).resolve().parent


def main():
    rows=[]
    for path in SKYFILES:
        name=path.name.replace(".fits","_input.json")
        expected=json.loads((HERE/name).read_text())
        header=fits.getheader(path)
        comments=[str(x) for x in header["COMMENT"]]
        start=next(i for i,s in enumerate(comments) if s.strip().startswith("{"))
        reported=json.loads("\n".join(comments[start:]))
        if reported!=expected:
            differences={k:{"requested":expected.get(k),"reported":reported.get(k)}
                         for k in set(expected)|set(reported) if expected.get(k)!=reported.get(k)}
            raise ValueError(f"SkyCalc response parameters disagree: {differences}")
        rows.append({"fits":str(path),"sha256":sha(path),"observatory":reported["observatory"],
                     "airmass":reported["airmass"],"wdelta_nm":reported["wdelta"],
                     "moon_enabled":reported["incl_moon"]})
    wave_nm=np.arange(365.,925.0001,.01)
    errors=[]
    for mag in (19.,21.07,22.):
        # Analytic flat-f_nu AB spectrum converted to SkyCalc photon radiance.
        photon=3631e-23*10**(-.4*mag)*1e8/(H*wave_nm*10.)
        errors.append(abs(sky_r_ab(wave_nm,photon)-mag))
    if max(errors)>1e-8:raise ValueError("AB photon conversion self-check failed")
    out={"status":"passed","scope":"ESO input metadata and AB-unit checks only; not Lenghu site validation",
         "models":rows,"flat_fnu_AB_max_error_mag":max(errors),"script_sha256":sha(Path(__file__))}
    (HERE/"zenith_sky_header_audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
