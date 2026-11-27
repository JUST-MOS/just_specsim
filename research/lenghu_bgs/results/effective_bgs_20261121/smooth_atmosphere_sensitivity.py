"""Compare conditional atmosphere candidates in the existing depth proxy only.

The frozen Redrock inputs and sky radiance are unchanged. At X=1 the candidate
uses the earlier study's optical-depth-linear-with-airmass approximation.
"""
from pathlib import Path
import json

import numpy as np
from astropy.io import fits

from effective_model import ARM_NAMES, DepthModel

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent / "atmosphere_qiu_calibrated_20261115"


def main():
    baseline = DepthModel()
    results = []
    for name in ("atmosphere_center_constant.fits", "atmosphere_center_relax870.fits"):
        with fits.open(CANDIDATES / name, memmap=False, checksum=True) as h:
            if h[0].header["AIRMASS"] != 1.2 or not h[0].header["NOTCAL"]:
                raise ValueError("candidate has unexpected provenance")
            wave = np.array(h["ATMOSPHERE"].data["WAVELENGTH"], dtype=float)
            trans = np.array(h["ATMOSPHERE"].data["TRANSMISSION"], dtype=float)
        alt = DepthModel()
        for arm in ARM_NAMES:
            detector_wave = alt.d[f"{arm}_wave_nm"]
            if detector_wave.min() < wave[0] or detector_wave.max() > wave[-1]:
                raise ValueError("candidate does not cover an arm")
            # Scale the high-resolution optical depth before interpolation.
            nominal_trans = np.interp(detector_wave, wave, trans ** (1 / 1.2))
            scene_trans = np.interp(detector_wave, wave, trans)
            for scene, new, key in (("nominal", nominal_trans, "nominal_atm"),
                                    ("dark", scene_trans, "scenario_atm"),
                                    ("bright", scene_trans, "scenario_atm")):
                old = alt.d[f"{arm}_{key}"]
                valid = alt.d[f"{arm}_valid"].astype(bool)
                if np.any(old[valid] <= 0) or np.any(new < 0) or np.any(new > 1):
                    raise ValueError("invalid atmosphere response")
                ratio = np.divide(new, old, out=np.ones_like(new), where=old > 0)
                alt.d[f"{scene}_{arm}_response_no_dust"] *= ratio
        for scene, t in (("nominal", 1817.), ("nominal", 3436.),
                         ("dark", 1500.), ("dark", 3000.),
                         ("bright", 2250.), ("bright", 3600.)):
            old = baseline.information(scene, t, per_arm=True)
            new = alt.information(scene, t, per_arm=True)
            results.append({"candidate":name, "scene":scene, "seconds":t,
                            "q_total_ratio":float(new.sum()/old.sum()),
                            "q_B_ratio":float(new[0]/old[0]),
                            "q_R_ratio":float(new[1]/old[1]),
                            "q_Z_ratio":float(new[2]/old[2])})
    result = {
        "status":"passed",
        "scope":"Proxy-only atmosphere sensitivity; fixed sky, wavelength mask, source and instrument assumptions; not a new redshift simulation or validated confidence interval.",
        "nominal_airmass_scaling":"T(X=1)=T(X=1.2)**(1/1.2) at high-resolution pixels before interpolation; same approximation as candidate construction",
        "cases":results,
    }
    (HERE / "smooth_atmosphere_sensitivity.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
