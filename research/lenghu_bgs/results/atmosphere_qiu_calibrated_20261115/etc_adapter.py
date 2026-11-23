"""Opt-in reader for the conditional Qiu-constrained source atmosphere.

Example: ``calc = LenghuPilotETC('dark'); calc.atmosphere = QiuConstrainedAtmosphere()``.
This does not change the 4200 m sky radiance or any existing simulation result.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot_4200"))
from lenghu_etc import AIRMASS, QiuObservedAtmosphere  # noqa: E402


class QiuConstrainedAtmosphere(QiuObservedAtmosphere):
    """ETC-compatible fixed-X transmission; inherited masks remain conservative.

    The returned boolean marks the Qiu label range, not a direct measurement
    at every wavelength. Deep-telluric and seam masks retain the pilot policy.
    """
    def __init__(self, path: Path | str | None = None):
        super().__init__()
        self.candidate_path = Path(path) if path is not None else HERE / "atmosphere_center_constant.fits"
        with fits.open(self.candidate_path, memmap=False, checksum=True) as hdus:
            hdr, data = hdus[0].header, hdus["ATMOSPHERE"].data
            if not hdr.get("NOTCAL", False) or not math.isclose(hdr["AIRMASS"], AIRMASS, abs_tol=1e-9):
                raise ValueError("Unexpected model status or airmass")
            if hdus["ATMOSPHERE"].columns["WAVELENGTH"].unit != "nm":
                raise ValueError("Expected wavelength in nm")
            self.candidate_wave_nm = np.array(data["WAVELENGTH"], dtype=float)
            self.candidate_trans = np.array(data["TRANSMISSION"], dtype=float)
        if (np.any(np.diff(self.candidate_wave_nm) <= 0) or
            np.any(~np.isfinite(self.candidate_trans)) or np.any(self.candidate_trans < 0) or
            np.any(self.candidate_trans > 1)):
            raise ValueError("Invalid model wavelength or transmission")

    def transmission(self, wavelength_nm, airmass=AIRMASS):
        """Return (transmission, within_Qiu_labels) for wavelengths in nm."""
        if not math.isclose(airmass, AIRMASS, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Candidate only applies to the 1.2 airmass reference")
        wave = np.asarray(wavelength_nm, dtype=float)
        if (np.any(~np.isfinite(wave)) or np.any(wave < self.candidate_wave_nm[0]) or
            np.any(wave > self.candidate_wave_nm[-1])):
            raise ValueError("Wavelength outside candidate coverage")
        return np.interp(wave, self.candidate_wave_nm, self.candidate_trans), (
            (wave >= self.wave_nm[0]) & (wave <= self.wave_nm[-1]))
