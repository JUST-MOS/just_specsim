"""Research-only 4200 m sky + measured Lenghu extinction adapter for JUST ETC.

Sky input is ground-level radiance; it replaces the ETC native sky rather than
being added to it.  In the 400--790 nm label range, the Qiu et al. Table 4
extinction slopes are linearly interpolated and interpreted as an effective
source transmission 10**(-0.4*k*X), with the paper's airmass formula at the
same target zenith angle as SkyCalc. Outside that interval a 3060 m ESO
transmission with Rayleigh scattering adjusted to the paper's assumed site
pressure is an explicitly uncalibrated fallback. No atmospheric
transmission is applied to the sky radiance.  See the companion report for
coarse-bin, narrow-telluric and 4200 m site limitations.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(STUDY / "instrument"))
sys.path.insert(0, str(STUDY / "pilot"))

from just_etc import JUSTExposureTimeCalculator
from just_etc import ETC_py_optimized as engine
from theoretical_lsf import make_resolution
from compare_sky_models import trace_fraction
from refine_sky_sampling import apply_sky

QIU_CSV = HERE / "extinction/qiu_2026_table4_mean_extinction.csv"
ESO_ATMOSPHERE = STUDY / "sky/skycalc_dark_001nm.fits"
SKY_DIR = STUDY / "sky/height_4200/derived"
AIRMASS = 1.2
ZENITH_DEG = math.degrees(math.acos(1.0 / AIRMASS))
ALTITUDE_DEG = 90.0 - ZENITH_DEG
QIU_AIRMASS = 1.0 / (math.cos(math.radians(ZENITH_DEG)) +
                     0.5057 * (6.0799 + ALTITUDE_DEG)**(-1.6364))
LENGHU_PRESSURE_HPA = float((460.0*u.Torr).to_value(u.hPa))
ESO_3060_PRESSURE_HPA = 708.0
DEEP_MOLECULAR_TRANSMISSION_THRESHOLD = 0.8
EDGE_MARGIN_PIXELS = 16
SEAM_MARGIN_NM = 0.8


class QiuObservedAtmosphere:
    """Qiu binned mean extinction with explicit out-of-range ESO fallback.

    The fitted k are labels of 5--10 nm bins; interpolation through labels is
    a pilot approximation and does not recover narrow atmospheric features.
    785 nm is not a measured table entry. The fallback uses fixed 3060 m ESO
    transmission components at airmass 1.2, with Rayleigh optical depth scaled
    by the assumed Lenghu/ESO reference-pressure ratio. Other airmasses are rejected.
    """
    def __init__(self, table_path=QIU_CSV, fallback_path=ESO_ATMOSPHERE):
        table = Table.read(table_path, format="ascii.csv")
        self.wave_nm = np.asarray(table["wavelength_nm"], dtype=float)
        self.k = np.asarray(table["k_mag_per_airmass"], dtype=float)
        self.sigma_k = np.asarray(table["sigma_k_mag_per_airmass"], dtype=float)
        if len(self.wave_nm) != 53 or self.wave_nm[0] != 400 or self.wave_nm[-1] != 790:
            raise ValueError("unexpected Qiu Table 4 range or row count")
        if np.any(np.diff(self.wave_nm) <= 0) or np.any(~np.isfinite(self.k)) or np.any(self.k < 0):
            raise ValueError("invalid Qiu extinction table")
        if np.any(~np.isfinite(self.sigma_k)) or np.any(self.sigma_k < 0):
            raise ValueError("invalid Qiu uncertainty column")
        with fits.open(fallback_path, memmap=False) as hdus:
            data = hdus[1].data
            columns = hdus[1].columns
            if columns["lam"].unit != "nm" or columns["trans"].unit != "1":
                raise ValueError("unexpected ESO atmosphere units")
            self.reference_wave_nm = np.array(data["lam"], dtype=float)
            native_trans = np.array(data["trans"], dtype=float)
            self.reference_molecular = np.array(data["trans_ma"], dtype=float)
            ozone = np.array(data["trans_o3"], dtype=float)
            rayleigh = np.array(data["trans_rs"], dtype=float)
            aerosol = np.array(data["trans_ms"], dtype=float)
            header_text = str(hdus[0].header)
        if '"observatory": "3060"' not in header_text or '"airmass": 1.2' not in header_text:
            raise ValueError("fallback atmosphere must be the documented ESO 3060m X=1.2 model")
        for component in (native_trans, self.reference_molecular, ozone, rayleigh, aerosol):
            if np.any(~np.isfinite(component)) or np.any(component < 0) or np.any(component > 1):
                raise ValueError("invalid ESO transmission component")
        np.testing.assert_allclose(native_trans, self.reference_molecular * ozone * rayleigh * aerosol,
                                   rtol=1e-12, atol=1e-14)
        self.reference_trans = (self.reference_molecular * ozone * aerosol *
                                rayleigh**(LENGHU_PRESSURE_HPA / ESO_3060_PRESSURE_HPA))
        if (np.any(np.diff(self.reference_wave_nm) <= 0) or np.any(~np.isfinite(self.reference_trans)) or
            np.any(self.reference_trans < 0) or np.any(self.reference_trans > 1) or
            np.any(~np.isfinite(self.reference_molecular)) or np.any(self.reference_molecular < 0) or
            np.any(self.reference_molecular > 1)):
            raise ValueError("invalid fallback atmospheric transmission")

    def transmission(self, wavelength_nm, airmass=AIRMASS):
        """Return source transmission and mask for the tabulated label range [nm]."""
        if not math.isclose(airmass, AIRMASS, rel_tol=0, abs_tol=1e-9):
            raise ValueError("ESO fallback is only defined for airmass 1.2")
        wave = np.asarray(wavelength_nm, dtype=float)
        if np.any(~np.isfinite(wave)) or np.any(wave < self.reference_wave_nm[0]) or np.any(wave > self.reference_wave_nm[-1]):
            raise ValueError("wavelength outside fallback atmosphere coverage")
        fallback = np.interp(wave, self.reference_wave_nm, self.reference_trans)
        qiu_domain = (wave >= self.wave_nm[0]) & (wave <= self.wave_nm[-1])
        k = np.interp(wave, self.wave_nm, self.k)
        output = np.where(qiu_domain, 10.0 ** (-0.4 * k * QIU_AIRMASS), fallback)
        if np.any(~np.isfinite(output)) or np.any(output < 0) or np.any(output > 1):
            raise ValueError("invalid source atmospheric transmission")
        return output, qiu_domain

    def telluric_and_seam_mask(self, pixel_edges_nm):
        """Flag deep model tellurics and joins; this is a conservative proxy.

        Deep telluric locations are identified by any 0.01 nm sample of the
        unrelated 3060 m ESO molecular transmission dropping below 0.8.  Such
        pixels are expanded by the theoretical LSF's 16-pixel half-width. The
        proxy is a mask only: it is never multiplied into Qiu's transmission.
        """
        edges = np.asarray(pixel_edges_nm, dtype=float)
        if edges.ndim != 1 or len(edges) < 34 or np.any(np.diff(edges) <= 0):
            raise ValueError("invalid detector edges")
        centers = (edges[:-1] + edges[1:]) / 2
        index = np.searchsorted(edges, self.reference_wave_nm, side="right") - 1
        in_arm = (index >= 0) & (index < len(centers))
        in_measured = (self.reference_wave_nm >= self.wave_nm[0]) & (self.reference_wave_nm <= self.wave_nm[-1])
        deep = in_arm & in_measured & (self.reference_molecular < DEEP_MOLECULAR_TRANSMISSION_THRESHOLD)
        hit = np.bincount(index[deep], minlength=len(centers)) > 0
        expanded = np.convolve(hit, np.ones(2*EDGE_MARGIN_PIXELS+1, dtype=int), mode="same") > 0
        seam = (np.abs(centers-self.wave_nm[0]) < SEAM_MARGIN_NM) | (np.abs(centers-self.wave_nm[-1]) < SEAM_MARGIN_NM)
        edge = np.zeros(len(centers), dtype=bool)
        edge[:EDGE_MARGIN_PIXELS] = edge[-EDGE_MARGIN_PIXELS:] = True
        return {"telluric_proxy": expanded, "seam": seam, "lsf_edge": edge,
                "fallback": (centers < self.wave_nm[0]) | (centers > self.wave_nm[-1])}


class LenghuPilotETC(JUSTExposureTimeCalculator):
    """Use the ordinary ETC source/noise model with one fixed external sky.

    Only the reference geometry and CCD arms are supported.  The external sky
    is convolved with the same theoretical LSF used by the Redrock writer.
    Sky and atmospheric files are assumed under the conditions in their headers,
    and cannot be silently reused at a different geometry.
    """
    def __init__(self, sky_condition, *, calc_mode="accurate"):
        super().__init__(calc_mode=calc_mode)
        if sky_condition not in ("dark", "bright"):
            raise ValueError("sky_condition must be dark or bright")
        self.condition = sky_condition
        self.atmosphere = QiuObservedAtmosphere()
        self.sky_path = SKY_DIR / f"{sky_condition}_4200m_log_sqm_as_vab.fits"
        with fits.open(self.sky_path, memmap=False, checksum=True) as h:
            self.sky_header = h[0].header.copy()
            if h[1].columns["lam"].unit != "nm" or h[1].columns["flux"].unit != "ph/s/m2/micron/arcsec2":
                raise ValueError("unexpected sky wavelength or photon radiance units")
            self.sky_wave = np.array(h[1].data["lam"], dtype=float)
            self.sky_flux = np.array(h[1].data["flux"], dtype=float)
            self.sky_guard = np.array(h[1].data["GUARD_MASK"], dtype=bool)
        if (self.sky_header["ALT_TGT"] != 4200 or self.sky_header["METHOD"] != "log" or
            not self.sky_header["NOTMEAS"] or not self.sky_header["SQMVAB"] or
            self.sky_header["SCENARIO"] != sky_condition or self.sky_header["AIRMASS"] != AIRMASS or
            self.sky_header["MOONFLG"] != (sky_condition == "bright") or
            self.sky_header["GUARDT"] != 0.01):
            raise ValueError("unexpected 4200m sky provenance")
        if sky_condition == "bright" and not (self.sky_header["MOONALT"] == 45 and
                                               self.sky_header["MOONSEP"] == 60 and
                                               self.sky_header["SUNMSEP"] == 90):
            raise ValueError("unexpected bright-sky Moon geometry")
        if np.any(np.diff(self.sky_wave) <= 0) or np.any(~np.isfinite(self.sky_flux)) or np.any(self.sky_flux < 0):
            raise ValueError("invalid ground sky model")
        self.sky_diagnostics = {}
        self.set_obs_conditions(zenith_angle=ZENITH_DEG, field_angle=0.0,
                                lunar_za=135.0 if sky_condition == "dark" else 45.0,
                                lunar_angle=60.0, lunar_phase=0.25)

    def _check_scene(self, obs):
        x = 1.0 / math.cos(math.radians(obs.zenithangle))
        if not math.isclose(x, AIRMASS, rel_tol=0, abs_tol=1e-9):
            raise ValueError("4200m reference sky and fallback atmosphere require airmass 1.2")
        if self._obs_params["field_angle"] != 0.0:
            raise ValueError("pilot sky integration currently supports only the on-axis field")
        if self.condition == "dark" and obs.lunarZA <= 90:
            raise ValueError("dark sky requires Moon below horizon")
        if self.condition == "bright" and not (math.isclose(obs.lunarZA, 45.0) and
                                               math.isclose(obs.lunarangle, 60.0) and
                                               math.isclose(obs.lunarphase, 0.25)):
            raise ValueError("bright sky requires the fixed half-Moon reference geometry")

    def _source_atmospheric_transmission(self, obs, lambda_nm):
        self._check_scene(obs)
        transmission, _ = self.atmosphere.transmission(lambda_nm)
        return float(transmission)

    def _arm_noise(self, obs, t_exp):
        self._check_scene(obs)
        if not np.isfinite(t_exp) or t_exp <= 0:
            raise ValueError("single-exposure time must be positive")
        sp = self._spectro
        output = []
        self.sky_diagnostics = {}
        for arm in range(sp.N_arms):
            if sp.Dtype[arm] != 0:
                raise ValueError("pilot noise only validated for CCD detector arms")
            n = int(sp.npix[arm])
            edges = sp.lmin[arm] + sp.dl[arm] * np.arange(n+1)
            centers = (edges[:-1] + edges[1:]) / 2
            aeff = np.array([engine.gsAeff(sp, obs, arm, w, 0.0) for w in centers])
            trace, trace_error = trace_fraction(sp, obs, arm, centers)
            radius = sp.fiber_ent_rad / sp.EFL[0] * engine.ARCSEC_PER_URAD
            sensitivity = aeff * trace * math.pi * radius**2 * t_exp
            resolution, valid, lsf_meta = make_resolution(sp, obs, arm, 10*centers)
            setup = (edges, centers, sensitivity, resolution, valid, trace_error, lsf_meta)
            sky_e, _, incoming, deposited = apply_sky(self.sky_wave, self.sky_flux, setup)
            if np.any(~np.isfinite(sky_e)) or np.any(sky_e < 0):
                raise ValueError("negative or nonfinite 4200m sky electrons")
            sample_factor = 1.0
            sky_ref = np.maximum.reduce([np.r_[sky_e[0], sky_e[:-1]], sky_e, np.r_[sky_e[1:], sky_e[-1]]])
            sky_stray = float(np.sum(sky_e) * sp.width[arm] * sp.pix[arm] / sp.sep[arm] / n)
            detector = float((sp.dark[arm] * t_exp * sample_factor + sp.read[arm]**2) * sp.width[arm])
            other_variance = (sample_factor*sky_e + (sp.sysfrac*sky_ref)**2 +
                              sp.diffuse_stray*sky_stray*sample_factor + detector)
            if np.any(~np.isfinite(other_variance)) or np.any(other_variance <= 0):
                raise ValueError("invalid modeled background+detector variance")
            quality = self.atmosphere.telluric_and_seam_mask(edges)
            self.sky_diagnostics[arm] = {"wave_nm": centers, "sky_e": sky_e,
                "background_variance": other_variance, "resolution": resolution,
                "valid_lsf": valid, "quality": quality, "trace_error": trace_error,
                "lsf": lsf_meta, "incoming": incoming, "deposited": deposited,
                "sky_stray_e": sky_stray, "detector_variance": detector,
                "sample_factor": sample_factor}
            output.append((other_variance, sample_factor))
        return output
