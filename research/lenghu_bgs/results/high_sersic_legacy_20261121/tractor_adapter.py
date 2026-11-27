"""Opt-in continuous mixture response for the frozen JUST simulator.

Importing or calling this module neither simulates noise nor runs Redrock.
The profile must be constructed from the appropriate catalog coefficients;
the input spectrum represents total source light.  The calculator's seeing
must match the saved instrument cache.  Legacy detector pixelization is not
part of the physical coupling to the JUST fiber.
"""
from pathlib import Path
import sys

import numpy as np
from desispec.resolution import Resolution

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "population_4200_camera_overlap"))
from simulate_batch import make_source_arm, resolution_and_mask, source_negative_mask, pixel_average
from just_etc import ETC_py_optimized as engine
from just_etc.mixture_fiber_coupling import etc_mixture_fiber_fraction


def make_source_arm_mixture(row, flux, source_wave, band, arm, inst, calc,
                            lsf_wave, lsf_sigma, mapping, profile,
                            fallback_sigma=.7, offset_angle_deg=0.):
    """Return expected response, source counts, resolution and mask for one arm."""
    old_response, _, _, _ = make_source_arm(row, flux, source_wave, band, arm,
                                            inst, calc, lsf_wave, lsf_sigma,
                                            mapping, fallback_sigma)
    wave_nm = inst[f"{band}_wave_nm"]
    nodes = inst[f"{band}_geo_wave_nodes_nm"]
    obs = calc._make_obs()
    old_geo = np.array([engine.gsGeometricThroughput(calc._spectro, obs, float(w),
                        float(row.SHAPE_R), .03, 0., 0) for w in nodes])
    new_geo = etc_mixture_fiber_fraction(profile, calc._spectro, obs, nodes,
                                         decenter_arcsec=.03, field_angle_deg=0.,
                                         offset_angle_deg=offset_angle_deg)
    numerator = np.interp(wave_nm, nodes, new_geo)
    denominator = np.interp(wave_nm, nodes, old_geo)
    if np.any(denominator <= 0) or np.any(numerator <= 0):
        raise ValueError("invalid galaxy fiber coupling")
    response = old_response * numerator / denominator
    just = Resolution(inst[f"{band}_lsf_just"])
    centers = 10 * wave_nm
    negative_mask = source_negative_mask(source_wave, flux, centers)
    bins = pixel_average(source_wave, np.maximum(flux, 0), inst[f"{band}_edges_aa"])
    counts = just.dot(response * bins)
    resdata, resmask = resolution_and_mask(just, response, lsf_wave, lsf_sigma,
                                           mapping, centers, fallback_sigma)
    if np.any(~np.isfinite(counts)) or np.any(counts < -1e-9):
        raise ValueError("invalid Gaussian-mixture source counts")
    return response, np.maximum(counts, 0), resdata, negative_mask | resmask
