"""Pilot-only JUST resolution matrix from the ETC's *theoretical* PSF.

Not measured or calibrated. The matrix maps a model sampled on detector pixel
centers to pixel-integrated output, and follows the DESI/Redrock DIA convention.
Use the same matrix for the source and the fitting template. Input spectra must
not already contain this JUST instrumental LSF; retain intrinsic galaxy velocity
dispersion. Mask edge pixels where the kernel is not fully represented.
"""
import numpy as np
from desispec.resolution import Resolution

from just_etc import ETC_py_optimized as engine


def make_resolution(spectro, observation, arm, wave_aa, *, nsample=5, ndiag=33):
    """Return (Resolution, interior_valid, metadata) for one detector arm.

    Parameters
    ----------
    spectro, observation : ETC SpectroAttrib, ObsAttrib
        Configuration and observing attributes used by gsSpectroDist.
    arm : int
        Zero-based arm index.
    wave_aa : 1-D array
        Increasing pixel centers, Angstroms; must match the native linear
        detector grid defined by spectro.lmin, spectro.dl and spectro.npix.
    nsample : int
        Number of wavelength nodes to evaluate; interpolate kernel coefficients
        linearly between these nodes.
    ndiag : int
        Odd number of sparse matrix diagonals. The 33-pixel model has been
        evaluated for flux conservation; changing this needs revalidation.
    """
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    if ndiag != 33:
        raise ValueError("only the tested 33-diagonal model is supported")
    if nsample < 2:
        raise ValueError("nsample must be at least 2")
    if not 0 <= arm < spectro.N_arms:
        raise ValueError("arm index out of range")
    if wave_aa.ndim != 1 or len(wave_aa) != spectro.npix[arm] or len(wave_aa) <= ndiag:
        raise ValueError("wave_aa must be the full native detector grid")
    expected = 10 * (spectro.lmin[arm] + spectro.dl[arm] * (np.arange(len(wave_aa)) + 0.5))
    if not np.all(np.isfinite(wave_aa)) or not np.allclose(wave_aa, expected, rtol=0, atol=1e-7):
        raise ValueError("wave_aa does not match native detector pixel centers")

    half = ndiag // 2
    nodes_aa = np.linspace(wave_aa[0], wave_aa[-1], nsample)
    profiles = np.empty((nsample, ndiag))
    raw_sum = np.empty(nsample)
    raw_min = np.empty(nsample)
    for index, node in enumerate(nodes_aa):
        profile = np.zeros(ndiag, dtype=np.float64)
        engine.gsSpectroDist(spectro, observation, arm, node / 10., float(half), 0., ndiag, profile)
        raw_sum[index] = profile.sum()
        raw_min[index] = profile.min()
        # Clip only negligible numerical ringing, never a significant negative
        # LSF. Normalize the finite, 33-pixel kernel to unit integrated flux.
        if not np.isfinite(profile).all() or raw_min[index] < -1e-6 or raw_sum[index] <= 0:
            raise ValueError("theoretical LSF has non-finite or significant negative values")
        profile = np.maximum(profile, 0.)
        profiles[index] = profile / profile.sum()

    # scipy DIA: data[k,j] = R[i,j], offsets[k] = j-i. Each column j
    # distributes one input pixel into output bins i around it.
    offsets = np.arange(half, -half - 1, -1)
    data = np.array([
        np.interp(wave_aa, nodes_aa, profiles[:, half - offset])
        for offset in offsets
    ])
    matrix = Resolution(data)
    valid = np.ones(len(wave_aa), dtype=bool)
    valid[:half] = valid[-half:] = False
    return matrix, valid, {
        "kind": "ETC model, not measured LSF",
        "wave_nodes_aa": nodes_aa.tolist(),
        "raw_sums": raw_sum.tolist(),
        "raw_minima": raw_min.tolist(),
        "ndiag": ndiag,
        "edge_mask_pixels": half,
    }
