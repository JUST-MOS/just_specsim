"""Spectrum helper: match (Mr, g-r), then redshift the nearest template."""

import os
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

# local
from .specio import empty_metatable, read_basis_templates
from .templates import BGS

def save_spectra(outdir, wave, flux, meta, objmeta):
    """Write spectra + metadata into ``outdir``.

    Files:
      spectra.fits  WAVELENGTH + FLUX HDUs
      meta.fits
      objmeta.fits
    """
    os.makedirs(outdir, exist_ok=True)
    flux = np.atleast_2d(np.asarray(flux))
    wave = np.asarray(wave)

    meta.write(os.path.join(outdir, 'meta.fits'), overwrite=True)
    objmeta.write(os.path.join(outdir, 'objmeta.fits'), overwrite=True)

    infile = os.path.join(outdir, 'spectra.fits')
    hdr = fits.Header()
    hdr['EXTNAME'] = 'WAVELENGTH'
    hdr['BUNIT'] = 'Angstrom'
    fits.writeto(infile, wave, header=hdr, overwrite=True)
    hdr = fits.Header()
    hdr['EXTNAME'] = 'FLUX'
    hdr['BUNIT'] = '10^-17 erg/(s*cm^2*Angstrom)'
    fits.append(infile, flux, header=hdr)
    return os.path.abspath(outdir)

class SpectrumMaker:
    """Nearest-neighbor template in (^{0.1}M_r, ^{0.1}(g-r))."""

    def __init__(self):

        meta = read_basis_templates(objtype='BGS', onlymeta=True)
        mabs = np.asarray(meta['SDSS_UGRIZ_ABSMAG_Z01'])
        self.templateid = np.asarray(meta['TEMPLATEID'])
        self.Mr = mabs[:, 2]
        self.gr = mabs[:, 1] - mabs[:, 2]
        self.decam_r = np.asarray(meta['DECAM_R'])
        self.tree = cKDTree(np.column_stack([self.Mr, self.gr]))
        self.maker = None

    def _bgs(self):
        if self.maker is None:
            self.maker = BGS()
        return self.maker

    def __call__(self, z, Mr, color, seed=1, nocolorcuts=True, saveto=None):

        z = np.atleast_1d(np.asarray(z, dtype=float))
        Mr = np.atleast_1d(np.asarray(Mr, dtype=float))
        color = np.atleast_1d(np.asarray(color, dtype=float))
        n = len(z)
        if not (len(Mr) == len(color) == n):
            raise ValueError('z, Mr, color must have the same length')

        _, nn = self.tree.query(np.column_stack([Mr, color]))

        input_meta = empty_metatable(nmodel=n, objtype='BGS', input_meta=True)
        input_meta['TEMPLATEID'] = self.templateid[nn].astype(np.int16)
        input_meta['REDSHIFT'] = z
        input_meta['MAG'] = self.decam_r[nn]
        input_meta['MAGFILTER'] = 'decam2014-r'
        input_meta['SEED'] = np.arange(n, dtype=np.int64) + int(seed)

        flux, wave, meta, objmeta = self._bgs().make_templates(
            input_meta=input_meta, nocolorcuts=nocolorcuts,
        )
        if n == 1:
            flux_out, meta_out, objmeta_out = flux[0], meta[0], objmeta[0]
        else:
            flux_out, meta_out, objmeta_out = flux, meta, objmeta

        if saveto is not None:
            os.makedirs(saveto, exist_ok=True)
            save_spectra(saveto, wave, flux, meta, objmeta)

        return wave, flux_out, meta_out, objmeta_out

