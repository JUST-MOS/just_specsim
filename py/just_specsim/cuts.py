"""
desitarget.cuts
===============

Target Selection for Legacy Surveys catalogue data derived from `the wiki`_.

A collection of helpful (static) methods to check whether an object's
flux passes a given selection criterion (*e.g.* LRG, ELG or QSO).

.. _`the Gaia data model`: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
.. _`the Legacy Surveys`: http://www.legacysurvey.org/
.. _`the wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection
.. _`the SV3 wiki`: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/SV3
.. _`Legacy Surveys mask`: http://www.legacysurvey.org/dr8/bitmasks/
"""
import warnings
from time import time
import os.path

import numbers
import sys

import fitsio
import numpy as np
import healpy as hp
from importlib import resources
import numpy.lib.recfunctions as rfn
from importlib import import_module

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Row

# from desitarget import io
# from desitarget.internal import sharedmem
# from desitarget.gaiamatch import match_gaia_to_primary, find_gaia_files_hp
# from desitarget.gaiamatch import pop_gaia_coords, pop_gaia_columns, unextinct_gaia_mags
# from desitarget.gaiamatch import gaia_dr_from_ref_cat, is_in_Galaxy, gaia_psflike
# from desitarget.targets import finalize, resolve
# from desitarget.geomask import bundle_bricks, pixarea2nside, sweep_files_touch_hp
# from desitarget.geomask import box_area, hp_in_box, is_in_box, is_in_hp
# from desitarget.geomask import cap_area, hp_in_cap, is_in_cap, imaging_mask

# ADM set up the DESI default logger
from .log import get_logger
log = get_logger()

def _check_BGS_targtype(targtype):
    """Fail if `targtype` is not one 'bright', 'faint' or 'wise'.
    """
    targposs = ['faint', 'bright', 'wise']

    if targtype not in targposs:
        msg = 'targtype must be one of {} not {}'.format(targposs, targtype)
        log.critical(msg)
        raise ValueError(msg)

def isBGS_colors(rfiberflux=None, gflux=None, rflux=None, zflux=None,
                 w1flux=None, w2flux=None, rfibertotflux=None, maskbits=None,
                 south=True, targtype=None, primary=None):
    """Standard color-based cuts used by all BGS target selection classes
    (see, e.g., :func:`~desitarget.cuts.isBGS` for parameters).
    """
    _check_BGS_targtype(targtype)

    if primary is None:
        primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    fmc = np.zeros_like(rflux, dtype='?')

    if south:
        bgs &= rflux > gflux * 10**(-1.0/2.5)
        bgs &= rflux < gflux * 10**(4.0/2.5)
        bgs &= zflux > rflux * 10**(-1.0/2.5)
        bgs &= zflux < rflux * 10**(4.0/2.5)
    else:
        bgs &= rflux > gflux * 10**(-1.0/2.5)
        bgs &= rflux < gflux * 10**(4.0/2.5)
        bgs &= zflux > rflux * 10**(-1.0/2.5)
        bgs &= zflux < rflux * 10**(4.0/2.5)

    g = 22.5 - 2.5*np.log10(gflux.clip(1e-16))
    r = 22.5 - 2.5*np.log10(rflux.clip(1e-16))
    z = 22.5 - 2.5*np.log10(zflux.clip(1e-16))
    w1 = 22.5 - 2.5*np.log10(w1flux.clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(rfiberflux.clip(1e-16))

    # Fibre Magnitude Cut (FMC) -- This is a low surface brightness cut
    # with the aim of increase the redshift success rate.
    fmc |= ((rfib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((rfib < 22.9) & (r < 20.0) & (r > 17.8))
    # ?????????????????????????????????????????????????????
    # Remove next line for the Main Survey?????????????????
    # ?????????????????????????????????????????????????????
    fmc |= ((rfib < 2.9 + r) & (r > 20))

    bgs &= fmc

    # BASS r-mag offset with DECaLS.
    offset = 0.04

    # D. Schlegel - ChangHoon H. color selection to get a high redshift
    # success rate.
    if south:
        schlegel_color = (z - w1) - 3/2.5 * (g - r) + 1.2
        rfibcol = (rfib < 20.75) | ((rfib < 21.5) & (schlegel_color > 0.))
    else:
        schlegel_color = (z - w1) - 3/2.5 * (g - (r-offset)) + 1.2
        rfibcol = (rfib < 20.75+offset) | ((rfib < 21.5+offset) &
                                           (schlegel_color > 0.))

    if targtype == 'bright':
        if south:
            bgs &= rflux > 10**((22.5-19.5)/2.5)
            bgs &= rflux <= 10**((22.5-12.0)/2.5)
            bgs &= rfibertotflux <= 10**((22.5-15.0)/2.5)
        else:
            bgs &= rflux > 10**((22.5-(19.5+offset))/2.5)
            bgs &= rflux <= 10**((22.5-12.0)/2.5)
            bgs &= rfibertotflux <= 10**((22.5-15.0)/2.5)
    elif targtype == 'faint':
        if south:
            bgs &= rflux > 10**((22.5-20.175)/2.5)
            bgs &= rflux <= 10**((22.5-19.5)/2.5)
            bgs &= (rfibcol)
        else:
            bgs &= rflux > 10**((22.5-(20.220))/2.5)
            bgs &= rflux <= 10**((22.5-(19.5+offset))/2.5)
            bgs &= (rfibcol)

    return bgs
