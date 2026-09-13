import warnings
import os
import fitsio
from astropy.io import fits
from astropy.table import Table
import numpy as np
from glob import glob

def read_basis_templates(objtype, subtype='', outwave=None, nspec=None,
                         infile=None, onlymeta=False, verbose=False):
    """Return the basis (continuum) templates for a given object type.  Optionally
    returns a randomly selected subset of nspec spectra sampled at
    wavelengths outwave.

    Args:

        objtype (str): object type to read (e.g., ELG, LRG, QSO, STAR, STD, WD,
          MWS_STAR, BGS).
        subtype (str, optional): template subtype, currently only for white
            dwarfs.  The choices are DA and DB and the default is to read both
            types.
        outwave (numpy.array, optional): array of wavelength at which to sample
            the spectra.
        nspec (int, optional): number of templates to return
        infile (str, optional): full path to input template file to read,
            over-riding the contents of the $DESI_BASIS_TEMPLATES environment
            variable.
        onlymeta (Bool, optional): read just the metadata table and return
        verbose: bool
            Be verbose. (Default: False)

    Returns:
        Tuple of (outflux, outwave, meta) where
        outflux is an Array [ntemplate,npix] of flux values [erg/s/cm2/A];
        outwave is an Array [npix] of wavelengths for FLUX [Angstrom];
        meta is a Meta-data table for each object.  The contents of this
        table varies depending on what OBJTYPE has been read.

    Raises:
        EnvironmentError: If the required $DESI_BASIS_TEMPLATES environment
            variable is not set.
        IOError: If the basis template file is not found.

    """
    ltype = objtype.lower()
    if objtype == 'STD':
        ltype = 'star'
    if objtype == 'MWS_STAR':
        ltype = 'star'

    if infile is None:
        infile = find_basis_template(ltype)

    if onlymeta:
        # log.info('Reading {} metadata.'.format(infile))
        print('Reading {} metadata.'.format(infile))

        if objtype.upper() == 'BAL': # non-standard data model
            meta = Table(fitsio.read(infile, ext=1, upper=True,
                                     columns=('BI_CIV','ERR_BI_CIV', 'NCIV_2000', 'VMIN_CIV_2000',
                                          'VMAX_CIV_2000', 'POSMIN_CIV_2000','FMIN_CIV_2000',
                                          'AI_CIV', 'ERR_AI_CIV','NCIV_450', 'VMIN_CIV_450',
                                          'VMAX_CIV_450', 'POSMIN_CIV_450', 'FMIN_CIV_450')))
        else:
            print(infile)
            meta = Table(fitsio.read(infile, ext=1, upper=True))

        if (objtype.upper() == 'WD') and (subtype != ''):
            keep = np.where(meta['WDTYPE'] == subtype.upper())[0]
            if len(keep) == 0:
                # log.warning('Unrecognized white dwarf subtype {}!'.format(subtype))
                print('Unrecognized white dwarf subtype {}!'.format(subtype))
            else:
                meta = meta[keep]

        return meta

    # log.info('Reading {}'.format(infile))
    print('Reading {}'.format(infile))

    if objtype.upper() == 'QSO':
        with fits.open(infile) as fx:
            format_version = _qso_format_version(infile)
            if format_version == 1:
                flux = fx[0].data * 1E-17
                hdr = fx[0].header
                from desispec.io.util import header2wave
                wave = header2wave(hdr)
                meta = Table(fx[1].data)
            elif format_version == 2:
                flux = fx['SDSS_EIGEN'].data.copy()
                wave = fx['SDSS_EIGEN_WAVE'].data.copy()
                meta = Table([np.arange(flux.shape[0]),], names=['PCAVEC',])
            else:
                raise IOError('Unknown QSO basis template format version {}'.format(format_version))
    elif objtype.upper() == 'BAL':
        flux, hdr = fitsio.read(infile, ext=1, columns='TEMP', header=True)
        w1 = hdr['CRVAL1']
        dw = hdr['CDELT1']
        w2 = w1 + dw*flux.shape[1]
        wave = np.arange(w1, w2, dw)

        meta = Table(fitsio.read(infile, ext=1, upper=True,
                                 columns=('BI_CIV','ERR_BI_CIV', 'NCIV_2000', 'VMIN_CIV_2000',
                                          'VMAX_CIV_2000', 'POSMIN_CIV_2000','FMIN_CIV_2000',
                                          'AI_CIV', 'ERR_AI_CIV','NCIV_450', 'VMIN_CIV_450',
                                          'VMAX_CIV_450', 'POSMIN_CIV_450', 'FMIN_CIV_450')))
    else:
        with fits.open(infile) as fx:
            try:
                flux = fx['FLUX'].data
                meta = Table(fx['METADATA'].data)
                wave = fx['WAVE'].data
            except:
                flux = fx[0].data
                meta = Table(fx[1].data)
                wave = fx[2].data

            if 'COLORS' in fx:
                colors = fx['COLORS'].data
                hdr = fx['COLORS'].header
                for ii, col in enumerate(hdr['COLORS'].split(',')):
                    meta[col.upper()] = colors[:, ii].flatten()

            if 'DESI-COLORS' in fx:
                colors = fx['DESI-COLORS'].data
                for col in colors.names:
                    meta[col.upper()] = colors[col]

        #- Check if we have correct version
        if objtype.upper() in ('ELG', 'LRG'):
            if 'BASS_G' not in meta.keys():
                log.error('missing BASS_G from template metadata')
                log.error('Is your DESI_BASIS_TEMPLATES too old? {}'.format(os.getenv('DESI_BASIS_TEMPLATES')))
                log.error('Please update DESI_BASIS_TEMPLATES to v3.0 or later')
                raise IOError('Incompatible basis templates; please update to v3.0 or later')

        if (objtype.upper() == 'WD') and (subtype != ''):
            if 'WDTYPE' not in meta.colnames:
                raise RuntimeError('Please upgrade to basis_templates >=2.3 to get WDTYPE support')

            keep = np.where(meta['WDTYPE'] == subtype.upper())[0]
            if len(keep) == 0:
                log.warning('Unrecognized white dwarf subtype {}!'.format(subtype))
            else:
                meta = meta[keep]
                flux = flux[keep, :]

    # Optionally choose a random subset of spectra. There must be a fast way to
    # do this using fitsio.
    ntemplates = flux.shape[0]
    if nspec is not None:
        these = np.random.choice(np.arange(ntemplates),nspec)
        flux = flux[these,:]
        meta = meta[these]
        ntemplates = nspec
    else:
        nspec = ntemplates

    # Optionally resample the templates at specific wavelengths.  Use
    # multiprocessing to speed this up.
    if outwave is None:
        outflux = flux # Do I really need to copy these variables!
        outwave = wave
    else:
        args = list()
        for jj in range(nspec):
            args.append((outwave, wave, flux[jj,:]))
        import multiprocessing
        ncpu = multiprocessing.cpu_count() // 2   #- avoid hyperthreading
        #- Force 'fork' (see py/desisim/pixsim.py parallel_project for why)
        with multiprocessing.get_context('fork').Pool(ncpu) as P:
            outflux = P.map(_resample_flux, args)
        outflux = np.array(outflux)

    return outflux, outwave, meta

def find_basis_template(objtype, indir=None):
    """
    Return the most recent template in $DESI_BASIS_TEMPLATE/{objtype}_template*.fits
    """
    if indir is None:
        indir = "DESI_templates/"

    objfile_wild = os.path.join(indir, objtype.lower()+'_templates_*.fits')
    objfiles = glob(objfile_wild)
    objfiles.sort(key=os.path.getmtime)
    if len(objfiles) > 0:
        return objfiles[-1]
    else:
        raise IOError('No {} templates found in {}'.format(objtype, objfile_wild))

def empty_metatable(nmodel=1, objtype='ELG', subtype='', simqso=False, input_meta=False):
    """Initialize template metadata tables depending on the given object type.

    Parameters
    ----------
    nmodel : :class:`int`
        Number of rows in output table.  Defaults to 1.
    objtype : :class:`str`
        Object type.  Defaults to ELG.
    subtype : :class:`str`
        Subtype for the given object type (e.g., LYA is objtype=QSO).
        Defaults to `.`
    simqso : :class:`bool`
        Initialize a templates.SIMQSO-style objmeta table rather than a
        templates.QSO one.  Defaults to False.
    input_meta : :class:`bool`
        Initialize an input_meta table for use with the various
        desisim.templates classes (see its use in, e.g.,
        desitarget.mock.mockmaker) Defaults to False.

    Returns
    -------
    meta : :class:`astropy.table.Table`
        Metadata table which is agnostic about the object type.
    objmeta : :class:`astropy.table.Table`
        Objtype-specific supplemental metadata table (e.g., containing the [OII]
        flux for ELG targets and surface gravity for stars.

    """
    from astropy.table import Table, Column

    targetid = np.arange(nmodel).astype(np.int64)

    # Objtype-agnostic metadata
    meta = Table()
    if input_meta:
        meta.add_column(Column(name='TEMPLATEID', length=nmodel, dtype='i2', data=np.zeros(nmodel)-1))
        meta.add_column(Column(name='SEED', length=nmodel, dtype='int64', data=np.zeros(nmodel)-1))
        meta.add_column(Column(name='REDSHIFT', length=nmodel, dtype='f8', data=np.zeros(nmodel)))
        meta.add_column(Column(name='MAG', length=nmodel, dtype='f4', data=np.zeros(nmodel)-1, unit='mag')) # normalization magnitude
        meta.add_column(Column(name='MAGFILTER', length=nmodel, dtype='U15')) # normalization filter
        return meta
    else:
        meta.add_column(Column(name='TARGETID', data=targetid))
        meta.add_column(Column(name='OBJTYPE', length=nmodel, dtype='U10'))
        meta.add_column(Column(name='SUBTYPE', length=nmodel, dtype='U10'))
        meta.add_column(Column(name='TEMPLATEID', length=nmodel, dtype='i2', data=np.zeros(nmodel)-1))
        meta.add_column(Column(name='SEED', length=nmodel, dtype='int64', data=np.zeros(nmodel)-1))
        meta.add_column(Column(name='REDSHIFT', length=nmodel, dtype='f8', data=np.zeros(nmodel)))
        meta.add_column(Column(name='MAG', length=nmodel, dtype='f4', data=np.zeros(nmodel)-1, unit='mag')) # normalization magnitude
        meta.add_column(Column(name='MAGFILTER', length=nmodel, dtype='U15')) # normalization filter
        meta.add_column(Column(name='FLUX_G', length=nmodel, dtype='f4', unit='nanomaggies'))
        meta.add_column(Column(name='FLUX_R', length=nmodel, dtype='f4', unit='nanomaggies'))
        meta.add_column(Column(name='FLUX_Z', length=nmodel, dtype='f4', unit='nanomaggies'))
        meta.add_column(Column(name='FLUX_W1', length=nmodel, dtype='f4', unit='nanomaggies'))
        meta.add_column(Column(name='FLUX_W2', length=nmodel, dtype='f4', unit='nanomaggies'))

        meta['OBJTYPE'] = objtype.upper()
        meta['SUBTYPE'] = subtype.upper()

    # Objtype-specific metadata
    objmeta = Table()
    if objtype.upper() == 'ELG' or objtype.upper() == 'LRG' or objtype.upper() == 'BGS':
        objmeta.add_column(Column(name='TARGETID', data=targetid))
        objmeta.add_column(Column(name='OIIFLUX', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='erg/(s*cm2)'))
        objmeta.add_column(Column(name='HBETAFLUX', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='erg/(s*cm2)'))
        objmeta.add_column(Column(name='EWOII', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='Angstrom'))
        objmeta.add_column(Column(name='EWHBETA', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='Angstrom'))
        objmeta.add_column(Column(name='D4000', length=nmodel, dtype='f4', data=np.zeros(nmodel)-1))
        objmeta.add_column(Column(name='VDISP', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='km/s'))
        objmeta.add_column(Column(name='OIIDOUBLET', length=nmodel, dtype='f4', data=np.zeros(nmodel)-1))
        objmeta.add_column(Column(name='OIIIHBETA', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='Dex'))
        objmeta.add_column(Column(name='OIIHBETA', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='Dex'))
        objmeta.add_column(Column(name='NIIHBETA', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='Dex'))
        objmeta.add_column(Column(name='SIIHBETA', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='Dex'))
        objmeta.add_column(Column(name='TRANSIENT_MODEL', length=nmodel, dtype='U20'))
        objmeta.add_column(Column(name='TRANSIENT_TYPE', length=nmodel, dtype='U10'))
        objmeta.add_column(Column(name='TRANSIENT_EPOCH', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='day'))
        objmeta.add_column(Column(name='TRANSIENT_RFLUXRATIO', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1))

    elif objtype.upper() == 'QSO':
        objmeta.add_column(Column(name='TARGETID', data=targetid))
        if simqso:
            objmeta.add_column(Column(name='MABS_1450', length=nmodel, dtype='f4',
                                      data=np.zeros(nmodel)-1, unit='mag'))
            objmeta.add_column(Column(name='SLOPES', length=nmodel, dtype='f4',
                                      data=np.zeros( (nmodel, 5) )-1))
            objmeta.add_column(Column(name='EMLINES', length=nmodel, dtype='f4',
                                      data=np.zeros( (nmodel, 73, 3) )-1))
        else:
            objmeta.add_column(Column(name='PCA_COEFF', length=nmodel, dtype='f4',
                                      data=np.zeros( (nmodel, 4) )))
        objmeta.add_column(Column(name='BAL_TEMPLATEID', length=nmodel, dtype='i2', data=np.zeros(nmodel)-1))
        objmeta.add_column(Column(name='DLA', length=nmodel, dtype=bool))
        #objmeta.add_column(Column(name='METALS', length=nmodel, dtype=bool))

    elif objtype.upper() == 'STAR' or objtype.upper() == 'STD' or objtype.upper() == 'MWS_STAR':
        objmeta.add_column(Column(name='TARGETID', data=targetid))
        objmeta.add_column(Column(name='TEFF', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='K'))
        objmeta.add_column(Column(name='LOGG', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='m/(s**2)'))
        objmeta.add_column(Column(name='FEH', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1))

    elif objtype.upper() == 'WD':
        objmeta.add_column(Column(name='TARGETID', data=targetid))
        objmeta.add_column(Column(name='TEFF', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='K'))
        objmeta.add_column(Column(name='LOGG', length=nmodel, dtype='f4',
                                  data=np.zeros(nmodel)-1, unit='m/(s**2)'))

    return meta, objmeta
