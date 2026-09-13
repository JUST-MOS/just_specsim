# 13 Sep 2026 11:12:47

import os
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# import sys
# sys.path.append("../py/just_specsim")
# from maker import SpectrumMaker

from just_specsim.maker import SpectrumMaker

if __name__ == "__main__":
    specmaker = SpectrumMaker()
    # wave, flux, meta, objmeta = specmaker(z=0.1, Mr=-21.0, color=0.7, saveto='./mockspectra/')
    wave, flux, meta, objmeta = specmaker(z=0.1, Mr=-21.0, color=0.7, saveto=None)

    print('wave', np.shape(wave), wave.min(), wave.max())
    print('flux', np.shape(flux), flux.min(), flux.max())
    print(meta)
    print(objmeta)
    # print('saved', sorted(os.listdir('./mockspectra/')))

    plt.figure(figsize=(10, 4))
    plt.plot(wave, flux)
    plt.xlabel('wavelength [A]')
    plt.ylabel('flux [1e-17 erg/s/cm2/A]')
    plt.title(r'$z=0.1$, $M_r=-21$, $g-r=0.7$')
    plt.tight_layout()
    plt.show()
