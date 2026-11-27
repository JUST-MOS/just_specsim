"""Redrock 0.15.4 with exact convolution reuse for paired simulation cases.

Only the coarse chi2 scan is patched: cases belonging to one source have an
identical resolution matrix and wavelength grid, so the official sparse
template convolution is performed once per redshift and the unchanged
Redrock _zchi2_one is evaluated separately for every case. If matrices/grids
do not match, they are processed individually. Template ranges, loss, OII
penalty, minima/refinement and quality flags remain the official code.
"""
__requires__ = "redrock==0.15.4"
import pkg_resources  # noqa: F401 -- activates version-pinned egg

import numpy as np
import redrock
import redrock.zscan as zs
from redrock.external import desi

if redrock.__version__ != "0.15.4":
    raise RuntimeError(f"expected Redrock 0.15.4, got {redrock.__version__}")

ORIGINAL = zs.calc_zchi2


def same_resolution(a, b):
    if len(a.spectra) != len(b.spectra):
        return False
    for x,y in zip(a.spectra,b.spectra):
        if x.wavehash != y.wavehash or x.Rcsr.shape != y.Rcsr.shape:
            return False
        for key in ("indptr","indices","data"):
            if not np.array_equal(getattr(x.Rcsr,key),getattr(y.Rcsr,key)):
                return False
    return True


def grouped_calc_zchi2(target_ids,target_data,dtemplate,progress=None):
    if dtemplate.comm is not None:
        return ORIGINAL(target_ids,target_data,dtemplate,progress=progress)
    # The IDs are 900000000+10*source_row+case. Avoid changing behavior for
    # unrelated Redrock datasets, including the earlier independent smoke.
    if any(not (900000000 <= int(t) < 900100000) for t in target_ids):
        return ORIGINAL(target_ids,target_data,dtemplate,progress=progress)
    nz=len(dtemplate.local.redshifts)
    ntargets=len(target_ids)
    nbasis=dtemplate.template.nbasis
    chi=np.zeros((ntargets,nz))
    penalty=np.zeros((ntargets,nz))
    coeff=np.zeros((ntargets,nz,nbasis))
    galaxy=dtemplate.template.template_type == "GALAXY"
    if galaxy:
        is_oii=(3724<=dtemplate.template.wave)&(dtemplate.template.wave<=3733)
        oii=dtemplate.template.flux[:,is_oii].T
    j=0
    shared_groups=0
    shared_targets=0
    while j<ntargets:
        end=j+1
        base=int(target_ids[j])//10
        while (end<ntargets and int(target_ids[end])//10==base and
               same_resolution(target_data[j],target_data[end])):
            end+=1
        if end-j > 1:
            shared_groups+=1
            shared_targets+=end-j
        data=[zs.spectral_data(target_data[k].spectra) for k in range(j,end)]
        for i in range(nz):
            template=dtemplate.local.data[i]
            tb=np.vstack([s.Rcsr.dot(template[s.wavehash]) for s in target_data[j].spectra])
            for k,(weights,flux,wflux) in zip(range(j,end),data):
                chi[k,i]=zs._zchi2_one(tb,weights,flux,wflux,coeff[k,i])
                if galaxy:
                    line=np.sum(oii.dot(coeff[k,i]))
                    if line<0:
                        penalty[k,i]=-line
        for _ in range(j,end):
            progress.put(1)
        j=end
    print(f"Cache {dtemplate.template.full_type}: targets={ntargets}, shared_groups={shared_groups}, "
          f"shared_targets={shared_targets}",flush=True)
    return chi,coeff,penalty


zs.calc_zchi2 = grouped_calc_zchi2


if __name__ == "__main__":
    print("Redrock version:",redrock.__version__,"module:",redrock.__file__,
          "coarse-scan optimization: exact per-source resolution-convolution reuse",flush=True)
    desi.rrdesi(comm=None)
