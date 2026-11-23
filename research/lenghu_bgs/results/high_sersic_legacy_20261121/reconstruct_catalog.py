"""Reconstruct matched DR9 fiber predictions from profile and brick WCS."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from desiutil.brick import Bricks

from compare_tractor import TractorDR94Profiles, continuous_fiber_fraction, sha
from pixel_diagnosis import pixel_fiber_fraction

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
SAMPLE = STUDY / "leader_195_205_20261116/extended_20p5/input/frozen_sample.csv"
PREVIOUS = STUDY / "sersic_fiber_model_20261121/fiber_coupling_per_target.csv"


def main():
    sample = pd.read_csv(SAMPLE)
    old = pd.read_csv(PREVIOUS)[["TARGETID", "legacy_new_frac"]]
    d = sample.merge(old, on="TARGETID", validate="one_to_one")
    if len(d) != 1681 or not d.TARGETID.is_unique:
        raise ValueError("unexpected frozen sample")
    ra0, dec0 = Bricks(bricksize=.25).brick_radec(d.TARGET_RA.to_numpy(), d.TARGET_DEC.to_numpy())
    phases = []
    for ra_c, dec_c, ra, dec in zip(ra0, dec0, d.TARGET_RA, d.TARGET_DEC):
        w = WCS({"NAXIS":2,"CTYPE1":"RA---TAN","CTYPE2":"DEC--TAN",
                 "CRPIX1":1800.5,"CRPIX2":1800.5,"CRVAL1":float(ra_c),"CRVAL2":float(dec_c),
                 "CD1_1":-.262/3600.,"CD1_2":0.,"CD2_1":0.,"CD2_2":.262/3600.})
        phases.append(np.asarray(w.all_world2pix([[ra,dec]],0)[0], dtype=np.float32))
    phases = np.asarray(phases)
    model=TractorDR94Profiles()
    records=[]
    for i, row in enumerate(d.itertuples(index=False)):
        amp,var=model.profile(row.morphology,row.SERSIC)
        e=np.hypot(row.SHAPE_E1,row.SHAPE_E2); q=(1-e)/(1+e)
        source = SimpleNamespace(shape_r=row.SHAPE_R, shape_e1=row.SHAPE_E1,
                                 shape_e2=row.SHAPE_E2, bx=float(phases[i,0]), by=float(phases[i,1]))
        pixel=pixel_fiber_fraction(source,amp,var)
        continuous=continuous_fiber_fraction(amp,var,row.SHAPE_R,q)
        cat=row.FIBERFLUX_R/row.FLUX_R
        records.append(dict(TARGETID=int(row.TARGETID), split=row.split, r_dered=row.r_dered,
            morphology=row.morphology, SERSIC=row.SERSIC, SHAPE_R=row.SHAPE_R,
            axis_ratio=q, bx=source.bx, by=source.by, catalog_fraction=cat,
            analytic_fraction=row.legacy_new_frac, continuous_fraction=continuous,
            pixel_fraction=pixel, analytic_over_catalog=row.legacy_new_frac/cat,
            continuous_over_catalog=continuous/cat, pixel_over_catalog=pixel/cat,
            pixel_over_continuous=pixel/continuous))
    result=pd.DataFrame(records)
    result.to_csv(HERE/"catalog_reconstruction_per_target.csv",index=False)
    test=result[result.split=="test"]
    groups={"test_all":test,"test_bright":test[test.r_dered<19.5],
            "test_high_n":test[test.SERSIC>=5],"calibration":result[result.split=="calibration"]}
    groups.update({"test_"+m:test[test.morphology==m] for m in ("REX","EXP","DEV","SER")})
    rows=[]
    for name,g in groups.items():
        r=dict(group=name,n=len(g))
        for key in ("analytic_over_catalog","continuous_over_catalog","pixel_over_catalog"):
            for label,value in zip(("p16","median","p84"),np.percentile(g[key],[16,50,84])):
                r[key+"_"+label]=float(value)
        r["pixel_within_1pct"]=float(np.mean(abs(g.pixel_over_catalog-1)<.01))
        r["pixel_max_abs_rel"]=float(np.max(abs(g.pixel_over_catalog-1)))
        rows.append(r)
    pd.DataFrame(rows).to_csv(HERE/"catalog_reconstruction_summary.csv",index=False)
    direct=pd.read_csv(HERE/"matched_legacy_bricks.csv")[["TARGETID","bx","by"]]
    joined=result.merge(direct,on="TARGETID",suffixes=("_derived","_catalog"),validate="one_to_one")
    direct_result=pd.read_csv(HERE/"pixel_diagnosis_per_target.csv")[["TARGETID","pixel_frac"]]
    joined=joined.merge(direct_result,on="TARGETID",validate="one_to_one")
    position=np.max(np.abs(joined[["bx_derived","by_derived"]].to_numpy()-joined[["bx_catalog","by_catalog"]].to_numpy()))
    flux_delta=np.max(np.abs(joined.pixel_fraction/joined.pixel_frac-1))
    summary=dict(n_all=len(result),n_test=len(test),groups=rows,brick_direct_n=len(joined),
        reconstructed_wcs_max_absolute_pixel_delta=float(position),
        reconstructed_wcs_max_pixel_fraction_relative_delta=float(flux_delta),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (SAMPLE,PREVIOUS,HERE/"matched_legacy_bricks.csv",
                     HERE/"pixel_diagnosis_per_target.csv",HERE/"compare_tractor.py",HERE/"pixel_diagnosis.py",Path(__file__))},
        redrock_run=False)
    (HERE/"catalog_reconstruction_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print("brick check",len(joined),"max pixel position difference",position,"max fraction relative difference",flux_delta,flush=True)


if __name__ == "__main__":
    main()
