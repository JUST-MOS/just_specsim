"""Independently approximate DR9 pixel-center rendering and aperture sums."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

from compare_tractor import TractorDR94Profiles, continuous_fiber_fraction, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[2]
PIX_SCALE = .262  # arcsec; DR9.6.2 runbrick default
FIBER_RADIUS_PIX = .75 / PIX_SCALE


def circle_pixel_area(x0, x1, y0, y1, radius):
    """Exact circle/square overlap in pixel units via piecewise primitives."""
    left, right = max(x0, -radius), min(x1, radius)
    if left >= right:
        return 0.
    cuts = [left, right]
    for y in (y0, y1):
        if abs(y) < radius:
            x = np.sqrt(radius**2-y**2)
            for edge in (-x, x):
                if left < edge < right:
                    cuts.append(float(edge))
    cuts = sorted(set(cuts))
    def primitive(x):
        x = np.clip(x, -radius, radius)
        return .5*(x*np.sqrt(max(radius**2-x*x, 0.)) + radius**2*np.arcsin(x/radius))
    result = 0.
    for a, b in zip(cuts[:-1], cuts[1:]):
        x = .5*(a+b)
        ycirc = np.sqrt(max(radius**2-x*x, 0.))
        top, bottom = min(y1, ycirc), max(y0, -ycirc)
        if top <= bottom:
            continue
        curve = primitive(b)-primitive(a)
        top_integral = y1*(b-a) if y1 <= ycirc else curve
        bottom_integral = y0*(b-a) if y0 >= -ycirc else -curve
        result += top_integral-bottom_integral
    return result


def aperture_samples(bx, by):
    """Pixel-center coordinates and exact aperture overlap fractions."""
    x_phase = bx - np.round(bx)
    y_phase = by - np.round(by)
    extent = int(np.ceil(FIBER_RADIUS_PIX+1.))
    grid = np.arange(-extent, extent+1)
    xx, yy = np.meshgrid(grid-x_phase, grid-y_phase)
    area = np.array([circle_pixel_area(x-.5, x+.5, y-.5, y+.5, FIBER_RADIUS_PIX)
                     for x,y in zip(xx.ravel(), yy.ravel())])
    if not np.isclose(area.sum(), np.pi*FIBER_RADIUS_PIX**2, rtol=1e-12):
        raise ValueError("aperture overlap does not conserve circle area")
    return xx.ravel()*PIX_SCALE, yy.ravel()*PIX_SCALE, area


def pixel_fiber_fraction(row, amps, variances, seeing_sigma=1./2.35):
    """Gaussian mixture sampled at pixel centers then overlap-weighted."""
    e = np.hypot(row.shape_e1, row.shape_e2)
    q = (1-e)/(1+e)
    theta = .5*np.arctan2(row.shape_e2, row.shape_e1)
    ct, st = np.cos(theta), np.sin(theta)
    x, y, aperture = aperture_samples(row.bx, row.by)
    v = np.asarray(variances) * row.shape_r**2
    cxx = seeing_sigma**2 + v*(q*q*ct*ct+st*st)
    cyy = seeing_sigma**2 + v*(q*q*st*st+ct*ct)
    cxy = v*ct*st*(q*q-1.)
    det = cxx*cyy-cxy*cxy
    if np.any(det <= 0):
        raise ValueError("invalid convolved covariance")
    exponent = (cyy[:,None]*x[None,:]**2 - 2*cxy[:,None]*x[None,:]*y[None,:] +
                cxx[:,None]*y[None,:]**2)/det[:,None]
    model = np.sum(np.asarray(amps)[:,None]*np.exp(-.5*exponent) /
                   (2*np.pi*np.sqrt(det))[:,None], axis=0) * PIX_SCALE**2
    return float(np.sum(model*aperture))


def main():
    data = pd.read_csv(HERE/"matched_legacy_bricks.csv")
    model = TractorDR94Profiles()
    records = []
    for row in data.itertuples(index=False):
        amp, var = model.profile(row.morphology, row.sersic)
        e = np.hypot(row.shape_e1, row.shape_e2)
        q = (1-e)/(1+e)
        continuous = continuous_fiber_fraction(amp, var, row.shape_r, q)
        pixel = pixel_fiber_fraction(row, amp, var)
        catalog = row.fiberflux_r / row.flux_r
        records.append(dict(TARGETID=row.TARGETID, split=row.split, morphology=row.morphology,
                            SERSIC=row.sersic, bx=row.bx, by=row.by, cat_frac=catalog,
                            continuous_frac=continuous, pixel_frac=pixel,
                            continuous_over_cat=continuous/catalog, pixel_over_cat=pixel/catalog,
                            pixel_over_continuous=pixel/continuous))
    result = pd.DataFrame(records)
    result.to_csv(HERE/"pixel_diagnosis_per_target.csv", index=False)
    groups = {"all_matched":result, "high_n":result[result.SERSIC>=5],
              "test_only":result[result.split=="test"]}
    out = []
    for name, group in groups.items():
        row = dict(group=name, n=len(group))
        for col in ("continuous_over_cat","pixel_over_cat","pixel_over_continuous"):
            row[col+"_p16"], row[col+"_median"], row[col+"_p84"] = [float(x) for x in np.percentile(group[col],[16,50,84])]
        row["pixel_within_1pct"] = float(np.mean(abs(group.pixel_over_cat-1)<.01))
        out.append(row)
    # Check selected edge pixels against an independent adaptive integration.
    errs=[]
    for x,y in ((2.6,.4),(1.9,1.7),(-2.8,-.6),(.15,2.8)):
        a,b,c,d=x-.5,x+.5,y-.5,y+.5
        def vertical(u):
            if abs(u) >= FIBER_RADIUS_PIX:
                return 0.
            s=np.sqrt(FIBER_RADIUS_PIX**2-u*u)
            return max(0.,min(d,s)-max(c,-s))
        direct=quad(vertical,a,b,epsabs=1e-10,points=[p for p in (-FIBER_RADIUS_PIX,FIBER_RADIUS_PIX) if a<p<b],limit=1000)[0]
        errs.append(abs(circle_pixel_area(a,b,c,d,FIBER_RADIUS_PIX)-direct))
    summary=dict(status="passed", n=len(result), groups=out,
                 max_area_check_abs=float(max(errs)),
                 matched_input_sha256=sha(HERE/"matched_legacy_bricks.csv"),
                 reference_source_sha256=sha(HERE/"reference/tractor_dr9_4/sersic.py"),
                 redrock_run=False)
    if max(errs)>1e-7:
        raise ValueError("circle-pixel overlap failed independent check")
    (HERE/"pixel_diagnosis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
