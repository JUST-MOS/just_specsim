"""Compare DR9.4 Tractor profile mixtures with analytic Sersic and LS catalog.

Reads, but does not execute, the separately retained public Tractor source.
The Gaussian coefficients are extracted as literal data from its AST; the
interpolation and circular-aperture integration are computed independently.
No optical spectra are generated and Redrock is not called.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
REF = HERE / "reference/tractor_dr9_4"
PRIOR = STUDY / "sersic_fiber_model_20261121/fiber_coupling_per_target.csv"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def assigned_literal(body, target):
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if ast.unparse(node.targets[0]) == target:
                value = node.value
                if isinstance(value, ast.Call):
                    value = value.args[0]
                return ast.literal_eval(value)
    raise ValueError("missing reference coefficient: {}".format(target))


class TractorDR94Profiles(object):
    """Numerical reconstruction using coefficients from the cited dr9.4 file."""

    def __init__(self):
        source = ast.parse((REF/"sersic.py").read_text())
        c = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "SersicMixture")
        init = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        fits, cores = assigned_literal(init.body, "fits"), assigned_literal(init.body, "self.cores")
        self.ranges = []
        for lo, hi, rows in fits:
            rows_x = np.array([r[0] for r in rows])
            amps = np.array([r[1] for r in rows])
            variances = np.array([r[2] for r in rows])
            order = 1 if len(rows) <= 3 else 3
            self.ranges.append((lo, hi,
                [InterpolatedUnivariateSpline(rows_x, amps[:, k], k=order) for k in range(amps.shape[1])],
                [InterpolatedUnivariateSpline(rows_x, np.log(variances[:, k]), k=order) for k in range(amps.shape[1])]))
        self.core = InterpolatedUnivariateSpline([c[0] for c in cores], [c[1] for c in cores], k=3)
        m = ast.parse((REF/"mixture_profiles.py").read_text())
        self.exp_amp = np.asarray(assigned_literal(m.body, "exp_amp"), dtype=float)
        self.exp_var = np.asarray(assigned_literal(m.body, "exp_var"), dtype=float)
        self.dev_amp = np.asarray(assigned_literal(m.body, "dev_amp"), dtype=float)
        self.dev_var = np.asarray(assigned_literal(m.body, "dev_var"), dtype=float)
        self.dev_core = float(assigned_literal(m.body, "dev_core"))

    def profile(self, morph, index, include_core=True):
        if morph in ("REX", "EXP"):
            amp = self.exp_amp / self.exp_amp.sum()
            return amp, self.exp_var.copy()
        if morph == "DEV":
            amp = self.dev_amp / self.dev_amp.sum()
            if include_core:
                amp = np.r_[amp * (1-self.dev_core), self.dev_core]
                return amp, np.r_[self.dev_var, 0.]
            return amp, self.dev_var.copy()
        if morph != "SER":
            raise ValueError("unsupported morphology {}".format(morph))
        value = float(np.clip(index, self.ranges[0][0], self.ranges[-1][1]))
        matches = [r for r in self.ranges if r[0] <= value < r[1]]
        if not matches and value == self.ranges[-1][1]:
            matches = [self.ranges[-1]]
        if len(matches) not in (1, 2):
            raise ValueError("reference Sersic index matched unexpected number of ranges")
        parts = []
        for lo, hi, amps, variances in matches:
            a = np.array([f(value) for f in amps])
            parts.append((a/a.sum(), np.exp([f(value) for f in variances])))
        if len(parts) == 2:
            fraction = (value-matches[1][0])/(matches[0][1]-matches[1][0])
            amp = np.r_[(1-fraction)*parts[0][0], fraction*parts[1][0]]
            var = np.r_[parts[0][1], parts[1][1]]
        else:
            amp, var = parts[0]
        if include_core and value > 1.:
            core = float(self.core(value))
            amp = np.r_[amp*(1-core)/amp.sum(), core]
            var = np.r_[var, 0.]
        if not np.isclose(amp.sum(), 1., rtol=1e-12):
            raise ValueError("profile normalization error")
        return amp, var


_x, _weight = leggauss(256)
_angles = (_x+1)*np.pi/4
_cos2 = np.cos(_angles)**2
_sin2 = np.sin(_angles)**2


def continuous_fiber_fraction(amp, var, re_arcsec, axis_ratio, seeing_sigma=1./2.35,
                              fiber_radius=.75):
    """Convolved elliptical Gaussian mixture in centered circular aperture."""
    amp, var = np.asarray(amp), np.asarray(var)
    cx = seeing_sigma**2 + var[:, None]*re_arcsec**2
    cy = seeing_sigma**2 + var[:, None]*(axis_ratio*re_arcsec)**2
    inverse_radius2 = _cos2[None, :]/cx + _sin2[None, :]/cy
    term = -np.expm1(-.5*fiber_radius**2*inverse_radius2) / (np.sqrt(cx*cy)*inverse_radius2)
    probability = .5 * np.sum(term*_weight[None, :], axis=1)
    return float(np.dot(amp, probability))


def main():
    model = TractorDR94Profiles()
    d = pd.read_csv(PRIOR)
    rows = []
    for row in d.itertuples(index=False):
        amp, var = model.profile(row.morphology, row.SERSIC)
        estimate = continuous_fiber_fraction(amp, var, row.SHAPE_R, row.axis_ratio)
        without_amp, without_var = model.profile(row.morphology, row.SERSIC, include_core=False)
        without = continuous_fiber_fraction(without_amp, without_var, row.SHAPE_R, row.axis_ratio)
        if not 0 < estimate <= 1:
            raise ValueError("invalid fiber fraction for {}".format(row.TARGETID))
        rows.append(dict(TARGETID=row.TARGETID, split=row.split, morphology=row.morphology,
                         SERSIC=row.SERSIC, SHAPE_R=row.SHAPE_R, axis_ratio=row.axis_ratio,
                         legacy_catalog_frac=row.legacy_catalog_frac,
                         analytic_frac=row.legacy_new_frac,
                         tractor_cont_frac=estimate, tractor_no_core_frac=without,
                         analytic_over_catalog=row.legacy_new_over_catalog,
                         tractor_over_catalog=estimate/row.legacy_catalog_frac,
                         tractor_over_analytic=estimate/row.legacy_new_frac,
                         core_fractional_change=estimate/without-1.))
    result = pd.DataFrame(rows)
    result.to_csv(HERE/"tractor_continuous_comparison.csv", index=False)
    test = result[result.split == "test"]
    groups = {"test_all":test, "test_n_ge_5":test[test.SERSIC>=5]}
    groups.update({"test_"+m:test[test.morphology==m] for m in ("REX","EXP","DEV","SER")})
    out = []
    for label, group in groups.items():
        row=dict(group=label, n=len(group))
        for col in ("analytic_over_catalog","tractor_over_catalog","tractor_over_analytic","core_fractional_change"):
            for p, value in zip((16,50,84), np.percentile(group[col], [16,50,84])):
                row[col+"_p{}".format(p)] = float(value)
        row["tractor_within_1pct"] = float(np.mean(abs(group.tractor_over_catalog-1)<.01))
        out.append(row)
    pd.DataFrame(out).to_csv(HERE/"tractor_continuous_summary.csv", index=False)
    result_json = dict(n_total=len(result), n_test=len(test), groups=out,
        tractor_release="dr9.4", tractor_commit="d476ade2c01e29849b42cd3aecaea760f7652a91",
        reference_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                                 (REF/"sersic.py", REF/"mixture_profiles.py", REF/"LICENSE", PRIOR, Path(__file__))},
        note="continuous centered aperture; no Tractor package or Legacypipe pixel rendering executed",
        redrock_run=False)
    (HERE/"tractor_continuous_summary.json").write_text(json.dumps(result_json, indent=2)+"\n")
    print(pd.DataFrame(out).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
