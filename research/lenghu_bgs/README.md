# JUST BGS redshift-recovery workflow

This research workflow reproduces the conditional Lenghu simulations used to
compare JUST exposure times with DESI BGS. It keeps the historical numerical
simulator and its Redrock 0.15.4 optimization unchanged, while constructing a
temporary, portable directory layout around separately stored inputs. Generic
instrument response, noise and fiber-coupling code comes from `just_etc`.

## Scientific scope

The test sample contains 1,332 selected DESI EDR deep-field galaxies with
0.1 < z < 0.4 and extinction-corrected total BASS-r < 20.5; 578 have r < 19.5.
The simulated source spectra are normalized FastSpecFit reference models, not
the BGS basis templates used by `SpectrumMaker`. Their reference redshifts and
source models are not fully independent, and the selection is not representative
of all galaxies at the same magnitude.

The 4200 m bright and dark sky spectra are extrapolated models, not measured
Lenghu sky spectra. The simulated runs use the original Qiu+ESO piecewise source
atmosphere, not the later smoothly joined candidate. Instrument and detector
parameters are engineering assumptions; these results are not measurements of
on-sky JUST performance. The old standard-dark direct fits used the older fiber
model; later 600 s nominal-depth mappings are exploratory and are not a direct
Redrock calibration of the updated fiber model.

See [results and version index](results/README.md) for the latest recovery
tables, old/new comparison, confidence definitions and historical analyses.

## Data and dependencies

The JUST group archive is under
`/home/Survey/JUST/share/bgs_etc_redrock_20261123/` on Gravity. Its
`historical_study/` directory contains the frozen inputs and complete results;
`redrock_templates_0p15p4/` contains the ten historically used fitting templates.
Transfer manifests and SHA-256 verification records accompany the data. The
separate `specsim_basis_templates/` directory is for the other `SpectrumMaker`
example and is not an input to this historical recovery calculation.

The workflow needs Python 3.12, `just_etc`, NumPy, SciPy, Astropy, pandas,
Numba, speclite, fitsio, DESI `desispec` and `desitarget`, plus Redrock 0.15.4
and its templates for fitting. Redrock 0.15.4 is incompatible with NumPy 2's
removal of `ndarray.newbyteorder`; use a compatible isolated stack. Verify the
actual environment and template hashes instead of relying on a successful import
alone. The Gravity setup and its tested versions are recorded alongside the run.

`frozen_code/` preserves the exact historical research scripts and their hashes
in `frozen_code_manifest.json`. The Tractor DR9.4 reference files in
`frozen_code/high_sersic_legacy_20261121/reference/tractor_dr9_4/` are upstream
GPLv2 material; their copyright notices and separate `LICENSE` are retained. The
rest of this repository follows its own stated license.

## Run a small comparison

Run from a checkout of this repository, with the dependencies above available
to the same Python interpreter used for all subprocesses. Set these examples to
the actual input archive, source package and a **new** output directory:

```sh
BGS_STUDY=/path/to/historical_study
BGS_ETC_PACKAGE=/path/to/just_etc/py/just_etc
BGS_TEMPLATES=/path/to/redrock_templates_0p15p4
BGS_OUTPUT=/path/to/new_run

python research/lenghu_bgs/workflow.py simulate \
  --data-root "$BGS_STUDY" --etc-package "$BGS_ETC_PACKAGE" \
  --count 5 --times 750 --output "$BGS_OUTPUT/spectra.fits"

python research/lenghu_bgs/workflow.py fit \
  --data-root "$BGS_STUDY" --etc-package "$BGS_ETC_PACKAGE" \
  --spectra "$BGS_OUTPUT/spectra.fits" --template-dir "$BGS_TEMPLATES" \
  --processes 2 --output-prefix "$BGS_OUTPUT/redrock"

python research/lenghu_bgs/workflow.py summary \
  --data-root "$BGS_STUDY" --results "$BGS_OUTPUT/redrock.csv" \
  --scan-count 1 --output "$BGS_OUTPUT/summary.csv"
```

Each simulated object is included in the denominator, including no result,
quality failures and wrong redshifts. The quality reference is ZWARN = 0 and
DELTACHI2 > 40; recovery requires |z_fit − z_VI| / (1 + z_VI) < 0.0033.
The summary reports the one-sided Clopper–Pearson lower bound with the specified
number of time points for Bonferroni adjustment. A small pilot is an engineering
check, not an estimate of the population success rate. Short (125/250/500 s)
and long (750–3600 s) grids use separate invocations, as in the original scan.

The driver reconstructs the original links in a temporary directory and refuses
to write output into the archived study. Historical batch drivers in
`frozen_code/` are retained for provenance and should not be run directly on a
partial or relocated archive. Use `compare_spectra.py` and
`compare_redshifts.py` to compare a generated subset against matched archived
pilot cases; compare by target, sky and exposure time rather than assuming the
same FITS row order.

## References and software provenance

- [DESI EDR FastSpecFit](https://data.desi.lbl.gov/doc/releases/edr/vac/fastspecfit/)
- [Redrock](https://github.com/desihub/redrock) and
  [The Tractor, DR9.4](https://github.com/dstndstn/tractor/tree/dr9.4)
- Kassis, T., Agarwal, V., He, Y., Patel, D. & Brueckner, A. M. (2026),
  *Scientific Agent Skills*, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065),
  used for the Astropy/FITS validation workflow.
