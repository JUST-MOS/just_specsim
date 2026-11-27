# JUST 4200 m: Tractor-profile Redrock rerun

## Scope and scientific design

- User authorized full Redrock reruns using the improved catalog-consistent galaxy fiber model, with old/new comparison for `r₀ < 19.5` and `r₀ < 20.5`. The 1332 frozen test galaxies include the same 578 bright targets and independent held-out roster as the original completed five-time sky scan.
- The five prespecified total times are 750, 1500, 2250, 3000 and 3600 s; sky cases are dark and bright. Use the frozen exposure splitting, 4200 m sky/instrument, segmented Qiu+ESO atmosphere used by that original scan, 800 nm seeing = 0.8 arcsec, and per-target Galactic extinction. The later smooth atmosphere candidate is not substituted in this paired comparison.
- Change the source spatial profile from the original circular exponential to the continuous Tractor DR9.4 Gaussian-mixture profile with frozen catalog morphology, ellipticity and Sérsic index. Legacy pixelized aperture measurement is not applied to the JUST fiber. Input is a total-light SED. Offset 0.03 arcsec is assumed along the major axis; wavelength-dependent morphology is not modeled.
- Each target/sky/time uses the frozen `sha256(20261114:TARGETID:sky:total_seconds:arm)` seed. If the old/new valid pixel masks differ, the same seed does not imply that every common pixel receives the same variate; record masks rather than assume exact noise pairing. In the five-target pilot the masks are identical.
- Success requires finite Redrock `Z`, `ZWARN=0`, `DELTACHI2>40`, and `abs(Z - VI_Z)/(1 + VI_Z) < 0.0033`; missing fits remain in the denominator. The main threshold is the first **tested** time with one-sided Clopper–Pearson lower confidence bound at least 0.95, with 0.05/5 Bonferroni correction across the five exposure times for each magnitude group and sky. Point estimates and pointwise bounds are secondary. There is no cross-group correction.
- The inherited quality cut has not been independently re-tuned for this changed source model; references and source spectra are not fully independent; the field, site sky and engineering instrument are conditional simulations, not measured JUST performance.

## Inputs and pilot

- Frozen sample and split: `../leader_195_205_20261116/extended_20p5/input/frozen_sample.csv`, SHA-256 `ae63f8abff1113103ef35094324480e35b5dd92a6dea91f0535ed68470f05404`.
- Previous per-scenario results: `../leader_195_205_20261116/extended_20p5/analysis_20261119/individual_results.csv`, validated against its analysis manifest before final analysis.
- Model and reference provenance: `../high_sersic_legacy_20261121/mixture_verification.json` and `simulate_tractor.py` manifest fields. The original simulator is called as an imported module without modification; only its source-arm function is replaced.
- The first five galaxies cover EXP, DEV and SER, including two with `n >= 5`; 50 case spectra (both skies, all five times). The original rerun reproduces all 12 spectral arrays of the archived first five objects exactly; paired old/new target, wavelength, mask and seed checks pass in `pilot/input_validation.json`.
- The Redrock pilot is in `pilot/`. The full run starts only after the pilot is checked against its spectra/truth and output FITS.
- Pilot completed: 50 attempted, 50 returned, and 50 successful (245.95 s). `run_scan.validate_spectra(0,5,...)` and `validate_fit(0,...)` passed. This is a five-target interface test, not a population success estimate.

## Live run

- Started 2026-11-21 17:41 China time using the command below, in exec session `41853`. Initial checkpoint: pilot batch complete and batch starting at test index 5 running. Read the atomic `run/progress.json` and current batch logs for actual current status; this note is not a live progress feed.
- At 17:58 +08, the pilot plus the first 50-target batch had completed (55/1332 galaxies, 550/13320 spectra, all fits returned). The first full batch's Redrock phase took 964.272 s, implying a rough remaining wall time of about 7 h, excluding variable simulation overhead; batch starting at index 55 was active.
- At 18:14 +08, pilot and the first two 50-target batches had completed (105/1332 galaxies, 1050/13320 spectra, all fits returned); the second full batch took 933.009 s. Batch starting at index 105 is active.
- At 18:31 +08, 155/1332 galaxies and 1550/13320 spectra had completed, with all fits returned and checked. The third full batch took 977.621 s; the batch starting at index 155 is active.
- At approximately 18:47 +08, 205/1332 galaxies and 2050/13320 spectra had completed, with every fit returned and checked. The fourth full batch took 937.067 s; the batch starting at index 205 is active.
- At 19:04 +08, 255/1332 galaxies and 2550/13320 spectra had completed, all returned and checked. The fifth full batch took 919.436 s; batch starting at index 255 is active. `vm_stat` showed a transient system-wide swapout increase followed by a stable interval; `memory_pressure -Q` reported 74% free during that batch. Cause is not established.
- At 19:20 +08, 305/1332 galaxies and 3050/13320 spectra had completed, all returned and checked. The sixth full batch took 931.995 s; batch starting at index 305 is active.
- At 19:37 +08, 355/1332 galaxies and 3550/13320 spectra had completed, all returned and checked. The seventh full batch took 915.927 s; batch starting at index 355 is active.
- At 19:53 +08, 405/1332 galaxies and 4050/13320 spectra had completed, all returned and checked. The eighth full batch took 923.850 s; batch starting at index 405 is active.
- At 20:09 +08, 455/1332 galaxies and 4550/13320 spectra had completed, all returned and checked. The ninth full batch took 938.485 s; batch starting at index 455 is active.
- At 20:26 +08, 505/1332 galaxies and 5050/13320 spectra had completed, all returned and checked; batch starting at index 505 is active.
- At 20:42 +08, 555/1332 galaxies and 5550/13320 spectra had completed, all returned and checked; batch starting at index 555 is active.
- At 20:59 +08, 605/1332 galaxies and 6050/13320 spectra had completed, all returned and checked; batch starting at index 605 is active.
- At 21:15 +08, 655/1332 galaxies and 6550/13320 spectra had completed, all returned and checked; batch starting at index 655 is active.
- At 21:32 +08, 705/1332 galaxies and 7050/13320 spectra had completed, all returned and checked; batch starting at index 705 is active.
- At 21:48 +08, 755/1332 galaxies and 7550/13320 spectra had completed, all returned and checked; batch starting at index 755 is active.
- At 22:04 +08, 805/1332 galaxies and 8050/13320 spectra had completed, all returned and checked; batch starting at index 805 is active.
- At 22:21 +08, 855/1332 galaxies and 8550/13320 spectra had completed, all returned and checked. Batch starting at index 855 is active; once it finishes, all 578 galaxies in the r0<19.5 test subset have complete cases and can be analyzed while the rest runs.
- At 22:37 +08, 905/1332 galaxies and 9050/13320 spectra had completed, all returned and checked. The complete r0<19.5 subset contains 578 galaxies; `analyze_bright_when_ready.py` was started independently while batch 905 proceeds.
- At 22:54 +08, 955/1332 galaxies and 9550/13320 spectra had completed, all returned and checked; batch starting at index 955 is active.
- At 23:10 +08, 1005/1332 galaxies and 10050/13320 spectra had completed, all returned and checked; batch starting at index 1005 is active.
- At 23:26 +08, 1055/1332 galaxies and 10550/13320 spectra had completed, all returned and checked; batch starting at index 1055 is active.
- Main scan completed at 2026-11-22 00:57:20 +08: 1332/1332 galaxies, 13320/13320 fits returned and checked; `run/progress.json` is complete.
- The full r0<19.5 subset has 573/578 dark and 572/578 bright successes at 750 s under the mixture model, with five-time-adjusted lower bounds 97.748% and 97.498%. Both pass at the earliest tested time, so no minimum below 750 s can be inferred. The old model first passed at 3000 s dark and was not confirmed through 3600 s bright. `early_bright/` records per-time curves, matched outcomes and hashes. A separate SciPy exact-binomial check independently confirms both 750-s adjusted passes.
- For the 750-s old-failure/new-success objects, `diagnose_coupling.py --group bright` reports median 623.1-nm new/original source coupling of 3.08 (bright sky; 52 objects) and 3.59 (dark sky; 44 objects). The complete paired table, percentiles and file hashes are in `early_bright/coupling_transition*`. This association reflects the combined profile changes, not a causal isolation of the high-n adjustment.
- Independently tested the old circular exponential Fourier integral on all 1332 sources using 100 nodes (du=.012), 400 nodes (du=.003) and 800 nodes (du=.003 with twice the cutoff), at 623.1 nm and survey seeing. For the complete 578-object bright subset, extended/original fraction has median 0.9995865 and minimum 0.9764937; these numerical differences do not explain the multi-fold aperture changes of the recovering objects. Saved old fractions match the original 100-node calculation. See `early_bright/original_quadrature*`.
- For the first full batch, all 1500 old/new spectrum-arm masks were also compared pixel by pixel against the SHA-verified historical archives and matched exactly; see `run/first_batch_mask_validation.json`. This does not establish pixelwise equivalence for the remaining batches.
- First full-batch spectra occupy 843.98 MiB for 50 targets. The rough expected total retained spectra size is about 22 GiB, with approximately 87 GiB filesystem space free after the first batch was generated.
- Historical self-comparison of `analyze_scan.py` was checked against independent `scipy.stats.binomtest`; `smoke_analysis.py` generated and removed temporary example plots/report successfully. It produced no simulated science results.

## Reproducible continuation

### Follow-up below the original 750 s floor

The complete bright sample already met the adjusted 95% threshold at the first 750-s point, so this grid gives only an upper limit. Before examining any short-time outcomes, selected 125, 250 and 500 s, one read each, for both skies and all 1332 test targets. These follow-up times were selected after seeing the 750-s result; final limits use alpha=.05/8 across the five original and three added times, and state this sequential design. Old-model values below 750 s have not been simulated. `simulate_short.py`, `short_scan.py` and `verify_short_pilot.py` are separate from the original locked driver. The five-galaxy, 30-spectrum short input pilot passed 90 independently reconstructed arm checks including seed, variance, resolution and mask (see `short_run/pilot/input_validation.json`). Run its Redrock pilot and the rest only after the main scan completes; do not run concurrent Redrock fits. Every short case is one exposure, with noise recomputed accordingly. The short and original runs can reuse synthetic IDs for different times; join them only on reference TARGETID, sky and total_s.

At approximately 23:32 +08, started `prefetch_short_spectra.py` in session 76260 while the main fit continued. It serially generated and validated the short input spectra only, without Redrock fits. System-wide swapouts rose by 181756 pages (16 KiB/page) over an observed interval, so a stop signal was requested; by the time it was executed, the prefetch process had already completed all 1332 objects at 23:50:37 +08. The signal therefore returned no such process; no files were interrupted. Its atomic ledger `short_run/prefetch_progress.json` is complete and the shared lock is released. The cause of system-wide swap activity was not isolated. On resumption at 2026-11-22 17:08 +08, system memory pressure reported 74% free and the filesystem had 54 GiB available.

At 17:08 +08 started `short_scan.py` in session 56938; this is now the only active Redrock fit, using the completed prefetched input. Full original-grid analysis `analyze_scan.py` is running separately in session 23425; it is read-only with respect to the fitted spectra. Because the short fit started first, the historical 50-case old-model refit must wait until the short Redrock scan finishes. Do not run concurrent Redrock fits.

The 5-galaxy / 30-case short Redrock pilot finished with 30/30 returned and passed batch validation; batch start 5 is active. The five-time full analysis has completed; under the planned eight-time alpha=.00625, the 1332-object r0<20.5 sample at 750 s has 1290/1332 successes and lower bound 95.445% dark, and 1273/1332 with lower 93.963% bright. Bright at 1500 s has 1314/1332 and lower 97.639%. An independent SciPy exact-binomial test confirms those threshold decisions. The first passing time on the original five-point grid is 750 s dark / 1500 s bright; short exposures remain untested as full samples. The prior circular model first passes at 3000 s dark and is unconfirmed through 3600 s bright under the same alpha. At 750 s, 132 dark and 161 bright full-sample targets change old-failure to new-success; none reverse. `analysis/coupling_transition*` records the paired geometry diagnostic; no independent pixel-mask or historical-refit audit has yet run.

`audit_main_five.py` independently recomputed all 40 original-grid success counts from fit/quality/truth columns and SciPy exact-binomial bounds at alpha=.00625; it confirms all previously reported first passing times on that five-time grid. Its explicitly provisional output is in `analysis/provisional_eight_audit.json`, `provisional_eight_bounds.csv` and `provisional_eight_thresholds.csv`; the three shorter times are not yet included.

At 2026-11-22 17:24 +08, short Redrock had completed 55/1332 galaxies and 330/7992 cases, all returned. Its first 50-galaxy batch took 695.035 s, implying roughly 5 h for the remaining 26 batches at similar speed; batch start 55 is active. Live ledger: `short_run/run/progress.json`; do not report partial ordered samples as population results.

At 17:36 +08, short Redrock reached 105/1332 galaxies and 630/7992 cases, all returned. The second 50-galaxy fit took 677.987 s; batch start 105 is active in session 56938.

At 17:47 +08, short Redrock reached 155/1332 galaxies and 930/7992 cases, all returned. The third 50-galaxy fit took 681.2 s; batch start 155 is active.

At 17:58 +08, short Redrock reached 205/1332 galaxies and 1230/7992 cases, all returned. The fourth full fit took 684.971 s; batch start 205 is active.

At 18:10 +08, short Redrock reached 255/1332 galaxies and 1530/7992 cases, all returned. The fifth full fit took 686.736 s; batch start 255 is active.

At 18:21 +08, short Redrock reached 305/1332 galaxies and 1830/7992 cases, all returned. The sixth full fit took 681.955 s; batch start 305 is active.

At 18:33 +08, short Redrock reached 355/1332 galaxies and 2130/7992 cases, all returned. The seventh full fit took 661.662 s; batch start 355 is active.

At 18:45 +08, short Redrock reached 405/1332 galaxies and 2430/7992 cases, all returned. The eighth full fit took 678.109 s; batch start 405 is active.

At 18:56 +08, short Redrock reached 455/1332 galaxies and 2730/7992 cases, all returned. The ninth full fit took 664.667 s; batch start 455 is active.

At 19:07 +08, short Redrock reached 505/1332 galaxies and 3030/7992 cases, all returned. The tenth full fit took 655.711 s; batch start 505 is active.

At 19:18 +08, short Redrock reached 555/1332 galaxies and 3330/7992 cases, all returned. The eleventh full fit took 658.592 s; batch start 555 is active.

At 19:29 +08, short Redrock reached 605/1332 galaxies and 3630/7992 cases, all returned. The twelfth full fit took 657.39 s; batch start 605 is active.

At 19:40 +08, short Redrock reached 655/1332 galaxies and 3930/7992 cases, all returned. The thirteenth full fit took 662.536 s; batch start 655 is active.

At 19:51 +08, short Redrock reached 705/1332 galaxies and 4230/7992 cases, all returned. The fourteenth full fit took 687.673 s; batch start 705 is active.

At 20:03 +08, short Redrock reached 755/1332 galaxies and 4530/7992 cases, all returned. The fifteenth full fit took 692.473 s; batch start 755 is active. After batch start 855 finishes, the entire 578-object bright subset can be analyzed with `analyze_eight_bright_when_ready.py` while the faint extension continues.

At 20:15 +08, short Redrock reached 805/1332 galaxies and 4830/7992 cases, all returned. The sixteenth full fit took 665.633 s; batch start 805 is active. Two more batches are needed to complete the bright subset.

At 20:26 +08, short Redrock reached 855/1332 galaxies and 5130/7992 cases, all returned. The seventeenth full fit took 667.308 s; batch start 855 is active. Once this batch completes, run `analyze_eight_bright_when_ready.py` for the full 578-object bright subset while the rest continues.

At 20:37 +08, short Redrock reached 905/1332 galaxies and 5430/7992 cases, all returned. The eighteenth full fit took 677.873 s; the entire r0<19.5 test subset has now completed all short exposures. Started `analyze_eight_bright_when_ready.py` in session 20050 while the remaining galaxies continue in session 56938.

The complete r0<19.5 eight-time analysis passed its batch/provenance checks: under alpha=.05/8, 500 s is the first passing test point for both skies. Dark has 566/578 = 97.924% and lower 95.9368%; bright has 563/578 = 97.405% and lower 95.2579%. At 250 s, dark has 464/578 = 80.277%, lower 75.825%; bright 456/578 = 78.893%, lower 74.347%. SciPy exact p-values against 95% at 500 s are 0.0002520 dark and 0.0028651 bright, both below .00625. The point-estimate 95% threshold is also 500 s in both skies. Previous circular model first passed dark at 3000 s and was unconfirmed bright through 3600 s. These are tested grid points, not exact minima, and the extra times were chosen after the 750-s result. Output and hashes are in `early_bright_extended/`. The full 1332-object short scan and independent pixel/runtime audits remain in progress.

At 20:49 +08, short Redrock reached 955/1332 galaxies and 5730/7992 cases, all returned. The nineteenth full fit took 686.418 s; batch start 955 is active.

At 21:01 +08, short Redrock reached 1005/1332 galaxies and 6030/7992 cases, all returned. The twentieth full fit took 697.214 s; batch start 1005 is active.

At 21:12 +08, short Redrock reached 1055/1332 galaxies and 6330/7992 cases, all returned. The twenty-first full fit took 682.429 s; batch start 1055 is active.

At 2026-11-23 01:13 +08, the ledger still showed 1055 completed and batch start 1055, but its Redrock log modification time and template stage were advancing. The reason for the wall-clock interval was not established; the session remained active, system reported 53% free memory and 51 GiB disk available. Continue monitoring rather than restarting an active batch.

At 2026-11-23 01:19 +08, short Redrock reached 1105/1332 galaxies and 6630/7992 cases, all returned. The completed batch's recorded fit elapsed time was 1297.85 s; wall-clock interval and fit-time discrepancy were not diagnosed. Batch start 1105 is active.

At 01:33 +08, short Redrock reached 1155/1332 galaxies and 6930/7992 cases, all returned. The next full fit took 832.171 s; batch start 1155 is active.

At 01:47 +08, short Redrock reached 1205/1332 galaxies and 7230/7992 cases, all returned. The next full fit took 825.956 s; batch start 1205 is active.

At 02:00 +08, short Redrock reached 1255/1332 galaxies and 7530/7992 cases, all returned. The next full fit took 824.609 s; batch start 1255 is active.

At 02:14 +08, short Redrock reached 1305/1332 galaxies and 7830/7992 cases, all returned. The next full fit took 841.089 s; the final 27-galaxy batch start 1305 is active.

At 02:23 +08, the short scan completed: 1332/1332 galaxies and 7992/7992 cases returned, with all 28 batches validated. The eight-time analysis is running in session 63553; the deferred historical 50-scenario old-model refit is running as the only Redrock fit in session 11758. After those finish, run `verify_old_pilot_fit.py`, `audit_pixel_masks.py`, `audit_completed.py`, `audit_eight.py`, `plot_scan.py`, `report_scan.py`, `verify_delivery.py`, `plot_eight.py`, `report_eight.py`, and `verify_extended_delivery.py`; inspect the final figure. The whole-sample 500-s short-time rates have not yet been aggregated.

`bound_short_remaining.py` independently checked the completed 955-target roster, CSV hashes and raw success flags. Under the maximally favorable assumption that every one of the 377 remaining targets succeeds, the full r0<20.5 sample at 500 s can have at most 1270/1332 successes dark and 1256/1332 bright, below the 1285 needed for the alpha=.00625 lower bound to reach 95%. Both 125 and 250 s are also provably excluded. Combined with the already completed five-time grid, this logically fixes the first passing tested times for r0<20.5 at 750 s dark and 1500 s bright, regardless of the remaining short-fit outcomes. This is an upper bound, not an estimate based on a representative partial sample. Full fitting continues to obtain actual rates at all short times and complete the audits. Snapshot and proof: `analysis/short_partial_ceiling.json` and `.csv`.

The separate 7992-fit follow-up runs its 30-fit pilot before subsequent 50-target batches, revalidates outputs and checkpoints each batch. The driver rejects execution while the original fit is incomplete. In this execution the old-model pilot refit was deferred until the short scan finishes. Do not modify the two short driver/simulator files after the run plan has been created; their hashes are pinned for resumption.

```sh
/usr/bin/caffeinate -i env PYTHONPATH=/Users/rain/just_etc/py PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_CACHE_DIR=/private/tmp/just_numba MPLCONFIGDIR=/private/tmp/just_fiber_mpl /Users/rain/miniconda3/envs/desi/bin/python -s -B -u doc/validation/lenghu_redshift_recovery_20261111/tractor_redrock_20261121/short_scan.py
```

After short batch start 855 completes, the entire bright subset has short results; `analyze_eight_bright_when_ready.py` can report it while the faint extension proceeds. After the short ledger reports complete and the original analysis/audits exist, run `analyze_eight.py`, `audit_eight.py`, `plot_eight.py`, `report_eight.py`, then verify links and hashes and visually inspect the plot. A temporary synthetic smoke of eight-time analysis, SciPy binomial checks and figure generation passed; this smoke produced no scientific output. The reporting script additionally requires the completed pixel-mask audit and historical pilot refit.

Working directory: `/Users/rain/just_etc`. The managed driver freezes input/code hashes in `run/run_plan.json`, locks the run, validates each input and fit, writes atomic `run/progress.json`, and preserves incomplete files in `run/interrupted/` before regenerating them. It uses two native Redrock worker processes to remain consistent with the previous stable runtime. Existing complete files are never silently overwritten. `status.py` reports the current template and its progress as well as the completed target counts; template progress is not overall scan progress.

```sh
/usr/bin/caffeinate -i env PYTHONPATH=/Users/rain/just_etc/py PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_CACHE_DIR=/private/tmp/just_numba MPLCONFIGDIR=/private/tmp/just_fiber_mpl /Users/rain/miniconda3/envs/desi/bin/python -s -B -u doc/validation/lenghu_redshift_recovery_20261111/tractor_redrock_20261121/run_scan.py
```

After `run/progress.json` reports `complete`, run `analyze_scan.py`, `audit_pixel_masks.py`, `audit_completed.py`, `plot_scan.py`, `report_scan.py`, and `verify_delivery.py` in that order using the same environment. Final analysis checks every batch, exact scenario coverage, old-result hashes and paired pixel masks before reporting thresholds. Actual progress must be read from `run/progress.json`; the existence of this work log is not evidence that the run completed.

As an additional runtime-comparability check, once the main fit has finished, refit `pilot/original_0000_spectra.fits` with the same `fit_stable.py --processes 2` wrapper into `pilot/original_0000_redrock`, then run `verify_old_pilot_fit.py`. This compares 50 identical old-model spectra and their quality/success flags to the archived old run; it is not a substitute for the full sample scan.

### 2026-11-23 eight-time aggregation and audits

All 21,312 improved-model cases returned and `analyze_eight.py` checked the complete 28-batch short-exposure roster and provenance. The first sampled times meeting the eight-time-adjusted 95% lower-bound criterion are 500 s dark and bright for r0<19.5, and 750 s dark / 1500 s bright for r0<20.5. The latter sample's newly completed 500-s results are 1204/1332 dark and 1182/1332 bright. `audit_eight.py` independently reproduced 52 curve rows and all eight old/new thresholds with SciPy exact binomial calculations; maximum lower-bound difference is 3.47e-13. The eight-time figure has been generated and visually inspected.

The deferred original-model 50-scenario Redrock refit returned all cases and `verify_old_pilot_fit.py` passed: redshift and Δχ² matched the archived values exactly, as did all quality and success flags. The full mask comparison `audit_pixel_masks.py` is running in session 25817; final reports and delivery verification remain pending.

At 2026-11-23 02:31 +08, the full paired mask audit completed: 39,960 spectrum/arm arrays matched, with no pixel-array differences. `audit_completed.py` checked the main-grid curves and 13,320 paired scenarios. Both the five-time and eight-time reports and their delivery manifests passed link, output SHA-256 and image/PDF checks; the eight-time figure was visually inspected. The primary final deliverable is `analysis_extended/RESULTS_zh.md` and its `delivery_manifest.json`; the five-time report is retained as a historical intermediate and uses alpha=.05/5. Total study directory is approximately 35 GiB, including retained simulated spectra and checkpoints. No shared inputs or old/new results were deleted.
