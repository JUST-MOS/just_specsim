# Result index and provenance

- [Updated fiber model: main report](tractor_redrock_20261121/analysis_extended/RESULTS_zh.md),
  [recovery figure](tractor_redrock_20261121/analysis_extended/recovery_comparison.png),
  [individual outcomes](tractor_redrock_20261121/analysis_extended/individual_results.csv),
  [success curves](tractor_redrock_20261121/analysis_extended/success_curves.csv), and
  [thresholds](tractor_redrock_20261121/analysis_extended/exposure_thresholds.csv).
- [Frozen sample and design](leader_195_205_20261116/extended_20p5/input/frozen_design.json)
  and its [per-target catalog](leader_195_205_20261116/extended_20p5/input/frozen_sample.csv).
- [Original circular fiber model](leader_195_205_20261116/extended_20p5/analysis_20261119/RESULTS_zh.md)
  and [fiber-model validation](high_sersic_legacy_20261121/REPORT_zh.md).
- [Historical nominal exposure study](effective_bgs_20261121/RESULTS_zh.md),
  [updated-fiber effective-depth proxy](effective_bgs_proxy_20261123/RESULTS_zh.md),
  and [illustrative point-estimate interpolation](effective_bgs_point95_20261123/RESULTS_zh.md).
- [Later smooth-atmosphere candidate](atmosphere_qiu_calibrated_20261115/README_zh.md):
  this was not substituted into the Redrock results above.

The updated result's main criterion is a one-sided 95% Clopper–Pearson lower
bound reaching 95%, adjusted for eight tested times. First passing **tested
shutter-open times** are 500 s in both skies for r < 19.5, and 750 s dark /
1500 s bright for r < 20.5. They are not exact minima or standard-dark nominal
effective depths; 125/250/500 s were added after examining the 750 s result.

The result files are retained from the original study. Two supporting reports
have link-only portability edits; [publication_edits.json](publication_edits.json)
records the original and published hashes. The full unmodified reports, input
spectra and historical run records remain in the Gravity archive. Historical
delivery manifests record checks made in the original directory, not a claim
that every original input is included in this Git repository.
Run `python research/lenghu_bgs/verify_results.py` from the repository checkout
to recompute the saved success curves and thresholds without the large spectra.
