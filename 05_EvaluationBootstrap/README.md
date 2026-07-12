# Evaluation, Bootstrap CIs, and Paired Deltas

Goal: Compute bootstrap 95 % CIs on per-(model, emotion) F1 and paired deltas across all upstream prediction files.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `05_EvaluationBootstrap.ipynb`: notebook for this model set

## Expected Inputs

- `02_BinaryBaselineReplication/outputs/02_classical_predictions/{model}__{emotion}.csv`
- `02_BinaryBaselineReplication/outputs/02_bert_predictions/{emotion}.csv`
- `03_WeakMultiLabelMTL/outputs/03_mtl_weak_predictions.csv`
- `04_PartialLabelMaskedMTL/outputs/04_mtl_masked_predictions.csv`

## Expected Outputs

- `outputs/05_bootstrap_ci.csv`
- `outputs/05_macro_summary.csv`
- `outputs/05_paired_deltas.csv`

## Notes

- Paired bootstrap: resample test sentence_ids with replacement; compute the metric on each resample for every model; report empirical 2.5 / 97.5 percentiles. Paired deltas use the SAME resample for both models so the test-set pairing is preserved.
- Primary metric: macro F1. Per-emotion is secondary.
- Subset accuracy is reported as supplementary only — Dataset B / C labels are incomplete, so subset accuracy is not a fair primary metric.
- No training here; safe to run on CPU in under a minute at 1,000 bootstrap iterations.
