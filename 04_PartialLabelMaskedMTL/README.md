# Partial-Label Masked Multi-Task Learning

Goal: Train the same shared encoder + eight sigmoid heads as step 03, but compute BCE only on observed labels.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `04_PartialLabelMaskedMTL.ipynb`: notebook for this model set

## Expected Inputs

- `dataset_C_partial_label/merged_partial_label.csv`
- `dataset_B_weak_multilabel/sentence_splits.csv` (shared with step 03 for direct comparability)

## Expected Outputs

- `outputs/04_mtl_masked_metrics.csv`
- `outputs/04_mtl_masked_predictions.csv`

## Notes

- Loss mask is the per-cell observation indicator (1 if observed, 0 if NA). BCE per cell is multiplied by the mask and normalised by the number of observed cells per batch.
- Evaluation is also masked — per-emotion metrics use only test cells where the emotion was observed.
- Same architecture and hyperparameters as step 03; only the loss differs. The direct comparison with step 03 in step 05 is the central claim of the paper.
