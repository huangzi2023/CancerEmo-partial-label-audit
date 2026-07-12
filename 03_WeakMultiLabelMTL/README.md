# Weak Multi-Label Multi-Task Learning

Goal: Train one shared encoder + eight sigmoid heads on Dataset B under weak-zero BCE.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `03_WeakMultiLabelMTL.ipynb`: notebook for this model set

## Expected Inputs

- `dataset_B_weak_multilabel/merged_weak_multilabel.csv`
- `dataset_B_weak_multilabel/sentence_splits.csv`

## Expected Outputs

- `outputs/03_mtl_weak_metrics.csv`
- `outputs/03_mtl_weak_predictions.csv`

## Notes

- All non-positive cells in Dataset B are treated as 0 during training. This is the naive multi-task baseline.
- Architecture: pretrained encoder → [CLS] pooling → dropout → `Linear(hidden, 8)` with sigmoid. Loss is plain `BCEWithLogitsLoss`.
- The macro F1 comparator is the masked variant in step 04. If the two converge, weak-zero is robust enough; if step 04 wins by a wide margin, the weak-zero assumption is biasing the model.
- Default encoder is `distilbert-base-uncased` for CPU smoke-tests; switch to `bert-base-uncased` on GPU for the publication target.
