# Binary Baseline Replication

Goal: Reproduce the eight per-emotion binary task baselines from Sosea & Caragea (2020) Table 4 on Dataset A.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `02_BinaryBaselineReplication.ipynb`: notebook for this model set

## Expected Inputs

- `dataset_A_binary/{Anger,…,Trust}.csv`

## Expected Outputs

- `outputs/02_baseline_metrics.csv`
- `outputs/02_macro_average_f1.csv`
- `outputs/02_classical_predictions/{model}__{emotion}.csv`
- `outputs/02_bert_metrics.csv`
- `outputs/02_bert_predictions/{emotion}.csv`

## Notes

- Phase 1 (classical, CPU): Majority, TF-IDF + Logistic Regression, TF-IDF + Naive Bayes. Runs in under a minute.
- Phase 2 (transformer, GPU): default `bert-base-uncased`, 3 epochs, batch 32, lr 2e-5. Use `--skip-bert` for CPU-only runs.
- Target macro F1: BERT ≈ 0.71 per the original paper. Classical baselines should land within a few F1 points of the original NB / LR ≈ 0.60-0.61.
- All models train on `split == 0`, evaluate on `split == 2`. The validation split is not used (no hyperparameter tuning).
