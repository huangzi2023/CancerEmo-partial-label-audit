# Data Audit and Dataset Construction

Goal: Audit the eight per-emotion CancerEmo binary CSV files and build Datasets A, B, and C.

## Folder Layout

- `data/`: input data for this model set
- `outputs/`: generated metrics, predictions, checkpoints, and figures
- `01_DataAuditAndBuild.ipynb`: notebook for this model set

## Expected Inputs

- `Anger_anon.csv`, `Anticipation_anon.csv`, `Disgust_anon.csv`, `Fear_anon.csv`, `Joy_anon.csv`, `Sadness_anon.csv`, `Surprise_anon.csv`, `Trust_anon.csv`

## Expected Outputs

- `processed/audit/audit_binary_files.csv`
- `processed/audit/sentence_overlap_audit.csv`
- `processed/audit/split_conflict_report.csv`
- `processed/audit/emotion_cooccurrence_observed.csv`
- `processed/dataset_A_binary/{Anger,…,Trust}.csv`
- `processed/dataset_B_weak_multilabel/merged_weak_multilabel.csv`
- `processed/dataset_B_weak_multilabel/sentence_splits.csv`
- `processed/dataset_C_partial_label/merged_partial_label.csv`

## Notes

- 19,914 total rows across the eight files; 11,642 unique sentences; 1,381 with ≥2 observed positive emotions; 2,289 with split conflicts; 50 % of sentences appear in ≥2 emotion files.
- Dataset B encodes 0 as a WEAK negative; Dataset C encodes 0 as observed negative and leaves NA where the emotion is unobserved.
- Sentence-level split: deterministic md5 hash mod 10 → 80 / 10 / 9 train / val / test. The same sentence_id always lands in the same split.
- Dataset C reuses Dataset B's `sentence_splits.csv` — both share the same sentence_id space.
