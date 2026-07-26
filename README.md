# Hidden Partial-Label Structure in CancerEmo: A Reproducibility Audit for Digital Health NLP

Analysis code for a reproducibility audit of the **CancerEmo** benchmark
(Sosea & Caragea, EMNLP 2020) for fine-grained emotion detection in
online cancer-community text.

The CancerEmo corpus is released as **eight independent per-emotion binary
CSV files**. This repository audits that release at the unique-sentence
level, reconstructs three aligned dataset encodings from the same source,
and compares independent-binary, weak-zero multi-task, and masked
partial-label multi-task BERT models — showing that treating unobserved
sentence–emotion cells as negatives (the default of a naive merge)
collapses rare-emotion F1, while an observation-masked loss recovers it.

> Manuscript under review. Full citation will be added on publication.

## Repository layout

```
scripts/                         # all analysis code (callable directly)
01_DataAuditAndBuild/            # cross-file audit; build Datasets A/B/C
02_BinaryBaselineReplication/    # classical + independent binary BERT (Tier 1)
03_WeakMultiLabelMTL/            # weak-zero shared-encoder MTL (Tier 2)
04_PartialLabelMaskedMTL/        # masked partial-label MTL (Tier 3)
05_EvaluationBootstrap/          # paired bootstrap, macro summary, deltas
06_SplitConflictLeakage/         # split-conflict leakage diagnostics
```

Each `NN_*/` folder contains a notebook and a `README.md` describing its
inputs, configuration, and outputs. The notebooks are thin wrappers around
the scripts in `scripts/`.

## Data

The raw CancerEmo data is **not redistributed here** (obtain it from the
original release). Download the eight `*_anon.csv` files from the
`tsosea2/CancerEmo` repository and its linked Google Drive archive, then
place them in `data/raw/`:

```
data/raw/{Anger,Anticipation,Disgust,Fear,Joy,Sadness,Surprise,Trust}_anon.csv
```

Paths are resolved by a `data_dir()` helper (env var `DATA_DIR`, else
`../data`, else `./data`).

## Reproduce

```bash
pip install -r requirements.txt

# 1) audit + build Datasets A/B/C (deterministic, CPU, < 1 min)
python scripts/data_audit_and_build.py

# 2) Tier 1 — classical + independent binary BERT (GPU for BERT)
python scripts/binary_baseline_replication.py --skip-bert                 # classical
python scripts/binary_baseline_replication.py --skip-classical \
    --bert-model bert-base-uncased --epochs 3 --seed 42                    # BERT
#   clean-split comparator (Tier 1*): add  --split-source hash

# 3) Tier 2 — weak-zero MTL   |   Tier 3 — masked partial-label MTL
python scripts/mtl_weak_zero.py     --model bert-base-uncased --epochs 3 --seed 42
python scripts/mtl_partial_masked.py --model bert-base-uncased --epochs 3 --seed 42
#   Tier 2-weighted (class-weighted weak-zero control): add  --class-weighted
#   (per-emotion pos_weight = n_negative / n_positive; writes *_weighted_* outputs)
python scripts/mtl_weak_zero.py --model bert-base-uncased --epochs 3 --seed 42 --class-weighted

# 4) diagnostics + evaluation
python scripts/weak_zero_threshold_robustness.py
python scripts/split_conflict_leakage_proxy.py
python scripts/split_conflict_leakage_bert.py --dry-run
python scripts/evaluation_bootstrap.py --n-boot 1000 --seed 42

# 5) multi-seed aggregation (after running seeds 42, 1, 2)
python scripts/aggregate_multiseed.py --root multiseed_results
```

All BERT results are reported as mean ± standard deviation over three
seeds (42, 1, 2).

## License

Released under the MIT License (see `LICENSE`).
