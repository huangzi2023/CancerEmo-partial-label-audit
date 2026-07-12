#!/usr/bin/env python3
"""Aggregate multi-seed Tier 1/2/3 runs into mean ± std (Paper A, List A #4).

The training scripts each take a single --seed and overwrite a fixed
output path. The Colab multi-seed cell (see COLAB_RUNBOOK.md) runs all
three tiers for several seeds and copies each run's per-emotion metrics
CSV into:

    <root>/seed<S>/02_bert_metrics.csv      (Tier 1, independent BERT)
    <root>/seed<S>/03_mtl_weak_metrics.csv  (Tier 2, weak-zero MTL)
    <root>/seed<S>/04_mtl_masked_metrics.csv(Tier 3, masked MTL)

This script reads those, computes each tier's macro F1 per seed
(mean of per-emotion f1, skipping NaN), and reports mean ± std across
seeds — both at the macro level and per emotion.

Run (CPU, instant):
    python scripts/aggregate_multiseed.py --root multiseed_results
Outputs -> <root>/multiseed_macro_summary.csv
           <root>/multiseed_per_emotion.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["Anger", "Anticipation", "Disgust", "Fear",
            "Joy", "Sadness", "Surprise", "Trust"]

TIERS = {
    "Tier1_bert_binary": "02_bert_metrics.csv",
    "Tier2_weak_zero": "03_mtl_weak_metrics.csv",
    "Tier3_masked": "04_mtl_masked_metrics.csv",
}


def macro_f1(df: pd.DataFrame) -> float:
    return float(np.nanmean(df.set_index("emotion").reindex(EMOTIONS)["f1"].values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="multiseed_results",
                    help="dir containing seed<N>/ subdirs")
    args = ap.parse_args()

    root = Path(args.root)
    seed_dirs = sorted(
        [d for d in root.glob("seed*") if d.is_dir()],
        key=lambda p: int(re.sub(r"\D", "", p.name) or 0),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"no seed*/ subdirs under {root.resolve()}")
    print(f"Found {len(seed_dirs)} seed dirs: {[d.name for d in seed_dirs]}")

    macro_rows = []          # (tier, seed, macro_f1)
    per_emo_rows = []        # (tier, emotion, seed, f1)
    for d in seed_dirs:
        seed = int(re.sub(r"\D", "", d.name) or 0)
        for tier, fname in TIERS.items():
            fp = d / fname
            if not fp.exists():
                print(f"  ! missing {fp} — skipping")
                continue
            df = pd.read_csv(fp)
            macro_rows.append({"tier": tier, "seed": seed, "macro_f1": macro_f1(df)})
            ind = df.set_index("emotion").reindex(EMOTIONS)
            for emo in EMOTIONS:
                per_emo_rows.append({
                    "tier": tier, "emotion": emo, "seed": seed,
                    "f1": float(ind.loc[emo, "f1"]) if not pd.isna(ind.loc[emo, "f1"]) else np.nan,
                })

    macro_df = pd.DataFrame(macro_rows)
    summary = (
        macro_df.groupby("tier")["macro_f1"]
        .agg(["mean", "std", "count"])
        .reindex(list(TIERS.keys()))
        .reset_index()
    )
    summary.columns = ["tier", "macro_f1_mean", "macro_f1_std", "n_seeds"]

    per_emo_df = pd.DataFrame(per_emo_rows)
    per_emo_summary = (
        per_emo_df.groupby(["tier", "emotion"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std"})
    )

    print("\n=== Macro F1 across seeds (mean ± std) ===")
    for _, r in summary.iterrows():
        std = 0.0 if pd.isna(r["macro_f1_std"]) else r["macro_f1_std"]
        print(f"  {r['tier']:20s} {r['macro_f1_mean']:.4f} ± {std:.4f}  (n={int(r['n_seeds'])})")

    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "multiseed_macro_summary.csv", index=False)
    per_emo_summary.to_csv(root / "multiseed_per_emotion.csv", index=False)
    print(f"\nWrote {root / 'multiseed_macro_summary.csv'}")
    print(f"Wrote {root / 'multiseed_per_emotion.csv'}")


if __name__ == "__main__":
    main()
