#!/usr/bin/env python3
"""Weak-zero threshold-robustness check (Paper A, defends Tier 2).

The weak-zero MTL collapses to macro F1 = 0.203, with five emotions at
F1 = 0.000. A reviewer will ask the obvious question: is that collapse a
real failure of the weak-zero loss, or merely an artefact of the 0.5
decision threshold? (The Tier-2 AUROCs stay 0.52-0.86, so the ranking is
informative — only the thresholded decision is broken.)

This script answers it WITHOUT retraining, using the saved weak-zero
probabilities (03_mtl_weak_predictions.csv, column prob_pos). For each
emotion it sweeps the threshold and reports:

  - F1 @ 0.5 (the reported number)
  - best achievable F1 over all thresholds, and the threshold that gets it
  - "oracle" macro F1 if every emotion used its own best threshold

If even the oracle per-emotion threshold cannot lift macro F1 to near the
masked-MTL level (0.737), the catastrophe is NOT a thresholding artefact —
it is a genuine consequence of training the rare-emotion heads on mostly
filler negatives. That is the defensive result the paper needs.

Run (CPU, seconds):
    python scripts/weak_zero_threshold_robustness.py
Outputs -> 03_WeakMultiLabelMTL/outputs/03_weak_zero_threshold_robustness.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

EMOTIONS = ["Anger", "Anticipation", "Disgust", "Fear",
            "Joy", "Sadness", "Surprise", "Trust"]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_predictions() -> Path:
    root = repo_root()
    candidates = [
        root / "03_WeakMultiLabelMTL" / "outputs" / "03_mtl_weak_predictions.csv",
        root.parent / "outputs_extracted" / "03_WeakMultiLabelMTL" / "outputs" / "03_mtl_weak_predictions.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "03_mtl_weak_predictions.csv not found. Extract outputs.zip to "
        "thesis/journal-1/outputs_extracted/ first."
    )


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, best_threshold) scanning unique probability cutoffs."""
    if y_true.sum() == 0:  # no positives in test -> F1 undefined for positive class
        return 0.0, 0.5
    grid = np.unique(np.concatenate([[0.0], prob, [1.0]]))
    best_f1, best_t = 0.0, 0.5
    for t in grid:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def main() -> None:
    pred_path = find_predictions()
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} weak-zero prediction cells from {pred_path.name}")

    rows = []
    for emo in EMOTIONS:
        sub = df[df["emotion"] == emo]
        yt = sub["y_true"].values.astype(int)
        prob = sub["prob_pos"].values.astype(float)
        f1_at_half = f1_score(yt, (prob >= 0.5).astype(int), zero_division=0)
        f1_best, t_best = best_threshold(yt, prob)
        rows.append({
            "emotion": emo,
            "n": int(len(sub)),
            "n_pos": int(yt.sum()),
            "f1_at_0.5": f1_at_half,
            "f1_best": f1_best,
            "best_threshold": t_best,
            "lift": f1_best - f1_at_half,
        })

    res = pd.DataFrame(rows)
    macro_half = res["f1_at_0.5"].mean()
    macro_oracle = res["f1_best"].mean()

    print("\nPer-emotion threshold sweep (weak-zero probabilities):")
    print(res.to_string(index=False))
    print(f"\nMacro F1 @ 0.5 (reported)            = {macro_half:.4f}")
    print(f"Macro F1 @ oracle per-emotion thresh = {macro_oracle:.4f}")
    print(f"Oracle lift                          = {macro_oracle - macro_half:+.4f}")
    print(f"Masked-MTL reference (Tier 3)        = 0.7370")
    gap = 0.7370 - macro_oracle
    print(f"Gap from masked MTL even with oracle = {gap:+.4f}")

    verdict = (
        "Threshold tuning does NOT close the gap to masked MTL -> the "
        "catastrophe is a genuine weak-zero training failure, not a 0.5-"
        "threshold artefact."
        if gap > 0.05 else
        "Threshold tuning largely recovers performance -> reframe the "
        "catastrophe as a calibration/threshold artefact, not a loss failure."
    )
    print("\nVerdict:", verdict)

    out_dir = repo_root() / "03_WeakMultiLabelMTL" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([
        res,
        pd.DataFrame([{
            "emotion": "MACRO",
            "n": int(res["n"].sum()),
            "n_pos": int(res["n_pos"].sum()),
            "f1_at_0.5": macro_half,
            "f1_best": macro_oracle,
            "best_threshold": np.nan,
            "lift": macro_oracle - macro_half,
        }]),
    ], ignore_index=True)
    out = out_dir / "03_weak_zero_threshold_robustness.csv"
    summary.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
