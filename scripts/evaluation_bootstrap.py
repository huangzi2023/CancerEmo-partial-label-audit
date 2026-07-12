"""
evaluation_bootstrap.py

Cross-model evaluation with paired bootstrap CIs (research plan §3.4
Experiment summary; §3.5 tables). No training; reads whatever
predictions exist on disk and computes:

    - per-(model, emotion) F1 with 95 % bootstrap CI
    - paired (model_a vs model_b) F1 delta with 95 % bootstrap CI per emotion
    - per-model macro F1 with 95 % bootstrap CI across emotions

Sources of predictions (whichever exist):

    02_BinaryBaselineReplication/outputs/02_classical_predictions/{model}__{emotion}.csv
        → majority, tfidf_lr, tfidf_nb (from binary_baseline_replication; wide-per-emotion)
    02_BinaryBaselineReplication/outputs/02_bert_predictions/{emotion}.csv
        → BERT / DistilBERT (from binary_baseline_replication; wide-per-emotion).
          Model name is recovered from 02_bert_metrics.csv if present.
    03_WeakMultiLabelMTL/outputs/03_mtl_weak_predictions.csv
        → MTL weak-zero (from mtl_weak_zero; long format with emotion column)
    04_PartialLabelMaskedMTL/outputs/04_mtl_masked_predictions.csv
        → MTL masked (from mtl_partial_masked; long format)

Bootstrap design: resample test sentence_ids with replacement; compute
metric on the resample for each model; report empirical 2.5 / 97.5
percentiles. Paired deltas use the SAME resample for both models so
the test-set sentence pairing is preserved (this is the standard
paired bootstrap, equivalent to McNemar in the limit).

Outputs:
    05_EvaluationBootstrap/outputs/05_bootstrap_ci.csv
    05_EvaluationBootstrap/outputs/05_paired_deltas.csv
    05_EvaluationBootstrap/outputs/05_macro_summary.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

EMOTIONS = [
    "Anger",
    "Anticipation",
    "Disgust",
    "Fear",
    "Joy",
    "Sadness",
    "Surprise",
    "Trust",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def collect_predictions(root: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Discover all available (model, emotion) prediction CSVs.

    Returns dict keyed by (model_name, emotion) → DataFrame with at
    least sentence_id, y_true, y_pred columns.
    """
    found: dict[tuple[str, str], pd.DataFrame] = {}

    classical_dir = root / "02_BinaryBaselineReplication" / "outputs" / "02_classical_predictions"
    if classical_dir.exists():
        for csv in classical_dir.glob("*__*.csv"):
            stem = csv.stem
            model, emotion = stem.split("__", 1)
            if emotion in EMOTIONS:
                df = pd.read_csv(csv)
                found[(model, emotion)] = df

    bert_dir = root / "02_BinaryBaselineReplication" / "outputs" / "02_bert_predictions"
    bert_metrics = root / "02_BinaryBaselineReplication" / "outputs" / "02_bert_metrics.csv"
    bert_model_name = "bert_unknown"
    if bert_metrics.exists():
        try:
            bm = pd.read_csv(bert_metrics)
            if not bm.empty and "model" in bm.columns:
                bert_model_name = str(bm["model"].iloc[0])
        except Exception:
            pass
    if bert_dir.exists():
        for csv in bert_dir.glob("*.csv"):
            emotion = csv.stem
            if emotion in EMOTIONS:
                df = pd.read_csv(csv)
                found[(bert_model_name, emotion)] = df

    weak_csv = root / "03_WeakMultiLabelMTL" / "outputs" / "03_mtl_weak_predictions.csv"
    if weak_csv.exists():
        weak = pd.read_csv(weak_csv)
        for emotion, sub in weak.groupby("emotion"):
            if emotion in EMOTIONS:
                found[("mtl_weak_zero", emotion)] = sub[["sentence_id", "y_true", "y_pred"]].copy()

    masked_csv = root / "04_PartialLabelMaskedMTL" / "outputs" / "04_mtl_masked_predictions.csv"
    if masked_csv.exists():
        masked = pd.read_csv(masked_csv)
        for emotion, sub in masked.groupby("emotion"):
            if emotion in EMOTIONS:
                found[("mtl_masked", emotion)] = sub[["sentence_id", "y_true", "y_pred"]].copy()

    return found


def bootstrap_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return (point f1, 2.5 percentile, 97.5 percentile)."""
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = f1_score(y_true, y_pred, zero_division=0)
    indices = rng.integers(0, n, size=(n_boot, n))
    samples = np.fromiter(
        (
            f1_score(y_true[idx], y_pred[idx], zero_division=0)
            for idx in indices
        ),
        dtype=float,
        count=n_boot,
    )
    return (float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def paired_bootstrap_delta(
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
    sids_a: np.ndarray,
    sids_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, bool]:
    """Align by sentence_id, then paired bootstrap of f1_a − f1_b."""
    df_a = pd.DataFrame({"sid": sids_a, "yt": y_true_a, "yp": y_pred_a})
    df_b = pd.DataFrame({"sid": sids_b, "yt": y_true_b, "yp": y_pred_b})
    merged = df_a.merge(df_b, on="sid", suffixes=("_a", "_b"))
    n = len(merged)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), False)
    yt_a = merged["yt_a"].values
    yt_b = merged["yt_b"].values
    yp_a = merged["yp_a"].values
    yp_b = merged["yp_b"].values
    point = f1_score(yt_a, yp_a, zero_division=0) - f1_score(yt_b, yp_b, zero_division=0)
    indices = rng.integers(0, n, size=(n_boot, n))
    samples = np.fromiter(
        (
            f1_score(yt_a[idx], yp_a[idx], zero_division=0)
            - f1_score(yt_b[idx], yp_b[idx], zero_division=0)
            for idx in indices
        ),
        dtype=float,
        count=n_boot,
    )
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    sig = (lo > 0.0) or (hi < 0.0)
    return (float(point), lo, hi, sig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = repo_root()
    out_dir = root / "05_EvaluationBootstrap" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = collect_predictions(root)
    if not preds:
        print(
            "No prediction CSVs found in 02_/03_/04_ outputs. "
            "Run data_audit_and_build / binary_baseline_replication / mtl_weak_zero / mtl_partial_masked first."
        )
        return

    models = sorted({m for m, _ in preds})
    emotions = sorted({e for _, e in preds}, key=EMOTIONS.index)
    print(f"Models discovered: {models}")
    print(f"Emotions discovered: {emotions}")
    print(f"Bootstrap iterations: {args.n_boot}\n")

    rng = np.random.default_rng(args.seed)

    rows = []
    for model in models:
        for emotion in emotions:
            key = (model, emotion)
            if key not in preds:
                continue
            df = preds[key]
            point, lo, hi = bootstrap_f1(
                df["y_true"].values.astype(int),
                df["y_pred"].values.astype(int),
                n_boot=args.n_boot,
                rng=rng,
            )
            rows.append(
                {
                    "model": model,
                    "emotion": emotion,
                    "n_test": len(df),
                    "f1": point,
                    "f1_lo_95": lo,
                    "f1_hi_95": hi,
                }
            )
    ci_df = pd.DataFrame(rows)
    ci_path = out_dir / "05_bootstrap_ci.csv"
    ci_df.to_csv(ci_path, index=False)

    macro_rows = []
    for model in models:
        sub = ci_df[ci_df["model"] == model]
        if sub["f1"].dropna().empty:
            continue
        macro_rows.append(
            {
                "model": model,
                "n_emotions": int(sub["f1"].notna().sum()),
                "macro_f1": float(sub["f1"].mean(skipna=True)),
                "macro_f1_lo_95": float(sub["f1_lo_95"].mean(skipna=True)),
                "macro_f1_hi_95": float(sub["f1_hi_95"].mean(skipna=True)),
            }
        )
    macro_df = pd.DataFrame(macro_rows).sort_values("macro_f1", ascending=False)
    macro_path = out_dir / "05_macro_summary.csv"
    macro_df.to_csv(macro_path, index=False)

    delta_rows = []
    for emotion in emotions:
        present = [m for m in models if (m, emotion) in preds]
        for a, b in combinations(present, 2):
            d_a = preds[(a, emotion)]
            d_b = preds[(b, emotion)]
            point, lo, hi, sig = paired_bootstrap_delta(
                d_a["y_true"].values.astype(int),
                d_a["y_pred"].values.astype(int),
                d_b["y_true"].values.astype(int),
                d_b["y_pred"].values.astype(int),
                d_a["sentence_id"].values,
                d_b["sentence_id"].values,
                n_boot=args.n_boot,
                rng=rng,
            )
            delta_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "emotion": emotion,
                    "delta_f1": point,
                    "delta_lo_95": lo,
                    "delta_hi_95": hi,
                    "sig_95": sig,
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    delta_path = out_dir / "05_paired_deltas.csv"
    delta_df.to_csv(delta_path, index=False)

    print("Per-(model, emotion) F1 with 95 % CI:")
    pivot = ci_df.pivot(index="emotion", columns="model", values="f1").round(3)
    pivot = pivot.reindex([e for e in EMOTIONS if e in pivot.index])
    print(pivot.to_string())
    print()
    print("Macro summary (sorted):")
    print(macro_df.round(3).to_string(index=False))
    print()
    sig_n = int(delta_df["sig_95"].sum()) if not delta_df.empty else 0
    print(
        f"Paired deltas: {len(delta_df)} pairs computed; {sig_n} with 95 % CI excluding 0."
    )
    print()
    print(f"Wrote {ci_path}")
    print(f"Wrote {macro_path}")
    print(f"Wrote {delta_path}")


if __name__ == "__main__":
    main()
