"""
binary_baseline_replication.py

Reproduce the per-emotion binary baselines from Sosea & Caragea (2020)
Table 4 on Dataset A. This single script runs two internal phases:

    1. classical Tier-1 (Majority, TF-IDF + LR, TF-IDF + NB) — CPU
    2. transformer baseline (BERT / DistilBERT) — GPU recommended

Both phases save per-emotion metrics and per-sentence test predictions
to ``02_BinaryBaselineReplication/outputs/`` for downstream paired
bootstrap in ``evaluation_bootstrap.py``.

Usage:
    # classical only (skip the GPU phase)
    python scripts/binary_baseline_replication.py --skip-bert

    # full pipeline targeting the original BERT macro F1 ≈ 0.71
    python scripts/binary_baseline_replication.py \
        --bert-model bert-base-uncased --epochs 3
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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


def data_dir() -> Path:
    explicit = os.environ.get("DATA_DIR")
    if explicit:
        return Path(explicit)
    nested = repo_root().parent / "data"
    if nested.exists():
        return nested
    return repo_root() / "data"


def sentence_id_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def hash_split(sid: str) -> int:
    """Deterministic 80/10/10 bucket — identical to data_audit_and_build.py,
    so the 'hash' split-source matches Datasets B/C exactly."""
    bucket = int(sid, 16) % 10
    if bucket <= 7:
        return 0
    if bucket == 8:
        return 1
    return 2


def split_dataset_a(df: "pd.DataFrame", split_source: str):
    """Return (train, test) for one emotion's Dataset-A file.

    split_source == "original": use the released per-file Split column
        (train=0, test=2) — reproduces the original benchmark.
    split_source == "hash": re-partition by sentence-hash bucket (train=0,
        test=2), identical to the Datasets B/C sentence-level split, so the
        binary baseline becomes directly comparable to the masked MTL.
    """
    if split_source == "hash":
        bucket = df["sentence"].astype(str).map(
            lambda s: hash_split(sentence_id_md5(s))
        )
        return df[bucket == 0], df[bucket == 2]
    return df[df["split"] == 0], df[df["split"] == 2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_auroc(y_true: np.ndarray, score: np.ndarray | None) -> float:
    if score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def build_classical_models() -> dict[str, object]:
    tfidf = lambda: TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True
    )
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tfidf_lr": Pipeline(
            [("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear"))]
        ),
        "tfidf_nb": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=1.0))]),
    }


def score_classical(model_name, model, X_train, y_train, X_test, y_test):
    if model_name == "majority":
        model.fit(np.zeros((len(X_train), 1)), y_train)
        y_pred = model.predict(np.zeros((len(X_test), 1)))
        prob = np.full(len(X_test), float(np.mean(y_train)))
        auroc = float("nan")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            prob = y_pred.astype(float)
        auroc = safe_auroc(y_test, prob)
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": auroc,
    }
    return metrics, y_pred, prob


def phase_classical(out_dir: Path, split_source: str = "original") -> None:
    print(f"[1/2] Tier-1 classical baselines (CPU)  split-source={split_source}")
    dataset_a = data_dir() / "processed" / "dataset_A_binary"
    preds_dir = out_dir / "02_classical_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for emotion in EMOTIONS:
        df = pd.read_csv(dataset_a / f"{emotion}.csv")
        train, test = split_dataset_a(df, split_source)
        X_train = train["sentence"].astype(str).values
        y_train = train["label"].values.astype(int)
        X_test = test["sentence"].astype(str).values
        y_test = test["label"].values.astype(int)
        test_ids = [sentence_id_md5(s) for s in X_test]

        for model_name, model in build_classical_models().items():
            metrics, y_pred, y_prob = score_classical(
                model_name, model, X_train, y_train, X_test, y_test
            )
            rows.append(
                {
                    "model": model_name,
                    "emotion": emotion,
                    "n_train": len(train),
                    "n_test": len(test),
                    "test_prevalence": float(np.mean(y_test)),
                    **metrics,
                }
            )
            pd.DataFrame(
                {
                    "sentence_id": test_ids,
                    "sentence": X_test,
                    "y_true": y_test.astype(int),
                    "y_pred": y_pred.astype(int),
                    "prob_pos": y_prob.astype(float),
                }
            ).to_csv(preds_dir / f"{model_name}__{emotion}.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "02_baseline_metrics.csv", index=False)
    macro = (
        metrics_df.groupby("model")[["precision", "recall", "f1", "auroc"]]
        .mean()
        .reset_index()
        .rename(columns={c: f"macro_{c}" for c in ["precision", "recall", "f1", "auroc"]})
    )
    macro.to_csv(out_dir / "02_macro_average_f1.csv", index=False)
    print(macro.round(3).to_string(index=False))


def phase_bert(out_dir: Path, args: argparse.Namespace) -> None:
    print(f"[2/2] Transformer baseline ({args.bert_model})  split-source={args.split_source}")
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_a = data_dir() / "processed" / "dataset_A_binary"
    preds_dir = out_dir / "02_bert_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"      device = {device}")

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.encodings = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    rows = []
    for emotion in args.emotions:
        df = pd.read_csv(dataset_a / f"{emotion}.csv")
        train, test = split_dataset_a(df, args.split_source)
        if train.empty or test.empty:
            print(f"      {emotion}: empty train or test; skipping")
            continue
        print(f"      {emotion}: train={len(train)}  test={len(test)}")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.bert_model, num_labels=2
        ).to(device)
        train_set = TextDataset(
            train["sentence"].astype(str).tolist(),
            train["label"].astype(int).tolist(),
            tokenizer, args.max_length,
        )
        test_set = TextDataset(
            test["sentence"].astype(str).tolist(),
            test["label"].astype(int).tolist(),
            tokenizer, args.max_length,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        steps = max(1, len(train_loader) * args.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * steps), num_training_steps=steps
        )
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += outputs.loss.item()
            print(f"        epoch {epoch + 1}/{args.epochs}  loss={running / max(1, len(train_loader)):.4f}")
        model.eval()
        probs_all = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs_all.extend(
                    torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy().tolist()
                )
        probs_arr = np.array(probs_all)
        preds = (probs_arr >= 0.5).astype(int)
        y_true = np.array(test["label"].astype(int).tolist())
        sentences = test["sentence"].astype(str).tolist()
        metrics = {
            "model": Path(args.bert_model).name,
            "emotion": emotion,
            "n_train": len(train),
            "n_test": len(test),
            "test_prevalence": float(y_true.mean()),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "auroc": float(roc_auc_score(y_true, probs_arr))
            if len(np.unique(y_true)) > 1 else float("nan"),
        }
        rows.append(metrics)
        pd.DataFrame(
            {
                "sentence_id": [sentence_id_md5(s) for s in sentences],
                "sentence": sentences,
                "y_true": y_true,
                "y_pred": preds,
                "prob_pos": probs_arr,
            }
        ).to_csv(preds_dir / f"{emotion}.csv", index=False)
        print(
            f"        → F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  AUROC={metrics['auroc']:.3f}"
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "02_bert_metrics.csv", index=False)
    if not metrics_df.empty:
        macro = metrics_df[["precision", "recall", "f1", "auroc"]].mean()
        print("      macro:", macro.round(3).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bert", action="store_true",
                    help="Run only the classical Tier-1 phase")
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--bert-model", default="distilbert-base-uncased")
    ap.add_argument("--emotions", nargs="+", default=EMOTIONS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-source", choices=["original", "hash"], default="original",
                    help="'original' = released per-file splits (benchmark); "
                         "'hash' = sentence-level split identical to Datasets B/C "
                         "(clean-split comparator for the parity claim)")
    args = ap.parse_args()

    out_dir = repo_root() / "02_BinaryBaselineReplication" / "outputs"

    if not args.skip_classical:
        phase_classical(out_dir, args.split_source)
    if not args.skip_bert:
        phase_bert(out_dir, args)

    print()
    print("Reference (Sosea & Caragea 2020 Table 4):")
    print("  EmoLex lexicon ≈ 0.42 | NB / LR ≈ 0.60-0.61 | BERT ≈ 0.71")


if __name__ == "__main__":
    main()
