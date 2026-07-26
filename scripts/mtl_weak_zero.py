"""
mtl_weak_zero.py

Shared-encoder multi-task learning on Dataset B, using weak-zero BCE
(unobserved emotion labels are treated as 0). Implements research plan
§3.4 Experiment 2.

Architecture: pretrained encoder → [CLS] pooling → dropout →
Linear(hidden, 8). Loss: BCEWithLogitsLoss over all 8 emotions for
every sentence in the train split. This is the "naive multi-task"
baseline whose comparison against mtl_partial_masked.py's masked
variant tests whether the weak-zero assumption biases the model.

Outputs:
    03_WeakMultiLabelMTL/outputs/03_mtl_weak_metrics.csv
    03_WeakMultiLabelMTL/outputs/03_mtl_weak_predictions.csv

Default model is distilbert-base-uncased for CPU smoke-tests; switch
to bert-base-uncased on GPU for the publication target.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from mtl_utils import (
    EMOTIONS,
    LOWER_EMOS,
    MultiLabelDataset,
    SharedEncoderMTL,
    best_device,
    build_split,
    load_dataset_B,
    per_emotion_metrics,
    repo_root,
    sentence_id_md5,
    set_seed,
)


def train_one_epoch(
    model: SharedEncoderMTL,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> float:
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
    model.train()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(1, len(loader))


def predict(
    model: SharedEncoderMTL, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())
    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = np.concatenate(all_labels, axis=0).astype(int)
    masks = np.concatenate(all_masks, axis=0)
    return preds, probs, labels, masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weighted", action="store_true",
                    help="Apply per-emotion positive-class weighting "
                         "(pos_weight = n_negative / n_positive on the train "
                         "split) in the weak-zero BCE. Tests whether the "
                         "Tier 2 -> Tier 3 gap is class-imbalance driven. "
                         "Default off = original weak-zero baseline.")
    args = ap.parse_args()

    set_seed(args.seed)
    root = repo_root()
    out_dir = root / "03_WeakMultiLabelMTL" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = best_device()
    tag = "weighted" if args.class_weighted else "weak_zero"
    print(f"Device: {device}  |  Model: {args.model}  |  variant: {tag}")

    merged, splits = load_dataset_B(root)
    train_s, train_y, train_m = build_split(merged, splits, 0, "weak_zero")
    test_s, test_y, test_m = build_split(merged, splits, 2, "weak_zero")
    print(f"Train: {len(train_s)}  Test: {len(test_s)}")
    n_pos = train_y.sum(axis=0)
    n_neg = train_y.shape[0] - n_pos
    print(
        "Per-emotion train positives / negatives: "
        + ", ".join(f"{e}={int(n_pos[j])}/{int(n_neg[j])}" for j, e in enumerate(EMOTIONS))
    )

    pos_weight = None
    if args.class_weighted:
        pw = (n_neg / np.clip(n_pos, 1, None)).astype("float32")
        pos_weight = torch.tensor(pw, device=device)
        print("pos_weight (n_neg/n_pos): "
              + ", ".join(f"{e}={pw[j]:.2f}" for j, e in enumerate(EMOTIONS)))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SharedEncoderMTL(args.model, num_labels=8).to(device)

    train_set = MultiLabelDataset(train_s, train_y, train_m, tokenizer, args.max_length)
    test_set = MultiLabelDataset(test_s, test_y, test_m, tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                               pos_weight=pos_weight)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={loss:.4f}")

    preds, probs, labels, _ = predict(model, test_loader, device)

    metrics_df = per_emotion_metrics(labels, preds, probs, mask=None)
    macro = metrics_df[["precision", "recall", "f1", "auroc"]].mean()
    micro_f1 = (
        2 * (preds * labels).sum() / max(1, (preds.sum() + labels.sum()))
    )
    label_stub = "mtl_weak_weighted_" if args.class_weighted else "mtl_weak_zero_"
    metrics_df["model"] = label_stub + args.model.split("/")[-1]

    file_stub = "03_mtl_weak_weighted" if args.class_weighted else "03_mtl_weak"
    metrics_path = out_dir / f"{file_stub}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    test_ids = [sentence_id_md5(s) for s in test_s]
    pred_records = []
    for i, sid in enumerate(test_ids):
        for j, low in enumerate(LOWER_EMOS):
            pred_records.append(
                {
                    "sentence_id": sid,
                    "emotion": EMOTIONS[j],
                    "y_true": int(labels[i, j]),
                    "y_pred": int(preds[i, j]),
                    "prob_pos": float(probs[i, j]),
                }
            )
    preds_df = pd.DataFrame(pred_records)
    preds_path = out_dir / f"{file_stub}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    print()
    print("Per-emotion (eval on ALL test cells, including weak-zeros):")
    print(metrics_df[["emotion", "n", "precision", "recall", "f1", "auroc"]].to_string(index=False))
    print()
    print("Macro averages:")
    print(macro.round(3).to_string())
    print(f"Micro F1 (across all label cells): {micro_f1:.3f}")
    print()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {preds_path}")


if __name__ == "__main__":
    main()
