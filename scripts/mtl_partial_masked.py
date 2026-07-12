"""
mtl_partial_masked.py

Shared-encoder multi-task learning on Dataset C with MASKED partial-
label BCE. Implements research plan §3.4 Experiment 3.

Same architecture as mtl_weak_zero.py (encoder + 8-way head). The
only difference is the loss: BCEWithLogitsLoss with reduction="none",
multiplied element-wise by the observation mask, then normalised by
the number of observed cells per batch. This means each emotion head
is updated ONLY by sentences that actually appeared in that emotion's
binary source file.

Evaluation is also masked: per-emotion metrics use only the test cells
where the emotion was observed. The head-to-head comparison with
mtl_weak_zero isolates the bias introduced by treating unobserved
labels as negatives.

Outputs:
    04_PartialLabelMaskedMTL/outputs/04_mtl_masked_metrics.csv
    04_PartialLabelMaskedMTL/outputs/04_mtl_masked_predictions.csv
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
    load_dataset_C,
    per_emotion_metrics,
    repo_root,
    sentence_id_md5,
    set_seed,
)


def masked_bce(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    masked = per_cell * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def train_one_epoch(
    model: SharedEncoderMTL,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = masked_bce(logits, labels, mask)
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
    args = ap.parse_args()

    set_seed(args.seed)
    root = repo_root()
    out_dir = root / "04_PartialLabelMaskedMTL" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = best_device()
    print(f"Device: {device}  |  Model: {args.model}")

    merged, splits = load_dataset_C(root)
    train_s, train_y, train_m = build_split(merged, splits, 0, "masked")
    test_s, test_y, test_m = build_split(merged, splits, 2, "masked")
    print(f"Train: {len(train_s)}  Test: {len(test_s)}")
    print(
        "Per-emotion train observed positives / observed: "
        + ", ".join(
            f"{e}={int((train_y[:, j] * train_m[:, j]).sum())}/{int(train_m[:, j].sum())}"
            for j, e in enumerate(EMOTIONS)
        )
    )

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
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"epoch {epoch + 1}/{args.epochs}  masked_train_loss={loss:.4f}")

    preds, probs, labels, masks = predict(model, test_loader, device)

    metrics_df = per_emotion_metrics(labels, preds, probs, mask=masks)
    macro = metrics_df[["precision", "recall", "f1", "auroc"]].mean(skipna=True)
    metrics_df["model"] = "mtl_masked_" + args.model.split("/")[-1]

    metrics_path = out_dir / "04_mtl_masked_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    test_ids = [sentence_id_md5(s) for s in test_s]
    pred_records = []
    for i, sid in enumerate(test_ids):
        for j, low in enumerate(LOWER_EMOS):
            if masks[i, j] == 0:
                continue
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
    preds_path = out_dir / "04_mtl_masked_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    print()
    print("Per-emotion (eval on OBSERVED test cells only):")
    print(metrics_df[["emotion", "n", "precision", "recall", "f1", "auroc"]].to_string(index=False))
    print()
    print("Macro averages (skipping emotions with no observed test cells):")
    print(macro.round(3).to_string())
    print()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {preds_path}")


if __name__ == "__main__":
    main()
