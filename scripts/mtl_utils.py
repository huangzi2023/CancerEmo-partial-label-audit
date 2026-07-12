"""
mtl_utils.py

Shared building blocks for the two shared-encoder multi-task learning
runs in `mtl_weak_zero.py` (weak-zero BCE on Dataset B) and
`mtl_partial_masked.py` (masked BCE on Dataset C). Both runs use the
same encoder + 8-way classification head; they differ only in how the
loss treats unobserved labels.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

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
LOWER_EMOS = [e.lower() for e in EMOTIONS]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Resolve the data directory.

    Priority:
        1. ``DATA_DIR`` environment variable, if set.
        2. ``../data`` relative to the repo root (the default nested layout
           used inside the author's writing workspace).
        3. ``data`` inside the repo (standalone clone fallback).
    """
    explicit = os.environ.get("DATA_DIR")
    if explicit:
        return Path(explicit)
    nested = repo_root().parent / "data"
    if nested.exists():
        return nested
    return repo_root() / "data"


def sentence_id_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MultiLabelDataset(Dataset):
    """Tokenises sentences once; serves (input_ids, attention_mask,
    labels_8d, mask_8d). For weak-zero training mask is all 1 so it can
    be safely ignored. For masked training mask encodes which of the
    eight emotion labels were observed."""

    def __init__(
        self,
        sentences: list[str],
        labels: np.ndarray,
        mask: np.ndarray,
        tokenizer,
        max_length: int = 128,
    ) -> None:
        assert labels.shape == mask.shape == (len(sentences), 8)
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
            "mask": self.mask[idx],
        }


class SharedEncoderMTL(torch.nn.Module):
    """Pretrained encoder pooled to [CLS] → dropout → Linear(d, 8)."""

    def __init__(self, model_name: str, num_labels: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


def load_dataset_B(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged weak-multilabel table + sentence-level splits."""
    b_dir = data_dir() / "processed" / "dataset_B_weak_multilabel"
    merged = pd.read_csv(b_dir / "merged_weak_multilabel.csv")
    splits = pd.read_csv(b_dir / "sentence_splits.csv")
    return merged, splits


def load_dataset_C(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset C reuses Dataset B's sentence_splits.csv."""
    c_dir = data_dir() / "processed" / "dataset_C_partial_label"
    b_dir = data_dir() / "processed" / "dataset_B_weak_multilabel"
    merged = pd.read_csv(c_dir / "merged_partial_label.csv")
    splits = pd.read_csv(b_dir / "sentence_splits.csv")
    return merged, splits


def build_split(
    merged: pd.DataFrame,
    splits: pd.DataFrame,
    split_value: int,
    mode: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return sentences, labels (n × 8), mask (n × 8) for the requested
    split. mode == "weak_zero" treats all 0s as negatives (mask is all
    1). mode == "masked" treats NA as unobserved (mask=0 there)."""
    assert mode in {"weak_zero", "masked"}
    df = merged.merge(splits[["sentence_id", "split"]], on="sentence_id")
    df = df[df["split"] == split_value].reset_index(drop=True)
    sentences = df["sentence"].astype(str).tolist()
    if mode == "weak_zero":
        labels = df[LOWER_EMOS].fillna(0).astype(int).values.astype("float32")
        mask = np.ones_like(labels, dtype="float32")
    else:  # masked
        raw = df[LOWER_EMOS]
        mask = raw.notna().astype("float32").values
        labels = raw.fillna(0).astype(int).values.astype("float32")
    return sentences, labels, mask


def per_emotion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-emotion precision / recall / F1 / AUROC. If mask is provided,
    only score on observed cells per emotion. mask shape == y_true."""
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rows = []
    for j, emotion in enumerate(EMOTIONS):
        col_mask = (mask[:, j] > 0) if mask is not None else np.ones(len(y_true), dtype=bool)
        yt = y_true[col_mask, j].astype(int)
        yp = y_pred[col_mask, j].astype(int)
        pp = y_prob[col_mask, j]
        if len(yt) == 0:
            rows.append({"emotion": emotion, "n": 0, "precision": float("nan"),
                          "recall": float("nan"), "f1": float("nan"), "auroc": float("nan")})
            continue
        auroc = float("nan")
        if len(np.unique(yt)) > 1:
            auroc = float(roc_auc_score(yt, pp))
        rows.append({
            "emotion": emotion,
            "n": int(len(yt)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "auroc": auroc,
        })
    return pd.DataFrame(rows)
