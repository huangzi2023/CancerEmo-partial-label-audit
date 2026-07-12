"""
data_audit_and_build.py

Audit the eight CancerEmo binary CSV files and build Datasets A, B,
and C. This single script runs four internal phases:

    1. cross-file audit (4 CSVs)
    2. Dataset A   (eight per-emotion binary CSVs, normalised columns)
    3. Dataset B   (outer-merged sentence-level weak multi-label table)
    4. Dataset C   (sentence-level partial-label table, NA-encoded)

Outputs land under ``$DATA_DIR/processed/`` (default: ``../data/processed/``).

Run from the repository root:

    python scripts/data_audit_and_build.py
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

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
SPLIT_NAME = {0: "train", 1: "validation", 2: "test"}


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
    """Deterministic 80/10/10 split bucket from md5 of sentence_id."""
    bucket = int(sid, 16) % 10
    if bucket <= 7:
        return 0
    if bucket == 8:
        return 1
    return 2


def load_one(emotion: str, raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"{emotion}_anon.csv")
    expected = {"Sentence", emotion, "Split"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{emotion}: missing columns {missing}")
    df = df.rename(columns={emotion: "label"})
    df["emotion"] = emotion
    return df[["emotion", "Sentence", "label", "Split"]]


def phase_audit(raw_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Phase 1 — produce the four audit CSVs."""
    print("[1/4] cross-file audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_emotion = {e: load_one(e, raw_dir) for e in EMOTIONS}

    rows = []
    for emotion in EMOTIONS:
        df = per_emotion[emotion]
        rows.append(
            {
                "emotion": emotion,
                "total": len(df),
                "positive": int((df["label"] == 1).sum()),
                "negative": int((df["label"] == 0).sum()),
                "train": int((df["Split"] == 0).sum()),
                "validation": int((df["Split"] == 1).sum()),
                "test": int((df["Split"] == 2).sum()),
            }
        )
    binary_df = pd.DataFrame(rows)
    binary_df.to_csv(out_dir / "audit_binary_files.csv", index=False)

    sent_to_emotions: dict[str, list[str]] = defaultdict(list)
    sent_to_positive: dict[str, list[str]] = defaultdict(list)
    sent_to_splits: dict[str, dict[str, int]] = defaultdict(dict)
    for emotion in EMOTIONS:
        df = per_emotion[emotion]
        for _, row in df.iterrows():
            s = row["Sentence"]
            sent_to_emotions[s].append(emotion)
            sent_to_splits[s][emotion] = int(row["Split"])
            if row["label"] == 1:
                sent_to_positive[s].append(emotion)

    overlap_rows = []
    for s, emos in sent_to_emotions.items():
        pos_emos = sent_to_positive.get(s, [])
        overlap_rows.append(
            {
                "sentence": s,
                "n_source_files": len(emos),
                "source_emotions": ";".join(emos),
                "n_observed_positive": len(pos_emos),
                "positive_emotions": ";".join(pos_emos),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(out_dir / "sentence_overlap_audit.csv", index=False)

    conflict_rows = []
    for s, splits in sent_to_splits.items():
        unique_splits = set(splits.values())
        if len(unique_splits) > 1:
            conflict_rows.append(
                {
                    "sentence": s,
                    "n_files": len(splits),
                    "splits_observed": ";".join(
                        f"{e}={SPLIT_NAME[v]}" for e, v in splits.items()
                    ),
                    "n_unique_splits": len(unique_splits),
                }
            )
    pd.DataFrame(conflict_rows).to_csv(out_dir / "split_conflict_report.csv", index=False)

    matrix = pd.DataFrame(0, index=EMOTIONS, columns=EMOTIONS, dtype=int)
    for _, row in overlap_df.iterrows():
        if row["n_observed_positive"] < 1:
            continue
        emos = row["positive_emotions"].split(";")
        for a in emos:
            for b in emos:
                matrix.at[a, b] += 1
    matrix.to_csv(out_dir / "emotion_cooccurrence_observed.csv")

    total = sum(len(per_emotion[e]) for e in EMOTIONS)
    print(
        f"      total rows = {total}  unique sentences = {len(overlap_df)}  "
        f"multi-file = {(overlap_df['n_source_files'] >= 2).sum()}  "
        f"multi-positive = {(overlap_df['n_observed_positive'] >= 2).sum()}  "
        f"split conflicts = {len(conflict_rows)}"
    )
    return per_emotion


def phase_dataset_A(per_emotion: dict[str, pd.DataFrame], out_dir: Path) -> None:
    print("[2/4] Dataset A — original 8 per-emotion binary CSVs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for emotion, df in per_emotion.items():
        normalised = df.rename(columns={"Sentence": "sentence", "Split": "split"})
        normalised = normalised[["sentence", "label", "split"]]
        normalised.to_csv(out_dir / f"{emotion}.csv", index=False)
    print(f"      wrote 8 normalised binary CSVs to {out_dir}")


def phase_dataset_B(per_emotion: dict[str, pd.DataFrame], out_dir: Path) -> None:
    print("[3/4] Dataset B — outer-merged weak multi-label table")
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences: list[str] = []
    seen: set[str] = set()
    positive_per_sent: dict[str, set[str]] = defaultdict(set)
    observed_per_sent: dict[str, set[str]] = defaultdict(set)
    splits_per_sent: dict[str, dict[str, int]] = defaultdict(dict)

    for emotion, df in per_emotion.items():
        for _, row in df.iterrows():
            s = row["Sentence"]
            if s not in seen:
                sentences.append(s)
                seen.add(s)
            observed_per_sent[s].add(emotion)
            splits_per_sent[s][emotion] = int(row["Split"])
            if int(row["label"]) == 1:
                positive_per_sent[s].add(emotion)

    rows = []
    for s in sentences:
        sid = sentence_id_md5(s)
        observed = observed_per_sent[s]
        positives = positive_per_sent[s]
        splits_map = splits_per_sent[s]
        unique_splits = set(splits_map.values())
        row = {"sentence_id": sid, "sentence": s}
        for emo, low in zip(EMOTIONS, LOWER_EMOS):
            row[low] = 1 if emo in positives else 0
        for emo, low in zip(EMOTIONS, LOWER_EMOS):
            row[f"mask_{low}"] = 1 if emo in observed else 0
        row["source_files"] = ";".join(sorted(observed))
        row["original_splits"] = ";".join(
            f"{emo}={SPLIT_NAME[splits_map[emo]]}" for emo in sorted(observed)
        )
        row["has_split_conflict"] = len(unique_splits) > 1
        row["n_source_files"] = len(observed)
        row["n_positive_emotions"] = len(positives)
        row["positive_emotion_set"] = ";".join(sorted(positives))
        rows.append(row)

    merged = pd.DataFrame(rows)
    merged.to_csv(out_dir / "merged_weak_multilabel.csv", index=False)

    splits = pd.DataFrame(
        {
            "sentence_id": merged["sentence_id"],
            "sentence": merged["sentence"],
            "split": [hash_split(sid) for sid in merged["sentence_id"]],
        }
    )
    splits["split_name"] = splits["split"].map(SPLIT_NAME)
    splits.to_csv(out_dir / "sentence_splits.csv", index=False)

    train = int((splits["split"] == 0).sum())
    val = int((splits["split"] == 1).sum())
    test = int((splits["split"] == 2).sum())
    print(
        f"      {len(merged)} unique sentences  "
        f"any-positive = {(merged['n_positive_emotions'] > 0).sum()}  "
        f"multi-emotion = {(merged['n_positive_emotions'] >= 2).sum()}  "
        f"split = {train}/{val}/{test}"
    )


def phase_dataset_C(per_emotion: dict[str, pd.DataFrame], out_dir: Path) -> None:
    print("[4/4] Dataset C — partial-label table (1 / 0 / NA)")
    out_dir.mkdir(parents=True, exist_ok=True)

    sentence_to_observed: dict[str, dict[str, int]] = {}
    for emotion, df in per_emotion.items():
        for _, row in df.iterrows():
            s = row["Sentence"]
            obs = sentence_to_observed.setdefault(s, {})
            obs[emotion] = int(row["label"])

    rows = []
    for s, observed in sentence_to_observed.items():
        row = {"sentence_id": sentence_id_md5(s), "sentence": s}
        for emo, low in zip(EMOTIONS, LOWER_EMOS):
            row[low] = observed.get(emo)
        row["n_observed_labels"] = len(observed)
        row["n_observed_positive"] = sum(1 for v in observed.values() if v == 1)
        row["n_observed_negative"] = sum(1 for v in observed.values() if v == 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    for low in LOWER_EMOS:
        df[low] = df[low].astype("Int64")
    df.to_csv(out_dir / "merged_partial_label.csv", index=False)
    print(
        f"      {len(df)} unique sentences  "
        f"max emotions observed per sentence = {int(df['n_observed_labels'].max())}"
    )


def main() -> None:
    root_data = data_dir()
    raw_dir = root_data / "raw"
    proc = root_data / "processed"

    per_emotion = phase_audit(raw_dir, proc / "audit")
    phase_dataset_A(per_emotion, proc / "dataset_A_binary")
    phase_dataset_B(per_emotion, proc / "dataset_B_weak_multilabel")
    phase_dataset_C(per_emotion, proc / "dataset_C_partial_label")

    print(f"\nAll outputs under {proc}")


if __name__ == "__main__":
    main()
