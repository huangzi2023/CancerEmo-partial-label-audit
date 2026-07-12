#!/usr/bin/env python3
"""CPU proxy for the split-conflict leakage contrast experiment (Paper A,
contribution #5).

See ../../EXP_SPLIT_CONFLICT_LEAKAGE.md for the full design. This script
implements §5 (the CPU proxy):

  1. Contamination quantification: under the LEAK regime (each
     (sentence, emotion) cell keeps its ORIGINAL per-file split), how many
     test cells belong to a sentence whose text also appears in training?
  2. Shared-representation leakage demonstration: a per-cell logistic
     regression whose sentence-feature weights are SHARED across all eight
     emotions (input = [TF-IDF(sentence) ; emotion one-hot]). This mirrors
     the BERT shared encoder on CPU: training on text S for one emotion
     tunes weights that are reused when the same text S is tested under a
     different emotion. We train it under CLEAN vs LEAK splits and stratify
     test F1 into contaminated vs clean cells.

The headline number (contamination fraction) is model-free and exact; the
linear model only establishes that the contamination is *exploitable*. The
full magnitude of F1 inflation is expected to be larger under the learned
BERT encoder (the GPU arm, §4A of the design).

Run:  python scripts/split_conflict_leakage_proxy.py
Outputs land under 06_SplitConflictLeakage/outputs/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

EMOTIONS = [
    "Anger", "Anticipation", "Disgust", "Fear",
    "Joy", "Sadness", "Surprise", "Trust",
]
SEED = 42

# train uses original buckets 0 (train) + 1 (validation); test is bucket 2.
TRAIN_BUCKETS = {0, 1}
TEST_BUCKET = 2


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    import os
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    sibling = repo_root().parent / "data"
    if sibling.exists():
        return sibling
    return repo_root() / "data"


def sentence_id_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def hash_split(sid: str) -> int:
    """Deterministic 80/10/10 bucket — identical to data_audit_and_build.py."""
    bucket = int(sid, 16) % 10
    if bucket <= 7:
        return 0
    if bucket == 8:
        return 1
    return 2


def load_cells(raw_dir: Path) -> pd.DataFrame:
    """One row per observed (sentence, emotion) cell."""
    rows = []
    for emo in EMOTIONS:
        df = pd.read_csv(raw_dir / f"{emo}_anon.csv")
        for _, r in df.iterrows():
            s = r["Sentence"]
            rows.append(
                {
                    "sid": sentence_id_md5(s),
                    "sentence": s,
                    "emotion": emo,
                    "label": int(r[emo]),
                    "orig_split": int(r["Split"]),
                }
            )
    cells = pd.DataFrame(rows)
    cells["hash_split"] = cells["sid"].map(hash_split)
    return cells


def assign_regime(cells: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Add a `bucket` column (0/1/2) per the chosen split regime."""
    out = cells.copy()
    if regime == "leak":
        out["bucket"] = out["orig_split"]
    elif regime == "clean":
        out["bucket"] = out["hash_split"]
    else:
        raise ValueError(regime)
    out["is_train"] = out["bucket"].isin(TRAIN_BUCKETS)
    out["is_test"] = out["bucket"] == TEST_BUCKET
    return out


# ---------------------------------------------------------------------------
# 1. Contamination quantification
# ---------------------------------------------------------------------------

def contamination_report(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For LEAK vs CLEAN: fraction of test cells whose sentence is also in
    the training set (text leakage surface)."""
    per_emotion = []
    summary = {}
    for regime in ("leak", "clean"):
        r = assign_regime(cells, regime)
        train_sids = set(r.loc[r["is_train"], "sid"])
        test = r.loc[r["is_test"]].copy()
        test["contaminated"] = test["sid"].isin(train_sids)
        summary[regime] = {
            "n_test_cells": int(len(test)),
            "n_contaminated": int(test["contaminated"].sum()),
            "frac_contaminated": float(test["contaminated"].mean()) if len(test) else 0.0,
        }
        for emo in EMOTIONS:
            sub = test[test["emotion"] == emo]
            per_emotion.append(
                {
                    "regime": regime,
                    "emotion": emo,
                    "n_test_cells": int(len(sub)),
                    "n_contaminated": int(sub["contaminated"].sum()),
                    "frac_contaminated": float(sub["contaminated"].mean()) if len(sub) else 0.0,
                }
            )
    return pd.DataFrame(per_emotion), summary


# ---------------------------------------------------------------------------
# 2. Shared-representation linear model
# ---------------------------------------------------------------------------

def build_features(
    sentences: list[str],
    emotions: list[str],
    vectorizer: TfidfVectorizer,
    fit: bool,
) -> csr_matrix:
    """[TF-IDF(sentence) ; emotion one-hot]. Text columns are SHARED across
    all emotions, so the trained weights couple emotions exactly like a
    shared encoder."""
    if fit:
        tfidf = vectorizer.fit_transform(sentences)
    else:
        tfidf = vectorizer.transform(sentences)
    emo_idx = {e: i for i, e in enumerate(EMOTIONS)}
    onehot = np.zeros((len(emotions), len(EMOTIONS)), dtype=np.float32)
    for row, e in enumerate(emotions):
        onehot[row, emo_idx[e]] = 1.0
    return hstack([tfidf, csr_matrix(onehot)]).tocsr()


def run_model(cells: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, dict]:
    r = assign_regime(cells, regime)
    train = r[r["is_train"]].reset_index(drop=True)
    test = r[r["is_test"]].reset_index(drop=True)

    train_sids = set(train["sid"])
    test = test.assign(contaminated=test["sid"].isin(train_sids))

    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = build_features(train["sentence"].tolist(), train["emotion"].tolist(), vec, fit=True)
    Xte = build_features(test["sentence"].tolist(), test["emotion"].tolist(), vec, fit=False)

    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, train["label"].values)
    pred = clf.predict(Xte)
    test = test.assign(pred=pred)

    # per-emotion F1 + macro
    per_emotion = []
    for emo in EMOTIONS:
        sub = test[test["emotion"] == emo]
        if len(sub) == 0 or sub["label"].nunique() < 1:
            f1 = p = rec = float("nan")
        else:
            f1 = f1_score(sub["label"], sub["pred"], zero_division=0)
            p = precision_score(sub["label"], sub["pred"], zero_division=0)
            rec = recall_score(sub["label"], sub["pred"], zero_division=0)
        per_emotion.append({"regime": regime, "emotion": emo, "n": int(len(sub)), "f1": f1, "precision": p, "recall": rec})
    per_df = pd.DataFrame(per_emotion)
    macro_f1 = float(np.nanmean(per_df["f1"]))

    # stratified F1: contaminated vs clean test cells (pooled across emotions)
    def strat_f1(mask_name: str, mask: pd.Series) -> dict:
        sub = test[mask]
        return {
            "regime": regime,
            "stratum": mask_name,
            "n": int(len(sub)),
            "f1": float(f1_score(sub["label"], sub["pred"], zero_division=0)) if len(sub) else float("nan"),
            "accuracy": float((sub["label"] == sub["pred"]).mean()) if len(sub) else float("nan"),
        }

    strat = [
        strat_f1("all", pd.Series([True] * len(test))),
        strat_f1("contaminated", test["contaminated"]),
        strat_f1("clean", ~test["contaminated"]),
    ]
    return per_df, {"macro_f1": macro_f1, "strat": strat, "n_test": int(len(test))}


# ---------------------------------------------------------------------------
# 3. Controlled-injection (§4A) — isolates leakage from sentence difficulty
# ---------------------------------------------------------------------------

def controlled_injection(cells: pd.DataFrame) -> dict:
    """Hold the eval cells FIXED; the only thing that varies between the two
    models is whether the eval sentences' text was present in training.

    eval set   = LEAK test cells that are contaminated (sentence also has a
                 train cell under another emotion) -> the 1,309 cells.
    train_leak = all LEAK-train cells (eval sentences' OTHER-emotion cells
                 are present -> text leaks in).
    train_clean= train_leak with every cell of the eval sentences removed
                 (eval sentences' text fully absent from training).

    delta = F1(leak) - F1(clean) on the identical eval set is attributable
    ONLY to text leakage, because sentence difficulty and eval composition
    are held constant.
    """
    r = assign_regime(cells, "leak")
    train_all = r[r["is_train"]].reset_index(drop=True)
    test_all = r[r["is_test"]].reset_index(drop=True)
    train_sids = set(train_all["sid"])

    eval_cells = test_all[test_all["sid"].isin(train_sids)].reset_index(drop=True)
    eval_sids = set(eval_cells["sid"])

    train_leak = train_all
    train_clean = train_all[~train_all["sid"].isin(eval_sids)].reset_index(drop=True)

    def fit_eval(train_df: pd.DataFrame) -> tuple[float, np.ndarray]:
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
        Xtr = build_features(train_df["sentence"].tolist(), train_df["emotion"].tolist(), vec, fit=True)
        Xev = build_features(eval_cells["sentence"].tolist(), eval_cells["emotion"].tolist(), vec, fit=False)
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, train_df["label"].values)
        pred = clf.predict(Xev)
        f1 = f1_score(eval_cells["label"], pred, zero_division=0)
        return float(f1), pred

    f1_leak, _ = fit_eval(train_leak)
    f1_clean, _ = fit_eval(train_clean)
    return {
        "n_eval_cells": int(len(eval_cells)),
        "n_eval_sentences": int(len(eval_sids)),
        "train_leak_rows": int(len(train_leak)),
        "train_clean_rows": int(len(train_clean)),
        "f1_leak": f1_leak,
        "f1_clean": f1_clean,
        "delta_leak_minus_clean": f1_leak - f1_clean,
    }


def main() -> None:
    raw_dir = data_dir() / "raw"
    out_dir = repo_root() / "06_SplitConflictLeakage" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(raw_dir)
    n_collisions = cells.groupby("sid")["sentence"].nunique()
    assert (n_collisions == 1).all(), "md5[:12] sentence_id collision detected"

    print(f"Loaded {len(cells)} observed cells / {cells['sid'].nunique()} unique sentences")

    # --- 1. contamination ---
    contam_per_emo, contam_summary = contamination_report(cells)
    contam_per_emo.to_csv(out_dir / "06_contamination_counts.csv", index=False)
    print("\n=== Contamination surface (test cells whose sentence is also in train) ===")
    for regime, s in contam_summary.items():
        print(f"  {regime:5s}: {s['n_contaminated']:5d}/{s['n_test_cells']:5d} "
              f"= {s['frac_contaminated']*100:5.1f}% contaminated")

    # --- 2. shared-representation model: leak vs clean ---
    rows_macro = []
    rows_strat = []
    per_emo_all = []
    for regime in ("clean", "leak"):
        per_df, res = run_model(cells, regime)
        per_emo_all.append(per_df)
        rows_macro.append({"regime": regime, "macro_f1": res["macro_f1"], "n_test": res["n_test"]})
        rows_strat.extend(res["strat"])
        print(f"\n=== Shared-representation linear model — {regime.upper()} ===")
        print(f"  macro F1 = {res['macro_f1']:.4f}  (n_test={res['n_test']})")
        for st in res["strat"]:
            print(f"    {st['stratum']:12s} n={st['n']:5d}  F1={st['f1']:.4f}  acc={st['accuracy']:.4f}")

    macro_df = pd.DataFrame(rows_macro)
    strat_df = pd.DataFrame(rows_strat)
    per_emo_df = pd.concat(per_emo_all, ignore_index=True)

    # leak - clean deltas
    clean_macro = macro_df.loc[macro_df.regime == "clean", "macro_f1"].iloc[0]
    leak_macro = macro_df.loc[macro_df.regime == "leak", "macro_f1"].iloc[0]
    print(f"\n=== Headline ===")
    print(f"  macro F1  CLEAN={clean_macro:.4f}  LEAK={leak_macro:.4f}  "
          f"delta(leak-clean)={leak_macro-clean_macro:+.4f}")
    leak_strat = strat_df[strat_df.regime == "leak"].set_index("stratum")["f1"]
    if {"contaminated", "clean"}.issubset(leak_strat.index):
        print(f"  LEAK F1 on contaminated cells={leak_strat['contaminated']:.4f} "
              f"vs clean cells={leak_strat['clean']:.4f} "
              f"(gap={leak_strat['contaminated']-leak_strat['clean']:+.4f})")

    macro_df.to_csv(out_dir / "06_leak_vs_clean_macro.csv", index=False)
    strat_df.to_csv(out_dir / "06_contamination_stratified.csv", index=False)
    per_emo_df.to_csv(out_dir / "06_leak_vs_clean_per_emotion.csv", index=False)

    # --- 3. controlled injection: isolate leakage from sentence difficulty ---
    inj = controlled_injection(cells)
    print(f"\n=== Controlled injection (§4A) — eval cells FIXED, only training text toggled ===")
    print(f"  eval: {inj['n_eval_cells']} cells / {inj['n_eval_sentences']} sentences")
    print(f"  train rows: leak={inj['train_leak_rows']}  clean={inj['train_clean_rows']}")
    print(f"  F1  with-text(leak)={inj['f1_leak']:.4f}  without-text(clean)={inj['f1_clean']:.4f}  "
          f"delta={inj['delta_leak_minus_clean']:+.4f}")
    pd.DataFrame([inj]).to_csv(out_dir / "06_controlled_injection.csv", index=False)
    print(f"\nWrote 5 CSVs to {out_dir}")


if __name__ == "__main__":
    main()
