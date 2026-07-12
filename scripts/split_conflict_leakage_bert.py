#!/usr/bin/env python3
"""BERT controlled-injection arm for the split-conflict leakage experiment
(Paper A, contribution #5; design §4A).

The CPU proxy (split_conflict_leakage_proxy.py) established that the
contamination surface is large (65.6 % of naive-split test cells) but that
a *shared linear* model does not gain from it — adding the conflict
sentences' cross-emotion cells to training corrupts the shared weights
(controlled-injection delta = -18.8 pp). The open question is whether a
high-capacity, memorizing BERT encoder behaves differently. This script
answers it with the SAME controlled-injection design and the SAME eval
cells, swapping the linear model for the shared-encoder MTL used in Tier 3.

Design (identical to the CPU proxy §4A):
  eval set    = LEAK test cells that are contaminated (sentence also has a
                train cell under another emotion)  -> 1,309 cells / 1,228 sents.
  train_leak  = all LEAK-train cells (eval sentences' OTHER-emotion cells
                present -> text leaks into training).
  train_clean = train_leak with every cell of the eval sentences removed
                (eval sentences' text fully absent from training).
  Both models are scored on the IDENTICAL eval set; the macro/pooled F1
  delta is attributable ONLY to the presence of the eval sentences' text
  in training (difficulty and eval composition are held constant).

Everything model-side (encoder, masked BCE, optimizer, schedule, metrics)
is reused verbatim from mtl_utils / mtl_partial_masked, so the only moving
part versus Tier 3 is the train/eval construction.

Run on a GPU (see COLAB_RUNBOOK.md Cell 8b):
    python scripts/split_conflict_leakage_bert.py \
        --model bert-base-uncased --epochs 3 --batch-size 32 \
        --max-length 128 --lr 2e-5 --seeds 42 1 2 --n-boot 1000

Use --dry-run to validate the data construction with no training (CPU-safe).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mtl_utils import (
    EMOTIONS,
    LOWER_EMOS,
    repo_root,
    sentence_id_md5,
    set_seed,
)
# Reuse the EXACT cell table + regime logic from the CPU proxy so the eval
# set is byte-identical across the linear and BERT arms.
from split_conflict_leakage_proxy import (
    TRAIN_BUCKETS,
    TEST_BUCKET,
    data_dir,
    load_cells,
)

EMO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


# ---------------------------------------------------------------------------
# Data construction (CPU-safe; this is the only thing that differs from Tier 3)
# ---------------------------------------------------------------------------

def _arrays_from_groups(groups: dict[str, dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """groups: sid -> {"sentence": str, "cells": [(emotion, label), ...]}.
    Returns sentences, labels (n×8), mask (n×8) in EMOTIONS order. mask=1 only
    on the cells listed for that sentence."""
    sentences, labels, masks = [], [], []
    for sid, g in groups.items():
        y = np.zeros(len(EMOTIONS), dtype="float32")
        m = np.zeros(len(EMOTIONS), dtype="float32")
        for emo, lab in g["cells"]:
            j = EMO_IDX[emo]
            y[j] = float(lab)
            m[j] = 1.0
        sentences.append(g["sentence"])
        labels.append(y)
        masks.append(m)
    return sentences, np.vstack(labels), np.vstack(masks)


def build_injection_sets(cells: pd.DataFrame):
    """Return (eval, train_leak, train_clean) each as (sentences, labels, mask),
    plus a flat eval-cell index for paired bootstrap."""
    is_train = cells["orig_split"].isin(TRAIN_BUCKETS)
    is_test = cells["orig_split"] == TEST_BUCKET
    train_sids = set(cells.loc[is_train, "sid"])

    eval_rows = cells[is_test & cells["sid"].isin(train_sids)]
    eval_sids = set(eval_rows["sid"])
    train_rows = cells[is_train]

    def group(rows: pd.DataFrame) -> dict:
        g: dict[str, dict] = {}
        for _, r in rows.iterrows():
            d = g.setdefault(r["sid"], {"sentence": r["sentence"], "cells": []})
            d["cells"].append((r["emotion"], int(r["label"])))
        return g

    eval_g = group(eval_rows)
    leak_g = group(train_rows)
    clean_g = {sid: v for sid, v in leak_g.items() if sid not in eval_sids}

    stats = {
        "n_eval_cells": int(len(eval_rows)),
        "n_eval_sentences": int(len(eval_sids)),
        "n_train_leak_sents": int(len(leak_g)),
        "n_train_clean_sents": int(len(clean_g)),
    }
    return (
        _arrays_from_groups(eval_g),
        _arrays_from_groups(leak_g),
        _arrays_from_groups(clean_g),
        stats,
    )


# ---------------------------------------------------------------------------
# Metrics + paired bootstrap on the fixed eval set
# ---------------------------------------------------------------------------

def pooled_and_macro_f1(labels: np.ndarray, preds: np.ndarray, mask: np.ndarray):
    obs = mask.astype(bool)
    yt = labels[obs].astype(int)
    yp = preds[obs].astype(int)
    from sklearn.metrics import f1_score
    pooled = float(f1_score(yt, yp, zero_division=0))
    per = []
    for j, e in enumerate(EMOTIONS):
        m = obs[:, j]
        if m.sum() == 0:
            per.append(np.nan)
        else:
            per.append(f1_score(labels[m, j].astype(int), preds[m, j].astype(int), zero_division=0))
    return pooled, float(np.nanmean(per)), per


def paired_bootstrap(labels, mask, preds_leak, preds_clean, n_boot, seed):
    """Resample eval SENTENCES with replacement; recompute pooled-F1 delta."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = mask[idx].astype(bool)
        yt = labels[idx]
        fl = f1_score(yt[m].astype(int), preds_leak[idx][m].astype(int), zero_division=0)
        fc = f1_score(yt[m].astype(int), preds_clean[idx][m].astype(int), zero_division=0)
        deltas.append(fl - fc)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data construction only; no model, no GPU")
    args = ap.parse_args()

    root = repo_root()
    out_dir = root / "06_SplitConflictLeakage" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(data_dir() / "raw")
    (eval_s, eval_y, eval_m), (leak_s, leak_y, leak_m), (clean_s, clean_y, clean_m), stats = \
        build_injection_sets(cells)

    print("Construction:", stats)
    # invariant checks
    eval_sid_set = {sentence_id_md5(s) for s in eval_s}
    clean_sid_set = {sentence_id_md5(s) for s in clean_s}
    leak_sid_set = {sentence_id_md5(s) for s in leak_s}
    assert eval_sid_set.isdisjoint(clean_sid_set), "clean train must NOT contain eval sentences"
    assert eval_sid_set.issubset(leak_sid_set), "leak train MUST contain eval sentences"
    assert int(eval_m.sum()) == stats["n_eval_cells"], "eval mask cell count mismatch"
    print(f"Invariants OK. eval cells={int(eval_m.sum())} "
          f"(expect 1309), eval sents={len(eval_s)} (expect 1228)")

    if args.dry_run:
        print("Dry run complete — data construction valid, no training performed.")
        return

    # ----- GPU path: reuse the Tier-3 model + masked loss verbatim -----
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from mtl_utils import MultiLabelDataset, SharedEncoderMTL, best_device
    from mtl_partial_masked import masked_bce, train_one_epoch, predict

    device = best_device()
    print(f"Device: {device}  |  Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eval_set = MultiLabelDataset(eval_s, eval_y, eval_m, tokenizer, args.max_length)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size)

    def train_and_eval(train_s, train_y, train_m, seed):
        set_seed(seed)
        model = SharedEncoderMTL(args.model, num_labels=8).to(device)
        ds = MultiLabelDataset(train_s, train_y, train_m, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        opt = AdamW(model.parameters(), lr=args.lr)
        total = max(1, len(loader) * args.epochs)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        for ep in range(args.epochs):
            loss = train_one_epoch(model, loader, opt, sched, device)
            print(f"  seed{seed} epoch{ep+1}/{args.epochs} loss={loss:.4f}")
        preds, probs, labels, masks = predict(model, eval_loader, device)
        return preds

    rows = []
    for seed in args.seeds:
        print(f"\n--- seed {seed} : train_leak ---")
        preds_leak = train_and_eval(leak_s, leak_y, leak_m, seed)
        print(f"--- seed {seed} : train_clean ---")
        preds_clean = train_and_eval(clean_s, clean_y, clean_m, seed)

        pl, ml, _ = pooled_and_macro_f1(eval_y, preds_leak, eval_m)
        pc, mc, _ = pooled_and_macro_f1(eval_y, preds_clean, eval_m)
        bmean, blo, bhi = paired_bootstrap(eval_y, eval_m, preds_leak, preds_clean, args.n_boot, seed)
        rows.append({
            "seed": seed,
            "pooled_f1_leak": pl, "pooled_f1_clean": pc, "pooled_delta": pl - pc,
            "macro_f1_leak": ml, "macro_f1_clean": mc, "macro_delta": ml - mc,
            "boot_delta_mean": bmean, "boot_delta_lo95": blo, "boot_delta_hi95": bhi,
        })
        print(f"  seed{seed}: pooled F1 leak={pl:.4f} clean={pc:.4f} "
              f"delta={pl-pc:+.4f}  boot95=[{blo:+.4f},{bhi:+.4f}]")

    df = pd.DataFrame(rows)
    print("\n=== Across seeds ===")
    print(df.to_string(index=False))
    print(f"\npooled delta: mean={df['pooled_delta'].mean():+.4f} "
          f"std={df['pooled_delta'].std():.4f}")
    df.to_csv(out_dir / "06_bert_controlled_injection.csv", index=False)
    print(f"Wrote {out_dir / '06_bert_controlled_injection.csv'}")


if __name__ == "__main__":
    main()
