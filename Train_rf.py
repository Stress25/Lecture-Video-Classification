"""
train_rf.py
Trains a Random Forest classifier on SBERT embeddings to produce per-segment
action class probabilities (text-only baseline).

"""
import argparse
import csv
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
)

CANONICAL_LABELS = [
    "explain", "write", "out", "out_writing", "out_erasing",
    "erase", "pick_eraser", "drop_eraser",
]
LABEL_MAP  = {lab: i for i, lab in enumerate(CANONICAL_LABELS)}
IDX_TO_LAB = {i: lab for lab, i in LABEL_MAP.items()}

NORMALISE = {
    "explain":"explain", "write":"write", "out":"out",
    "out_writing":"out_writing", "outwriting":"out_writing", "out-writing":"out_writing",
    "out_erasing":"out_erasing", "out-erasing":"out_erasing",
    "erase":"erase", "fingererase":"erase",
    "pick_eraser":"pick_eraser", "pickerase":"pick_eraser", "pick":"pick_eraser",
    "drop_eraser":"drop_eraser", "droperase":"drop_eraser", "drop":"drop_eraser",
}


def load_data(emb_path, mask_path, meta_path):
    embeddings = np.load(emb_path).astype(np.float32)
    mask       = np.load(mask_path).astype(np.float32)
    keep, norm_labels, segment_ids = [], [], []
    dropped = {}
    with open(meta_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            canonical = NORMALISE.get(row["label"])
            if canonical is None:
                dropped[row["label"]] = dropped.get(row["label"], 0) + 1
            else:
                keep.append(i)
                norm_labels.append(LABEL_MAP[canonical])
                segment_ids.append(int(row["segment_id"]))
    keep       = np.array(keep, dtype=np.int64)
    embeddings = embeddings[keep]
    mask       = mask[keep]
    labels     = np.array(norm_labels, dtype=np.int64)
    seg_ids    = np.array(segment_ids, dtype=np.int64)
    print(f"Loaded  {len(keep) + sum(dropped.values())} segments total")
    if dropped:
        print(f"Dropped {sum(dropped.values())} non-canonical: {dropped}")
    print(f"Kept    {len(embeddings)} segments across {len(CANONICAL_LABELS)} classes")
    print(f"  Has text : {int(mask.sum())} / {len(mask)}")
    print(f"  Silent   : {int((1-mask).sum())} / {len(mask)}")
    return embeddings, labels, mask, seg_ids


def make_split(labels, test_size, seed):
    from collections import Counter
    counts    = Counter(labels.tolist())
    can_strat = np.array([counts[l] >= 2 for l in labels])
    strat_idx   = np.where(can_strat)[0]
    force_train = np.where(~can_strat)[0]
    tr, te = train_test_split(
        strat_idx, test_size=test_size,
        random_state=seed, stratify=labels[strat_idx],
    )
    train_idx = np.concatenate([tr, force_train])
    np.random.default_rng(seed).shuffle(train_idx)
    return train_idx, te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings",   required=True)
    p.add_argument("--mask",         required=True)
    p.add_argument("--meta",         required=True)
    p.add_argument("--out-dir",      default="results/rf_text")
    p.add_argument("--test-size",    type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-estimators", type=int,   default=200)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    embeddings, labels, mask, seg_ids = load_data(
        args.embeddings, args.mask, args.meta
    )
    n_classes = len(CANONICAL_LABELS)

    counts = np.bincount(labels, minlength=n_classes)
    print("\nClass counts:")
    for i, lab in enumerate(CANONICAL_LABELS):
        print(f"  {lab:<14} {counts[i]:>5}")

    train_idx, test_idx = make_split(labels, args.test_size, args.seed)
    X_train, Y_train = embeddings[train_idx], labels[train_idx]
    X_test,  Y_test  = embeddings[test_idx],  labels[test_idx]
    print(f"\nSplit: train={len(train_idx)}  test={len(test_idx)}")

    # Majority baseline
    maj      = int(np.argmax(counts))
    naive_acc = accuracy_score(Y_test, np.full(len(Y_test), maj))
    naive_f1  = f1_score(Y_test, np.full(len(Y_test), maj),
                         average="macro", zero_division=0)
    print(f"\nMajority baseline (always '{IDX_TO_LAB[maj]}'):")
    print(f"  Accuracy : {naive_acc:.4f}")
    print(f"  Macro-F1 : {naive_f1:.4f}")

    # Random Forest
    print("\n. Random Forest ")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.seed,
    )
    clf.fit(X_train, Y_train)

    probs_test = clf.predict_proba(X_test)
    preds_test = np.argmax(probs_test, axis=1)

    acc = accuracy_score(Y_test, preds_test)
    f1m = f1_score(Y_test, preds_test, average="macro",    zero_division=0)
    f1w = f1_score(Y_test, preds_test, average="weighted", zero_division=0)

    print(f"  Test accuracy  : {acc:.4f}")
    print(f"  Test macro-F1  : {f1m:.4f}")
    print(f"  Test wtd-F1    : {f1w:.4f}")
    print()

    present = sorted(set(Y_test.tolist()) | set(preds_test.tolist()))
    print(classification_report(
        Y_test, preds_test,
        labels=present,
        target_names=[IDX_TO_LAB[i] for i in present],
        zero_division=0,
    ))

    # Feature importance — top 20 embedding dimensions
    importances = clf.feature_importances_
    top20 = np.argsort(importances)[::-1][:20]
    print("Top 20 most important embedding dimensions:")
    for rank, idx in enumerate(top20):
        print(f"  {rank+1:>2}. dim {idx:>3}  importance={importances[idx]:.4f}")

    # Save full-set probs
    probs_all = clf.predict_proba(embeddings)
    rf_out    = os.path.join(args.out_dir, "rf_text_probs.npy")
    np.save(rf_out, probs_all)
    np.save(os.path.join(args.out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(args.out_dir, "test_idx.npy"),  test_idx)
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    # Per-segment probs CSV
    prob_csv = os.path.join(args.out_dir, "test_segment_probs.csv")
    with open(prob_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["segment_id", "gt_label", "pred_label", "correct"] +
            [f"p_{lab}" for lab in CANONICAL_LABELS]
        )
        for i in range(len(test_idx)):
            gt_idx   = int(Y_test[i])
            pred_idx = int(np.argmax(probs_test[i]))
            writer.writerow(
                [seg_ids[test_idx[i]], IDX_TO_LAB[gt_idx],
                 IDX_TO_LAB[pred_idx], 1 if pred_idx == gt_idx else 0] +
                [f"{p:.4f}" for p in probs_test[i]]
            )

    # Per-segment probability print
    print("\n" + "=" * 60)
    print("PER-SEGMENT PROBABILITY VECTORS  (test set)")
    print("=" * 60)
    max_lab = max(len(l) for l in CANONICAL_LABELS)
    lab_header = "  ".join(f"{l[:7]:>7}" for l in CANONICAL_LABELS)
    print(f"  {'seg':>5}  {'gt_label':<{max_lab}}  {'pred_label':<{max_lab}}  "
          f"{'ok':>2}  {lab_header}")
    print("  " + "-" * (5 + max_lab*2 + 12 + len(CANONICAL_LABELS)*9 + 6))
    for i in range(len(test_idx)):
        gt_idx   = int(Y_test[i])
        gt_lab   = IDX_TO_LAB[gt_idx]
        prob     = probs_test[i]
        pred_idx = int(np.argmax(prob))
        pred_lab = IDX_TO_LAB[pred_idx]
        correct  = "OK" if pred_idx == gt_idx else "--"
        prob_str = "  ".join(f"{p:>7.3f}" for p in prob)
        print(f"  {seg_ids[test_idx[i]]:>5}  {gt_lab:<{max_lab}}  "
              f"{pred_lab:<{max_lab}}  {correct:>2}  {prob_str}")

    summary = {
        "n_segments_kept": int(len(embeddings)),
        "n_classes": n_classes,
        "n_train": int(len(train_idx)),
        "n_test":  int(len(test_idx)),
        "majority_baseline": {"acc": float(naive_acc), "macro_f1": float(naive_f1)},
        "random_forest": {"acc": float(acc), "macro_f1": float(f1m), "wtd_f1": float(f1w)},
        "probs_path": rf_out,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Majority baseline : acc={naive_acc:.4f}  macro-f1={naive_f1:.4f}")
    print(f"  Random Forest     : acc={acc:.4f}  macro-f1={f1m:.4f}")
    print(f"  Lift over baseline: acc={acc-naive_acc:+.4f}  "
          f"macro-f1={f1m-naive_f1:+.4f}")
    print(f"\n  Probs saved -> {rf_out}  shape {probs_all.shape}")
    print(f"  Per-segment CSV -> {prob_csv}")


if __name__ == "__main__":
    main()