"""
predict_text_probs.py
Trains SVM, MLP, and Random Forest classifiers on training video SBERT
embeddings, then applies each to test video embeddings.

pipeline:
  - Trained on : training video segments (17 videos)
  - Evaluated on: test video segments    (17 videos)

Outputs saved to --out-dir:
  svm_text_probs.npy      (N_test_total, 8)
  mlp_text_probs.npy      (N_test_total, 8)
  rf_text_probs.npy       (N_test_total, 8)
  test_segment_probs.csv  (SVM predictions per segment)
  summary.json

Usage:
  python predict_text_probs.py \
      --train-embeddings embeddings_train/overlap_embeddings.npy \
      --train-mask       embeddings_train/overlap_mask.npy \
      --train-meta       embeddings_train/overlap_meta.csv \
      --test-embeddings  embeddings_test/overlap_embeddings.npy \
      --test-mask        embeddings_test/overlap_mask.npy \
      --test-meta        embeddings_test/overlap_meta.csv \
      --out-dir          results/final
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
)

CANONICAL_LABELS = [
    "drop_eraser", "erase", "explain", "out",
    "out_erasing", "out_writing", "pick_eraser", "write",
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


def load_embeddings(emb_path, mask_path, meta_path):
    embeddings = np.load(emb_path).astype(np.float32)
    mask       = np.load(mask_path).astype(np.float32)
    keep, norm_labels, seg_ids = [], [], []
    dropped = {}
    with open(meta_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            canonical = NORMALISE.get(row["label"])
            if canonical is None:
                dropped[row["label"]] = dropped.get(row["label"], 0) + 1
            else:
                keep.append(i)
                norm_labels.append(LABEL_MAP[canonical])
                seg_ids.append(int(row["segment_id"]))
    keep    = np.array(keep,       dtype=np.int64)
    labels  = np.array(norm_labels, dtype=np.int64)
    seg_ids = np.array(seg_ids,    dtype=np.int64)
    if dropped:
        print(f"  Dropped {sum(dropped.values())} non-canonical: {dropped}")
    return embeddings[keep], labels, mask[keep], seg_ids, keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-embeddings", required=True)
    p.add_argument("--train-mask",       required=True)
    p.add_argument("--train-meta",       required=True)
    p.add_argument("--test-embeddings",  required=True)
    p.add_argument("--test-mask",        required=True)
    p.add_argument("--test-meta",        required=True)
    p.add_argument("--out-dir",          default="results/final")
    p.add_argument("--seed",             type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    n_classes = len(CANONICAL_LABELS)

    # ── Load data ──
    print("=" * 60)
    print("Loading TRAINING embeddings...")
    X_train, y_train, _, _, _ = load_embeddings(
        args.train_embeddings, args.train_mask, args.train_meta)
    print(f"  Training segments : {len(X_train)}")

    counts_train = np.bincount(y_train, minlength=n_classes)
    print("\n  Class counts (train):")
    for i, lab in enumerate(CANONICAL_LABELS):
        print(f"    {lab:<14} {counts_train[i]:>5}")

    print("\nLoading TEST embeddings...")
    X_test, y_test, _, seg_ids_test, keep_test = load_embeddings(
        args.test_embeddings, args.test_mask, args.test_meta)
    print(f"  Test segments     : {len(X_test)}")

    counts_test = np.bincount(y_test, minlength=n_classes)
    print("\n  Class counts (test):")
    for i, lab in enumerate(CANONICAL_LABELS):
        print(f"    {lab:<14} {counts_test[i]:>5}")

    # Majority baseline
    maj      = int(np.argmax(counts_test))
    naive_acc = accuracy_score(y_test, np.full(len(y_test), maj))
    naive_f1  = f1_score(y_test, np.full(len(y_test), maj),
                         average="macro", zero_division=0)
    print(f"\nMajority baseline (always '{IDX_TO_LAB[maj]}'):")
    print(f"  Accuracy : {naive_acc:.4f}   Macro-F1 : {naive_f1:.4f}")

    #  Define classifiers 
    classifiers = {
        "SVM":          SVC(kernel="rbf", C=10.0, gamma="scale",
                            class_weight="balanced", probability=True,
                            random_state=args.seed),
        "MLP":          MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                                      early_stopping=True,
                                      random_state=args.seed),
        "Random Forest": RandomForestClassifier(n_estimators=200,
                                                class_weight="balanced",
                                                random_state=args.seed),
    }

    # Output file names per classifier
    out_files = {
        "SVM":           "svm_text_probs.npy",
        "MLP":           "mlp_text_probs.npy",
        "Random Forest": "rf_text_probs.npy",
    }

    all_results = {}
    svm_probs_test = None   

    n_total = int(np.load(args.test_embeddings).shape[0])

    for clf_name, clf in classifiers.items():
        print("\n" + "=" * 60)
        print(f"[{clf_name}] Training on {len(X_train)} segments...")
        clf.fit(X_train, y_train)

        # Evaluate on test set
        pt   = clf.predict_proba(X_test)       # (N_test_kept, 8)
        pd   = np.argmax(pt, axis=1)
        acc  = accuracy_score(y_test, pd)
        f1m  = f1_score(y_test, pd, average="macro",    zero_division=0)
        f1w  = f1_score(y_test, pd, average="weighted", zero_division=0)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Macro-F1  : {f1m:.4f}")
        print(f"  Wtd-F1    : {f1w:.4f}")
        print()

        present = sorted(set(y_test.tolist()) | set(pd.tolist()))
        print(classification_report(
            y_test, pd, labels=present,
            target_names=[IDX_TO_LAB[i] for i in present],
            zero_division=0,
        ))

        # Save full-set probs (N_total, 8) with uniform prior for dropped rows
        probs_full = np.full((n_total, n_classes),
                             1.0 / n_classes, dtype=np.float32)
        probs_full[keep_test] = pt
        out_path = os.path.join(args.out_dir, out_files[clf_name])
        np.save(out_path, probs_full)
        print(f"  Saved -> {out_path}  shape {probs_full.shape}")

        all_results[clf_name] = {
            "acc": float(acc), "f1_macro": float(f1m), "f1_wtd": float(f1w),
            "probs_path": out_path,
        }

        if clf_name == "SVM":
            svm_probs_test = pt  

    # Save per-segment CSV 
    prob_csv = os.path.join(args.out_dir, "test_segment_probs.csv")
    with open(prob_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["segment_id", "gt_label", "pred_label", "correct"] +
            [f"p_{lab}" for lab in CANONICAL_LABELS]
        )
        for i in range(len(X_test)):
            gt_idx   = int(y_test[i])
            pred_idx = int(np.argmax(svm_probs_test[i]))
            writer.writerow(
                [seg_ids_test[i], IDX_TO_LAB[gt_idx],
                 IDX_TO_LAB[pred_idx],
                 1 if pred_idx == gt_idx else 0] +
                [f"{p:.4f}" for p in svm_probs_test[i]]
            )

    # Summary 
    best = max(all_results.items(), key=lambda x: x[1]["f1_macro"])

    summary = {
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "majority_baseline": {"acc": float(naive_acc), "macro_f1": float(naive_f1)},
        "results": {k: {"acc": v["acc"], "f1_macro": v["f1_macro"],
                        "f1_wtd": v["f1_wtd"]}
                    for k, v in all_results.items()},
        "best_classifier": best[0],
        "label_order": CANONICAL_LABELS,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY : ALL CLASSIFIERS")
    print("=" * 60)
    print(f"  Train segments : {len(X_train)}")
    print(f"  Test  segments : {len(X_test)}")
    print(f"\n  {'Classifier':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print("  " + "-" * 44)
    print(f"  {'Majority baseline':<20} {naive_acc:>10.4f} {naive_f1:>10.4f}")
    for clf_name, res in all_results.items():
        print(f"  {clf_name:<20} {res['acc']:>10.4f} {res['f1_macro']:>10.4f}")
    print(f"\n  Best: {best[0]}  (macro-F1={best[1]['f1_macro']:.4f})")
    print(f"\n  Files saved to: {args.out_dir}/")
    for clf_name, fname in out_files.items():
        print(f"    {fname}")
    print(f"    test_segment_probs.csv")
    print(f"    summary.json")


if __name__ == "__main__":
    main()