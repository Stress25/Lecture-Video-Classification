"""
The script produces (N,8) probability vectors such as [0.85, 0.03, 0.01, 0.02, 0.01, 0.01, 0.05, 0.02]
meaning 85% explain, 3% write, 1% out
"""
import argparse
import csv
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CANONICAL_LABELS = [
    "drop_eraser",
    "erase", 
    "explain",
    "out",
    "out_erasing",
    "out_writing",
    "pick_eraser",
    "write",
]

# convert labels to numbers
LABEL_MAP = {lab: i for i, lab in enumerate(CANONICAL_LABELS)}
IDX_TO_LAB = {i: lab for lab, i in LABEL_MAP.items()}

# Maps every observed variant in the data to canonical class
# Anything not listed here is dropped from training/evaluation.
NORMALISE = {
    # explain
    "explain":      "explain",
    # write
    "write":        "write",
    # out
    "out":          "out",
    # out_writing variants
    "out_writing":  "out_writing",
    "outwriting":   "out_writing",
    "out-writing":  "out_writing",
    # out_erasing variants
    "out_erasing":  "out_erasing",
    "out-erasing":  "out_erasing",
    # erase variants
    "erase":        "erase",
    "fingererase":  "erase",
    # pick_eraser variants
    "pick_eraser":  "pick_eraser",
    "pickerase":    "pick_eraser",
    "pick":         "pick_eraser",
    # drop_eraser variants
    "drop_eraser":  "drop_eraser",
    "droperase":    "drop_eraser",
    "drop":         "drop_eraser",
    # change and video_switch are not in the paper's 8 classes so they are  dropped
}

# load data
def load_data(emb_path, mask_path, meta_path):
    embeddings = np.load(emb_path).astype(np.float32)
    mask       = np.load(mask_path).astype(np.float32)

    keep, norm_labels, segment_ids = [], [], []
    dropped = {}

    with open(meta_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            raw       = row["label"]
            canonical = NORMALISE.get(raw)   # normalise FIRST
            if canonical is None:
                dropped[raw] = dropped.get(raw, 0) + 1
            else:
                keep.append(i)
                norm_labels.append(LABEL_MAP[canonical])
                segment_ids.append(int(row["segment_id"]))

    keep       = np.array(keep,        dtype=np.int64)
    embeddings = embeddings[keep]
    mask       = mask[keep]
    labels     = np.array(norm_labels, dtype=np.int64)
    seg_ids    = np.array(segment_ids, dtype=np.int64)

    print(f"Loaded  {len(keep) + sum(dropped.values())} segments total")
    if dropped:
        print(f"Dropped {sum(dropped.values())} non-canonical:")
        for lab, n in sorted(dropped.items(), key=lambda x: -x[1]):
            print(f"  '{lab}': {n}")
    print(f"Kept    {len(embeddings)} segments across {len(CANONICAL_LABELS)} classes")
    print(f"  Has text : {int(mask.sum())} / {len(mask)}")
    print(f"  Silent   : {int((1-mask).sum())} / {len(mask)}")
    return embeddings, labels, mask, seg_ids



def class_weight_tensor(labels_train,n_classes,device):
    # rare classes get larger weights
    """
    Explain      1/2000
    VideoSwitch  1/20
    """

    counts = np.bincount(labels_train,minlength=n_classes).astype(float)
    w = np.where(counts > 0, 1.0 / counts, 0.0)
    if w.sum() > 0:
        w = w * n_classes / w.sum()
    return torch.tensor(w,dtype=torch.float32).to(device)


def make_split(embeddings,labels,test_size,seed):
    from collections import Counter
    # count how many times each unique class appears in the labels array
    counts = Counter(labels.tolist())
    # Stratified data
    can_strat = np.array([counts[i] >= 2 for i in labels])
    # safe indices
    strat_index = np.where(can_strat)[0]
    # extract array indices of singleton classes with only 1 sample
    force_train = np.where(~can_strat)[0]

    # pass only the safe indices(strat_index)
    tr,te = train_test_split(strat_index,test_size=test_size,random_state=seed,
                             stratify=labels[strat_index])
    
    """
    recombine the stratified training indices (tr) and glues the 
    singleton indices to end of them. This guarantees that any class with only
    one example is forced into the training data so the model actually gets to learn from it.
    """
    train_dx = np.concatenate([tr,force_train])
    rng = np.random.default_rng(seed)
    rng.shuffle(train_dx)
    return train_dx,te
 
# svm classifier
def train_svm(X_train, y_train, X_test, y_test, seed):
    print("\nSVM")
    clf = SVC(kernel="rbf",C=10.0,gamma="scale",class_weight="balanced",probability=True,random_state=seed)
    
    """
    standard SVMs do not output probabilities (they output the mathematical distance to the decision boundary)
    """
    # clf = CalibratedClassifierCV(base_svm,cv=2)
    clf.fit(X_train,y_train)

    # generating predictions
    # Calculates probability of each class for the test set
    probs_test = clf.predict_proba(X_test)
    # take the probabilities and pick the final class prediction by finding the highest probability
    preds_test = np.argmax(probs_test,axis=1)
    probs_all = clf.predict_proba(np.vstack([X_train,X_test]))

    # evaluating performance
    acc = accuracy_score(y_test,preds_test)
    f1m = f1_score(y_test,preds_test,average="macro",zero_division=0)
    f1w = f1_score(y_test,preds_test,average="weighted",zero_division=0)

    print(f"  Test accuracy  : {acc:.4f}")
    print(f"  Test macro-F1  : {f1m:.4f}")
    print(f"  Test wtd-F1    : {f1w:.4f}")
    print()
    # Use only labels that actually appear in test set for the report
    present = sorted(set(y_test.tolist()) | set(preds_test.tolist()))
    print(classification_report(
        y_test, preds_test,
        labels=present,
        target_names=[IDX_TO_LAB[i] for i in present],
        zero_division=0,
    ))
    return clf, probs_test, probs_all, acc, f1m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--mask",       required=True)
    p.add_argument("--meta",       required=True)
    p.add_argument("--out-dir",    default="results/text_classifier")
    p.add_argument("--test-size",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.3)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir,exist_ok=True)

    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device {device}")

    # load data
    embeddings,labels,mask,seg_ids = load_data(args.embeddings,args.mask, args.meta)
    n_classes = len(CANONICAL_LABELS)

    # class distribution
    print("\nclass counts:")
    counts = np.bincount(labels,minlength=n_classes)
    for i , lab in enumerate(CANONICAL_LABELS):
        print(f" {lab}{counts[i]}")

    # split
    train_idx , test_idx = make_split(embeddings,labels,args.test_size,args.seed)

    print(f"\nsplit: train = {len(train_idx)}, test = {len(test_idx)}")

    X_train = embeddings[train_idx]
    Y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    Y_test = labels[test_idx]
    X_all = embeddings

    # majority baseline
    majority_cls = int(np.argmax(counts))
    naive_preds = np.full(len(Y_test),majority_cls)
    naive_acc = accuracy_score(Y_test,naive_preds)
    naive_f1 = f1_score(Y_test,naive_preds,average="macro",zero_division=0)

    print(f"\nMajority baseline (always '{IDX_TO_LAB[majority_cls]}'):")
    print(f"  Accuracy  : {naive_acc:.4f}")
    print(f"  Macro-F1  : {naive_f1:.4f}")

    # svm
    svm_clf, svm_probs_test, svm_probs_all, svm_acc, svm_f1 = train_svm(
        X_train, Y_train, X_test, Y_test, args.seed
    )

    svm_probs_save = svm_probs_all[:len(train_idx)]

    n_classes_actual = svm_probs_all.shape[1] 
    svm_ordered = np.zeros((len(embeddings),n_classes_actual),dtype=np.float32)
    svm_ordered[train_idx] = svm_probs_all[:len(train_idx)]
    svm_ordered[test_idx] = svm_probs_all[len(train_idx):]

    svm_out = os.path.join(args.out_dir, "svm_text_probs.npy")
    np.save(svm_out, svm_ordered)

    # Save train/test indices for reproducibility
    np.save(os.path.join(args.out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(args.out_dir, "test_idx.npy"),  test_idx)

    # summary
    summary = {
        "n_segments": int(len(embeddings)),
        "n_train": int(len(train_idx)),
        "n_test":  int(len(test_idx)),
        "majority_baseline": {"acc": float(naive_acc), "macro_f1": float(naive_f1)},
        "svm": {"acc": float(svm_acc), "macro_f1": float(svm_f1)},
        "svm_probs_path": svm_out,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
 
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Majority baseline : acc={naive_acc:.4f}  f1={naive_f1:.4f}")
    print(f"  SVM               : acc={svm_acc:.4f}  f1={svm_f1:.4f}")
    print(f"\n  Probability arrays saved:")
    print(f"    SVM → {svm_out}  shape {svm_ordered.shape}")
    print(f"\n  Use these .npy files in fusion.py with the pose probs.")

    # Per-segment probability vectors (test set)
    print("\n" + "=" * 60)
    print("PER-SEGMENT PROBABILITY VECTORS  (test set)")
    print("=" * 60)
    max_lab = max(len(l) for l in CANONICAL_LABELS)
    lab_header = "  ".join(f"{l[:7]:>7}" for l in CANONICAL_LABELS)
    print(f"  {'seg':>5}  {'gt_label':<{max_lab}}  {'pred_label':<{max_lab}}  "
          f"{'ok':>2}  {lab_header}")
    print("  " + "-" * (5 + max_lab*2 + 12 + len(CANONICAL_LABELS)*9 + 6))
 
    for i in range(len(test_idx)):
        seg_id   = int(seg_ids[test_idx[i]])
        gt_idx   = int(Y_test[i])
        gt_lab   = IDX_TO_LAB[gt_idx]
        prob     = svm_probs_test[i]
        pred_idx = int(np.argmax(prob))
        pred_lab = IDX_TO_LAB[pred_idx]
        correct  = "OK" if pred_idx == gt_idx else "--"
        prob_str = "  ".join(f"{p:>7.3f}" for p in prob)
        print(f"  {seg_id:>5}  {gt_lab:<{max_lab}}  {pred_lab:<{max_lab}}  "
              f"{correct:>2}  {prob_str}")
 
    # Save as CSV
    prob_csv = os.path.join(args.out_dir, "test_segment_probs.csv")
    with open(prob_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["segment_id", "gt_label", "pred_label", "correct"] +
            [f"p_{lab}" for lab in CANONICAL_LABELS]
        )
        for i in range(len(test_idx)):
            seg_id   = int(seg_ids[test_idx[i]])
            gt_idx   = int(Y_test[i])
            gt_lab   = IDX_TO_LAB[gt_idx]
            prob     = svm_probs_test[i]
            pred_idx = int(np.argmax(prob))
            pred_lab = IDX_TO_LAB[pred_idx]
            correct  = 1 if pred_idx == gt_idx else 0
            writer.writerow(
                [seg_id, gt_lab, pred_lab, correct] +
                [f"{p:.4f}" for p in prob]
            )
    print(f"\n  Per-segment probs saved to {prob_csv}")
 
if __name__ == "__main__":
    main()
 