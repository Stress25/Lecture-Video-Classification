"""
enriched_dataset.py

Merges aligned CSV (segment metadata + transcripts) with SVM/MLP/RF
probability vectors to produce one enriched CSV per segment.


Usage:
  python enriched_dataset.py \
      --aligned    output_test/all_test_aligned_contained.csv \
      --svm-probs  results/contained_final/svm_text_probs.npy \
      --mlp-probs  results/contained_final/mlp_text_probs.npy \
      --rf-probs   results/contained_final/rf_text_probs.npy \
      --meta       embeddings_test/contained_meta.csv \
      --out        results/enriched_test_dataset.csv
"""
import argparse
import csv
import os
import numpy as np

CANONICAL_LABELS = [
    "drop_eraser", "erase", "explain", "out",
    "out_erasing", "out_writing", "pick_eraser", "write",
]
NORMALISE = {
    "explain":"explain", "write":"write", "out":"out",
    "out_writing":"out_writing", "outwriting":"out_writing", "out-writing":"out_writing",
    "out_erasing":"out_erasing", "out-erasing":"out_erasing",
    "erase":"erase", "fingererase":"erase",
    "pick_eraser":"pick_eraser", "pickerase":"pick_eraser", "pick":"pick_eraser",
    "drop_eraser":"drop_eraser", "droperase":"drop_eraser", "drop":"drop_eraser",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aligned",   required=True)
    p.add_argument("--svm-probs", required=True)
    p.add_argument("--mlp-probs", default=None)
    p.add_argument("--rf-probs",  default=None)
    p.add_argument("--meta",      required=True)
    p.add_argument("--out",       default="results/enriched_test_dataset.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # ── Load probability arrays ──
    svm_probs = np.load(args.svm_probs).astype(np.float32)
    mlp_probs = np.load(args.mlp_probs).astype(np.float32) if args.mlp_probs else None
    rf_probs  = np.load(args.rf_probs).astype(np.float32)  if args.rf_probs  else None

    print(f"SVM probs shape : {svm_probs.shape}")
    if mlp_probs is not None: print(f"MLP probs shape : {mlp_probs.shape}")
    if rf_probs  is not None: print(f"RF  probs shape : {rf_probs.shape}")

    # ── Load meta — row i in meta = row i in probs array ──
    # DO NOT use segment_id as a key: it restarts from 0 per video.
    # The meta CSV and aligned CSV share the same row ordering (both derived
    # from the same pooled CSV), so we walk them in lockstep.
    meta_rows    = list(csv.DictReader(open(args.meta,    newline="")))
    aligned_rows = list(csv.DictReader(open(args.aligned, newline="")))

    print(f"Meta rows       : {len(meta_rows)}")
    print(f"Aligned CSV rows: {len(aligned_rows)}")

    if len(meta_rows) != len(aligned_rows):
        print("WARNING: meta and aligned row counts differ — "
              "they may not share the same source CSV.")

    # ── Build enriched rows ──
    fieldnames = [
        "segment_id", "video_id", "label", "canonical_label",
        "start_frame", "end_frame", "start_time", "end_time", "duration_sec",
        "word_count", "is_silence", "has_text", "transcript",
        "svm_pred", "svm_correct",
    ]
    if mlp_probs is not None:
        fieldnames += ["mlp_pred", "mlp_correct"]
    if rf_probs is not None:
        fieldnames += ["rf_pred", "rf_correct"]
    for lab in CANONICAL_LABELS:
        fieldnames.append(f"p_{lab}")
    if mlp_probs is not None:
        for lab in CANONICAL_LABELS:
            fieldnames.append(f"mlp_p_{lab}")
    if rf_probs is not None:
        for lab in CANONICAL_LABELS:
            fieldnames.append(f"rf_p_{lab}")

    enriched = []
    skipped  = 0

    # Walk aligned and meta in lockstep — row i in aligned = row i in meta = row i in probs
    for prob_idx, (row, meta_row) in enumerate(zip(aligned_rows, meta_rows)):
        raw_label = row["label"]
        canonical = NORMALISE.get(raw_label)

        svm_p    = svm_probs[prob_idx]
        svm_pred = CANONICAL_LABELS[int(np.argmax(svm_p))]
        svm_ok   = 1 if (canonical and svm_pred == canonical) else 0

        out = {
            "segment_id":      row["segment_id"],
            "video_id":        row.get("video_id", ""),
            "label":           raw_label,
            "canonical_label": canonical if canonical else "dropped",
            "start_frame":     row.get("start_frame", ""),
            "end_frame":       row.get("end_frame", ""),
            "start_time":      row.get("start_time", ""),
            "end_time":        row.get("end_time", ""),
            "duration_sec":    row.get("duration_sec", ""),
            "word_count":      row.get("word_count", ""),
            "is_silence":      row.get("is_silence", ""),
            "has_text":        row.get("has_text", ""),
            "transcript":      row.get("transcript", ""),
            "svm_pred":        svm_pred,
            "svm_correct":     svm_ok,
        }

        if mlp_probs is not None:
            mlp_p    = mlp_probs[prob_idx]
            mlp_pred = CANONICAL_LABELS[int(np.argmax(mlp_p))]
            out["mlp_pred"]    = mlp_pred
            out["mlp_correct"] = 1 if (canonical and mlp_pred == canonical) else 0

        if rf_probs is not None:
            rf_p    = rf_probs[prob_idx]
            rf_pred = CANONICAL_LABELS[int(np.argmax(rf_p))]
            out["rf_pred"]    = rf_pred
            out["rf_correct"] = 1 if (canonical and rf_pred == canonical) else 0

        for i, lab in enumerate(CANONICAL_LABELS):
            out[f"p_{lab}"] = f"{svm_p[i]:.4f}"
        if mlp_probs is not None:
            for i, lab in enumerate(CANONICAL_LABELS):
                out[f"mlp_p_{lab}"] = f"{mlp_probs[prob_idx][i]:.4f}"
        if rf_probs is not None:
            for i, lab in enumerate(CANONICAL_LABELS):
                out[f"rf_p_{lab}"] = f"{rf_probs[prob_idx][i]:.4f}"

        enriched.append(out)

    # ── Write ──
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"\nWritten {len(enriched)} rows → {args.out}")

    # ── Quick stats ──
    canon = [r for r in enriched if r["canonical_label"] != "dropped"]
    if canon:
        svm_acc = sum(r["svm_correct"] for r in canon) / len(canon)
        print(f"SVM accuracy (enriched): {svm_acc:.4f}")
        if mlp_probs is not None:
            mlp_acc = sum(r["mlp_correct"] for r in canon) / len(canon)
            print(f"MLP accuracy (enriched): {mlp_acc:.4f}")


if __name__ == "__main__":
    main()