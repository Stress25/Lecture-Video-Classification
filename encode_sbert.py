"""
The script reads transcript segments, identifies silent segments, converts all 
spoken text into normalized SBERT sentence embeddings, assigns a neutral embedding to 
silence, creates a text-presence mask, and saves everything in NumPy files so the embeddings 
can be fused later with pose or video features for machine learning.
"""
import argparse
import csv
import os
import numpy as np

SILENCE_TOKEN = "[silence]"

def load_csv(filepath):
    rows = []
    with open(filepath,newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("aligned_csv")
    p.add_argument("out_prefix",
                   help="path prefix, e.g. embeddings/contained")
    p.add_argument("--model", default="all-MiniLM-L6-v2",
                   help="any sentence-transformers model name")
    p.add_argument("--silence", default="mean",
                   choices=["mean", "zero"],
                   help="embedding to use for silent segments")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    # load SBERT MODEL
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    rows = load_csv(args.aligned_csv)
    print(f"Loaded {len(rows)} segments from {args.aligned_csv}")

    # split into text-bearing vs silent
    texts = []
    text_idx = []
    is_silence = np.zeros(len(rows),dtype=np.int8)
    for i, r in enumerate(rows):
        silent = (r.get("is_silence","0") == "1") or \
                  (r["transcript"].strip() == SILENCE_TOKEN) or \
                  (r["transcript"].strip() == "")
        if silent:
            is_silence[i] = 1
        else:
            texts.append(r["transcript"])
            text_idx.append(i)

    
    n_silent = int(is_silence.sum())
    print(f"Text Segments : {len(text_idx)}")
    print(f"Silent Segments : {n_silent}")

    print(f"Loading SBERT MODEL: {args.model}")
    dim = model.get_embedding_dimension()
    print(f"Embedding dim: {dim}")

    embeddings = np.zeros((len(rows),dim),dtype = np.float32)

    if texts:
        print(f"Encoding {len(texts)} transcripts...")
        enc = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-norm; good for cosine fusion
        ).astype(np.float32)

        for k,i in enumerate(text_idx):
            embeddings[i] = enc[k]
        if args.silence == "mean":
            silence_vec = enc.mean(axis=0)
            norm = np.linalg.norm(silence_vec)
            if norm > 0:
                silence_vec = silence_vec / norm
        else:
            silence_vec = np.zeros(dim,dtype=np.float32)
    else:
        silence_vec = np.zeros(dim,dtype=np.float32)

    # fill silent rows
    for i in range(len(rows)):
        if is_silence[i] == 1:
            embeddings[i] = silence_vec

    has_text = (1-is_silence).astype(np.int8)

    # save outputs
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir,exist_ok=True)

    emb_path = f"{args.out_prefix}_embeddings.npy"
    mask_path = f"{args.out_prefix}_mask.npy"
    meta_path = f"{args.out_prefix}_meta.csv"

    np.save(emb_path,embeddings)
    np.save(mask_path,has_text)

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "label", "word_count",
                         "is_silence", "has_text"])
        for i, r in enumerate(rows):
            writer.writerow([r["segment_id"], r["label"],
                             r["word_count"], int(is_silence[i]),
                             int(has_text[i])])
            
    print()
    print(f"  embeddings -> {emb_path}   shape {embeddings.shape}")
    print(f"  mask       -> {mask_path}  shape {has_text.shape} "
          f"({int(has_text.sum())} text / {n_silent} silent)")
    print(f"  meta       -> {meta_path}")
    print(f"  silence embedding strategy: {args.silence}")
 
 
if __name__ == "__main__":
    main()


