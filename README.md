# Lecture-Video-Classification
ASR-Derived Transcript Features as Complementary Signal for Pose-Based Speaker Action Classification in Lecture 

**Kritika Bhat** 
2178208
DePaul University 
CSC 575  
Supervisor: Professor Kenny Davila Castellanos

---

## What This Is

A multimodal text feature pipeline that complements the 2S-AGCN pose classifier (Xu et al., 2021) for speaker action classification in the LectureMath dataset. Whisper transcribes lecture audio, words are aligned to action segments, and SBERT embeddings feed three classifiers. The MLP achieves macro-F1 of 0.157 , and crucially, its errors are complementary to the pose model's, motivating a weighted late-fusion architecture.

---

## Setup

```bash
pip install faster-whisper sentence-transformers scikit-learn numpy pandas
# also requires ffmpeg
```

---

## Run the Pipeline (in order)

```bash
# 1. Preprocess annotations + transcribe + align
python batch_preprocess.py

# 2. Pool per-video CSVs into one file
python pool_test.py

# 3. Encode transcripts with SBERT
python encode_sbert.py output_test/all_test_aligned_contained.csv embeddings_test/contained

# 4. Train classifiers + save probability files
python predict_text_prob.py \
    --train-embeddings embeddings_train/contained_embeddings.npy \
    --train-mask       embeddings_train/contained_mask.npy \
    --train-meta       embeddings_train/contained_meta.csv \
    --test-embeddings  embeddings_test/contained_embeddings.npy \
    --test-mask        embeddings_test/contained_mask.npy \
    --test-meta        embeddings_test/contained_meta.csv \
    --out-dir          results/contained_final

# 5. Build enriched dataset (merges probs + metadata)
python enriched_dataset.py \
    --aligned    output_test/all_test_aligned_contained.csv \
    --svm-probs  results/contained_final/svm_text_probs.npy \
    --mlp-probs  results/contained_final/mlp_text_probs.npy \
    --rf-probs   results/contained_final/rf_text_probs.npy \
    --meta       embeddings_test/contained_meta.csv \
    --out        results/enriched_test_dataset.csv
```

---

## Results

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| Majority Baseline | 46.5% | 0.090 |
| SVM (RBF) | 46.3% | 0.138 |
| Random Forest | 45.6% | 0.119 |
| **MLP** | **46.8%** | **0.157** |

Contained alignment outperforms overlap for SVM and RF; equivalent for MLP.  
Fusion evaluation pending access to 2S-AGCN pose score files. Proposed fusion weight: **α = 0.2**.

---

## Dataset

LectureMath — 34 videos, 16.7 hours, 8 action classes, 17 train / 17 test videos (speaker-based split).  
Training: 5,659 canonical segments · Test: 5,994 canonical segments.

---

## Key Files

| File | Purpose |
|---|---|
| `batch_pipeline.py` | Annotation preprocessing, audio transcription, alignment |
| `pool_test.py` | Pool per-video CSVs for encoding |
| `encode_sbert.py` | SBERT encoding (all-MiniLM-L6-v2, 384-dim) |
| `predict_text_prob.py` | Train SVM / MLP / RF, save `.npy` probability files |
| `enriched_dataset.py` | Merge probabilities with segment metadata |
| `results/enriched_test_dataset.csv` | Final dataset: 6,090 rows × 43 columns |
