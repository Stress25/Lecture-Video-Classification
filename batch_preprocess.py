"""
batch_pipeline.py
Runs the full preprocessing pipeline on multiple lecture videos:
  1. Parse XML annotations → labelled_segments.csv
  2. Extract audio (ffmpeg) → .wav
  3. Transcribe (faster-whisper) → whisper.json
  4. Align words to segments (both modes) → aligned_contained.csv, aligned_overlap.csv


Output structure:
  output_dir/
    <video_id>/
      labelled_segments.csv
      <video_id>.wav
      whisper.json
      aligned_contained.csv
      aligned_overlap.csv
      pipeline.log             per-video log

Usage:
  python batch_pipeline.py \\
      --video-dir /path/to/videos \\
      --xml-dir   /path/to/xmls \\
      --out-dir   /path/to/output \\
      --model     medium \\
      --ids 00000_000_001 00000_001_001 00000_002_001 00000_003_001 00000_004_001
"""
import argparse
import os
import subprocess
import sys
import time
import traceback

# Add the folder containing your scripts to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocess_annotations import preprocess_annotations
from align_Segments import (
    load_segments, load_words, assign_contained,
    assign_overlap, attach_transcripts, write_csv, print_diagnostics,
)


def log(msg, logfile=None):
    print(msg)
    if logfile:
        with open(logfile, "a") as f:
            f.write(msg + "\n")


def step_annotate(xml_path, out_csv, logfile):
    log(f"  [1/4] Parsing annotations: {xml_path}", logfile)
    t = time.time()
    segs = preprocess_annotations(xml_path, out_csv)
    log(f"        Done — {len(segs)} segments  ({time.time()-t:.1f}s)", logfile)
    return len(segs)


def step_audio(mp4_path, wav_path, logfile):
    log(f"  [2/4] Extracting audio → {wav_path}", logfile)
    t = time.time()
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", wav_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-500:]}")
    size_mb = os.path.getsize(wav_path) / 1e6
    log(f"        Done — {size_mb:.1f} MB  ({time.time()-t:.1f}s)", logfile)


def step_transcribe(wav_path, json_path, model_size, device, logfile):
    log(f"  [3/4] Transcribing with Whisper ({model_size}) ...", logfile)
    t = time.time()

    # Run transcription as a subprocess to avoid Python 3.12 + faster-whisper
    # segfault on Mac Apple Silicon when importing directly.
    transcribe_script = os.path.join(SCRIPT_DIR, "transcribe_lecture.py")
    cmd = [
        sys.executable, transcribe_script,
        wav_path, json_path,
        "--model", model_size,
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Whisper transcription failed:\n{result.stderr[-1000:]}"
        )

    # Count words from the output JSON
    import json as _json
    with open(json_path) as f:
        data = _json.load(f)
    n_words = sum(len(s.get("words", [])) for s in data.get("segments", []))
    n_segs = len(data.get("segments", []))

    elapsed = time.time() - t
    rtf = elapsed / data.get("duration_sec", 1)
    log(f"        Done — {n_segs} whisper segs, "
        f"{n_words} words, RTF={rtf:.2f}x  ({elapsed:.1f}s)", logfile)
    return n_words


def step_align(segments_csv, whisper_json, out_dir, logfile):
    log(f"  [4/4] Aligning words to segments (both modes) ...", logfile)
    t = time.time()

    segments = load_segments(segments_csv)
    words, audio_dur, language = load_words(whisper_json)

    for mode, assign_fn, out_name in [
        ("contained", assign_contained, "aligned_contained.csv"),
        ("overlap",   assign_overlap,   "aligned_overlap.csv"),
    ]:
        buckets = assign_fn(segments, words)
        segs_with_text = attach_transcripts(
            [dict(s) for s in segments], buckets
        )
        out_path = os.path.join(out_dir, out_name)
        write_csv(segs_with_text, out_path)

        n_silent = sum(1 for s in segs_with_text if s["is_silence"] == 1)
        log(f"        [{mode}] silent={n_silent}/{len(segments)} "
            f"-> {out_path}", logfile)

    log(f"        Done  ({time.time()-t:.1f}s)", logfile)


def process_one(video_id, mp4_path, xml_path, out_dir,
                model_size, device, keep_wav):
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, "pipeline.log")

    # Clear old log for this run
    with open(logfile, "w") as f:
        f.write(f"=== {video_id}  started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    wav_path = os.path.join(out_dir, f"{video_id}.wav")
    segments_csv = os.path.join(out_dir, "labelled_segments.csv")
    whisper_json = os.path.join(out_dir, "whisper.json")

    t_total = time.time()
    try:
        n_segs = step_annotate(xml_path, segments_csv, logfile)
        step_audio(mp4_path, wav_path, logfile)
        n_words = step_transcribe(wav_path, whisper_json,
                                  model_size, device, logfile)
        step_align(segments_csv, whisper_json, out_dir, logfile)

        if not keep_wav and os.path.exists(wav_path):
            os.remove(wav_path)
            log(f"  wav removed (--no-keep-wav)", logfile)

        elapsed = time.time() - t_total
        log(f"\n  DONE in {elapsed:.1f}s  "
            f"(segments={n_segs}, words={n_words})", logfile)
        return True, n_segs, n_words

    except Exception:
        msg = traceback.format_exc()
        log(f"\n  Failed:\n{msg}", logfile)
        return False, 0, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True,
                   help="Folder containing .mp4 files")
    p.add_argument("--xml-dir", required=True,
                   help="Folder containing .xml annotation files "
                        "(can be same as --video-dir)")
    p.add_argument("--out-dir", required=True,
                   help="Root output folder; one sub-folder per video")
    p.add_argument("--ids", nargs="+", required=True,
                   help="Video IDs to process, e.g. 00000_000_001 00000_001_001")
    p.add_argument("--model", default="medium",
                   choices=["tiny", "base", "small", "medium",
                            "large-v2", "large-v3"])
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--keep-wav", action="store_true",
                   help="Keep the extracted .wav file (deleted by default "
                        "to save disk space)")
    args = p.parse_args()

    # Resolve device
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    print("=" * 60)
    print(f"Batch pipeline  —  {len(args.ids)} videos")
    print(f"  Whisper model : {args.model} on {device}")
    print(f"  Output root   : {args.out_dir}")
    print("=" * 60)

    summary = []
    for i, vid_id in enumerate(args.ids):
        print(f"\n[{i+1}/{len(args.ids)}] {vid_id}")

        mp4 = os.path.join(args.video_dir, vid_id.replace("LectureMath_", "") + ".mp4")
        xml = os.path.join(args.xml_dir, vid_id + ".xml")
        out = os.path.join(args.out_dir,    vid_id)

        # Check files exist before starting
        missing = [p for p in [mp4, xml] if not os.path.exists(p)]
        if missing:
            print(f"  Skipped: files not found: {missing}")
            summary.append((vid_id, "skipped", 0, 0))
            continue

        ok, n_segs, n_words = process_one(
            vid_id, mp4, xml, out,
            model_size=args.model,
            device=device,
            keep_wav=args.keep_wav,
        )
        summary.append((vid_id, "ok" if ok else "failed", n_segs, n_words))

    # Final summary table
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"{'video_id':<20} {'status':>8} {'segments':>9} {'words':>7}")
    print("-" * 50)
    total_segs = total_words = 0
    for vid_id, status, n_segs, n_words in summary:
        print(f"{vid_id:<20} {status:>8} {n_segs:>9} {n_words:>7}")
        total_segs += n_segs
        total_words += n_words
    print("-" * 50)
    print(f"{'TOTAL':<20} {'':>8} {total_segs:>9} {total_words:>7}")

    failed = [s for s in summary if s[1] == "failed"]
    if failed:
        print(f"\nFailed videos: {[s[0] for s in failed]}")
        print("Check pipeline.log inside each failed video's output folder.")


if __name__ == "__main__":
    main()