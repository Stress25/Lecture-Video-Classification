"""
align segments with transcript.py
Joins the action-segment CSV (from preprocess_annotations.py) with the
Whisper word-timestamp JSON (from transcribe_lecture.py), producing an
augmented CSV with a `transcript` column per segment.

Two word-to-segment assignment strategies (switchable with --mode):
 contained  : each word is assigned to the SINGLE segment that holds the
               majority of the word's time duration (mostly contained).
               No word appears in more than one segment.
 
  overlap    : each word is assigned to EVERY segment whose time range it
               overlaps. A word straddling a boundary appears in both
               segments. Words can therefore be counted more than once.
 
Segments with no words get the transcript text "[silence]" so that the
downstream SBERT step receives a real, consistent token instead of an
empty string (which would produce a meaningless embedding).
"""

import argparse
import csv
import json
from collections import Counter

SILENCE_TOKEN = "[silence]"

def load_words(whisper_json_path):
    # flatten the whisper json into one time sorted list of words dicts
    with open(whisper_json_path) as f:
        data = json.load(f)

    words = []
    for seg in data.get("segments",[]):
        for w in seg.get("words",[]):
            start = float(w["start_sec"])
            end = float(w["end_sec"])
            words.append({
                "word": w["word"],
                "start": start,
                "end": end,
                "dur": max(end - start, 1e-9),
                "prob":float(w.get("prob",1.0)),
            })
    words.sort(key=lambda x:x["start"])
    return words,data.get("duration_sec"),data.get("language")


def load_segments(csv_path):
    # load action-segment CSV produced by preprocess_annotations.py
    segments = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            segments.append({
                "segment_id": int(row["segment_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "start_time_ms": float(row["start_time_ms"]),
                "end_time_ms": float(row["end_time_ms"]),
                "start_time_sec": float(row["start_time_sec"]),
                "end_time_sec": float(row["end_time_sec"]),
                "duration_sec": float(row["duration_sec"]),
                "label": row["label"],
            })
    return segments

def overlap_seconds(w_start,w_end,s_start,s_end):
    return max(0.0,min(w_end,s_end) - max(w_start, s_start))

def assign_contained(segments,words):
    """
    Each word goes to the single segment holding the MAJORITY of its
    duration. We bucket each word once by finding its best-overlap segment.
 
    Both lists are time-sorted, so we sweep with a moving window of
    candidate segments rather than re-scanning all segments per word.
    """
    buckets = {s["segment_id"]: [] for s in segments}

    seg_lo = 0
    n_seg = len(segments)

    for w in words:
        while seg_lo < n_seg and segments[seg_lo]["end_time_sec"] <= w["start"]:
            seg_lo += 1

    
         # overlap
        best_seg = None
        best_overlap = 0.0

        j = seg_lo
        while j < n_seg and segments[j]["start_time_sec"] < w["end"]:
            ov = overlap_seconds(
                w["start"], w["end"],
                segments[j]["start_time_sec"], segments[j]["end_time_sec"],
            )
            if ov > best_overlap:
                best_overlap = ov
                best_seg = segments[j]["segment_id"]
            j += 1

        # if a word overlaps nothing,drop it
        if best_seg is not None and best_overlap > 0:
            buckets[best_seg].append(w)
    
    return buckets

def attach_transcripts(segments,buckets):
    "Turn each word bucket into transcript/word count/mean prob fields"
    for seg in segments:
        bucket = buckets[seg["segment_id"]]
        if bucket:
            seg["transcript"] = " ".join(w["word"] for w in bucket).strip()
            seg["word_count"] = len(bucket)
            seg["mean_word_prob"] = round(
                sum(w["prob"] for w in bucket) / len(bucket), 3
            )
            seg["is_silence"] = 0
        else:
            # Map silence to an explicit token so SBERT gets a real input.
            seg["transcript"] = SILENCE_TOKEN
            seg["word_count"] = 0
            seg["mean_word_prob"] = ""
            seg["is_silence"] = 1

    return segments

def assign_overlap(segments,words):
    """
    Each word goes to EVERY segment it overlaps. A boundary-straddling
    word lands in both neighbouring segments.
    """
    buckets = {s["segment_id"]:[] for s in segments}

    seg_lo = 0
    n_seg = len(segments)

    for w in words:
        while seg_lo < n_seg and segments[seg_lo]["end_time_sec"] <= w["start"]:
            seg_lo += 1
        
        j = seg_lo
        while j < n_seg and segments[j]["start_time_sec"] < w["end"]:
            ov = overlap_seconds(
                w["start"], w["end"],
                segments[j]["start_time_sec"], segments[j]["end_time_sec"],
            )
            if ov > 0:
                buckets[segments[j]["segment_id"]].append(w)
            j += 1
 
    return buckets

def write_csv(segments, output_path):
    field_names = [
        "segment_id", "start_frame", "end_frame",
        "start_time_ms", "end_time_ms",
        "start_time_sec", "end_time_sec",
        "duration_sec", "label",
        "word_count", "mean_word_prob", "is_silence", "transcript",
    ]
    with open(output_path,"w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=field_names)
        writer.writeheader()
        for seg in segments:
            row = {k:seg.get(k,"") for k in field_names}
            row["start_time_ms"] = f"{seg['start_time_ms']:.2f}"
            row["end_time_ms"] = f"{seg['end_time_ms']:.2f}"
            row["start_time_sec"] = f"{seg['start_time_sec']:.3f}"
            row["end_time_sec"] = f"{seg['end_time_sec']:.3f}"
            row["duration_sec"] = f"{seg['duration_sec']:.3f}"
            writer.writerow(row)


def print_diagnostics(segments,words,mode,audio_duration,language):

    print("="*60)
    print(f"Aligning whisper words to action segments (mode: {mode})")
    print("="*60)
    print(f"  Action segments  : {len(segments):>6}")
    print(f"  Whisper words    : {len(words):>6}")
    print(f"  Audio duration   : {audio_duration} sec  (language: {language})")

    assigned = sum(s["word_count"] for s in segments)
    print(f" Word slots filled: {assigned:>6}"
          f"({'with duplicates' if mode == 'overlap' else 'no duplicates'})")
    
    label_words = Counter()
    label_segments = Counter()
    for s in segments:
        label_words[s["label"]] += s["word_count"]
        label_segments[s["label"]] += 1
    print()
    print("  Words per label (avg):")
    max_lab = max(len(l) for l in label_segments)
    for lab, n in label_segments.most_common():
        avg = label_words[lab] / n if n else 0
        print(f"    {lab:<{max_lab}}  {label_words[lab]:>5} words / "
              f"{n:>3} segs  ({avg:5.1f} avg)")
 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("segments_csv")
    p.add_argument("whisper_json")
    p.add_argument("output_csv")
    p.add_argument("--mode", default="contained",
                   choices=["contained", "overlap"],
                   help="contained = one segment per word (majority "
                        "duration); overlap = every overlapping segment")
    args = p.parse_args()
 
    segments = load_segments(args.segments_csv)
    words, audio_duration, language = load_words(args.whisper_json)
 
    if args.mode == "contained":
        buckets = assign_contained(segments, words)
    else:
        buckets = assign_overlap(segments, words)
 
    segments = attach_transcripts(segments, buckets)
    print_diagnostics(segments, words, args.mode, audio_duration, language)
    write_csv(segments, args.output_csv)
    print(f"\n  written to: {args.output_csv}")
    
if __name__ == "__main__":
    main()