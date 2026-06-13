import argparse
import json
import os
import sys
import time

from faster_whisper import WhisperModel

def pick_device_and_compute(preferred="auto"):
    # pick the fastest backened available
    if preferred == "cpu":
        return "cpu","int8"
    if preferred == "cuda":
        return "cuda","float16"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda","float16"
    except ImportError:
        pass
    return "cpu","int8"

def transcribe(audio_path,output_path,model_size="medium",device_pref="auto",language="en"):
    device,compute_type = pick_device_and_compute(device_pref)
    print(f"Loading model: {model_size} on {device} ({compute_type})")

    t0 = time.time()
    model = WhisperModel(model_size,device=device,compute_type=compute_type)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print(f"Transcribing: {audio_path}")

    t0 = time.time()
    #  language=language,         # forcing 'en' avoids slow language detection
    #     word_timestamps=True,      # required for per-segment alignment later
    #     vad_filter=True,           # cuts long silences, reduces hallucinations
    #     vad_parameters={"min_silence_duration_ms": 500},
    #     beam_size=5,               # default; lower = faster, slightly less accurate
    #     condition_on_previous_text=True,  # helps with consistent vocab in lectures
    segments_iter , info = model.transcribe(audio_path,language=language,word_timestamps=True,vad_filter=True,vad_parameters={"min_silence_duration_ms":500},beam_size=5,condition_on_previous_text=True)

    out = {
        "audio_path":os.path.abspath(audio_path),
        "model": model_size,
        "language": language,
        "duration_sec": round(info.duration,3),
        "segments": [],
    }

    n_words = 0
    for seg in segments_iter:
        seg_dict = {
            "id": seg.id,
            "start_sec": round(seg.start,3),
            "end_sec": round(seg.end,3),
            "text": seg.text.strip(),
            "words": [],
        }

        if seg.words:
            for w in seg.words:
                seg_dict["words"].append({
                    "word": w.word.strip(),
                    "start_sec": round(w.start,3),
                    "end_sec": round(w.end,3),
                    "prob": round(w.probability,3),
                })
            n_words += len(seg.words)
        out["segments"].append(seg_dict)

        if seg.id % 100 == 0:
            elapsed = time.time() - t0
            print(f"seg {seg.id} t = {seg.end}s")
            print(f" {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    rtf = elapsed / out["duration_sec"]

    print (f" whisper segments: {len(out["segments"])}")
    print(f" total words: {n_words}")
    print(f" Language detected: {info.language}")

    with open(output_path, "w") as f:
        json.dump(out,f,indent=2,ensure_ascii=False)
        print(f"Written to {output_path}")
        return out        
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("output_json")
    p.add_argument("--model",default="medium", choices=["tiny", "base", "small", "medium","large-v2", "large-v3"])
    p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    p.add_argument("--language",default="en")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}")
        sys.exit(1)

    transcribe(args.audio, args.output_json,
               model_size=args.model,
               device_pref=args.device,
               language=args.language)