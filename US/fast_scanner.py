# fast_scan.py
import os, sys, re
from faster_whisper import WhisperModel

CHUNK_DIR = "chunks"

files = sorted(f for f in os.listdir(CHUNK_DIR) if f.endswith(".mp3"))

# tiny, int8 → fastest
model = WhisperModel("tiny", compute_type="int8")

hits = []

for i, fname in enumerate(files, 1):
    path = os.path.join(CHUNK_DIR, fname)
    print(f"[scan {i}/{len(files)}] {fname}", flush=True)

    segments, _ = model.transcribe(
        path,
        language="en",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    for seg in segments:
        text = seg.text.strip()
        if "letter" in text.lower():
            print(f"    -> {fname} {seg.start:.2f}-{seg.end:.2f}s: {text!r}", flush=True)
            hits.append({
                "file": fname,
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })

print("\n=== files with possible letters ===")
seen = set()
for h in hits:
    if h["file"] not in seen:
        print(h["file"])
        seen.add(h["file"])
