#!/usr/bin/env python3
import os
import re
import time
import json
from faster_whisper import WhisperModel

CHUNK_DIR = "chunks"
FAST_MODEL = "tiny.en"   # small, fast, good for detection
SLOW_MODEL = "medium"    # more accurate, used only on hits

# patterns for detection
PAT_NUM = re.compile(
    r"The\s+(\d+)(st|nd|rd|th)\s+letter\s+(?:in|of|and)\s+keyword\s+is\s+(.+)",
    re.IGNORECASE,
)
PAT_WORD = re.compile(
    r"The\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+letter\s+(?:in|of|and)\s+keyword\s+is\s+(.+)",
    re.IGNORECASE,
)
ORD_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

def extract_letter(rest: str):
    rest = rest.strip()
    # "D-Delta", "D, Delta", "D – Delta"
    m = re.match(r"([A-Z])\s*[-,–]\s*", rest)
    if m:
        return m.group(1)
    # "D delta"
    m = re.match(r"([A-Z])\b", rest)
    if m:
        return m.group(1)
    # last resort
    m = re.search(r"\b([A-Z])\b", rest)
    if m:
        return m.group(1)
    return None

def detect_hits_in_segments(fname, segments, pass_id):
    hits = []
    seg_idx = 0
    for seg in segments:
        seg_idx += 1
        text = seg.text.strip()
        print(f"    [{pass_id} {fname} seg{seg_idx:03d} {seg.start:6.2f}-{seg.end:6.2f}] {text}")

        idx = None
        letter = None

        m1 = PAT_NUM.search(text)
        if m1:
            idx = int(m1.group(1))
            rest = m1.group(3)
            letter = extract_letter(rest)
        else:
            m2 = PAT_WORD.search(text)
            if m2:
                idx = ORD_MAP.get(m2.group(1).lower())
                rest = m2.group(2)
                letter = extract_letter(rest)

        # also catch your ASR oddity: "password and keyword is G-Golf."
        if "keyword is" in text.lower() and idx is None:
            # try to guess number from context? leave idx=None, still store
            letter = extract_letter(text.split("keyword is", 1)[1])
            # mark index unknown
            if letter:
                hits.append({
                    "file": fname,
                    "index": None,
                    "letter": letter,
                    "raw": text,
                    "start": seg.start,
                    "end": seg.end,
                })
                print(f"      -> HIT (no index) letter={letter} text={text!r}")
                continue

        if idx is not None and letter is not None:
            hit = {
                "file": fname,
                "index": idx,
                "letter": letter,
                "raw": text,
                "start": seg.start,
                "end": seg.end,
            }
            hits.append(hit)
            print(f"      -> HIT idx={idx} letter={letter} text={text!r}")
    return hits

def main():
    if not os.path.isdir(CHUNK_DIR):
        print("folder 'chunks' missing")
        return

    files = sorted(f for f in os.listdir(CHUNK_DIR) if f.endswith(".mp3"))
    total = len(files)
    print(f"found {total} audio chunks")

    # ---------- PASS 1: fast detector ----------
    print("\n=== PASS 1: fast detection (tiny.en) ===")
    fast_model = WhisperModel(
        FAST_MODEL,
        device="cpu",
        compute_type="int8",   # faster on CPU
    )

    rough_hits = []     # chunks to re-run
    all_fast_hits = []  # for debug

    for i, fname in enumerate(files, 1):
        path = os.path.join(CHUNK_DIR, fname)
        t0 = time.time()
        print(f"[{i}/{total}] {fname} (fast)")

        segments, _ = fast_model.transcribe(
            path,
            language="en",
            beam_size=1,           # speed
            vad_filter=True,
        )
        hits = detect_hits_in_segments(fname, segments, pass_id="FAST")
        dt = time.time() - t0
        print(f"  -> {len(hits)} hits, time={dt:.2f}s")

        if hits:
            rough_hits.append(fname)
            all_fast_hits.extend(hits)

    print("\nFAST PASS hits:")
    print(json.dumps(all_fast_hits, indent=2))

    if not rough_hits:
        print("\nno hits in pass 1. stop.")
        return

    # ---------- PASS 2: accurate on hit files ----------
    print("\n=== PASS 2: accurate on hit files (medium) ===")
    slow_model = WhisperModel(
        SLOW_MODEL,
        device="cpu",
        compute_type="float32",   # accurate
    )

    final_hits = []
    for fname in rough_hits:
        path = os.path.join(CHUNK_DIR, fname)
        print(f"\nre-scan {fname} (slow)")
        t0 = time.time()
        segments, _ = slow_model.transcribe(
            path,
            language="en",
            beam_size=5,
            best_of=5,
            vad_filter=True,
        )
        hits = detect_hits_in_segments(fname, segments, pass_id="SLOW")
        dt = time.time() - t0
        print(f"  -> {len(hits)} refined hits, time={dt:.2f}s")
        final_hits.extend(hits)

    # ---------- assemble keywords ----------
    # group by index; sort; show gaps
    final_hits.sort(key=lambda x: (x["index"] is None, x["index"]))

    print("\n=== FINAL HITS (sorted) ===")
    print(json.dumps(final_hits, indent=2))

    # guess keyword from known indexes
    letters_by_idx = {}
    for h in final_hits:
        if h["index"] is not None:
            letters_by_idx[h["index"]] = h["letter"]

    if letters_by_idx:
        max_idx = max(letters_by_idx)
        keyword = []
        for i in range(1, max_idx + 1):
            ch = letters_by_idx.get(i, "?")
            keyword.append(ch)
        print("\nKEYWORD (best guess):", "".join(keyword))
    else:
        print("\nno indexed letters found")

    # write to file for manual search
    with open("hits.json", "w", encoding="utf-8") as f:
        json.dump(final_hits, f, indent=2)

if __name__ == "__main__":
    main()
