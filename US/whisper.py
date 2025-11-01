import os
import re
import json
from faster_whisper import WhisperModel

CHUNK_DIR = "chunks"

# run medium only on the files we know have something
FILES = [
    "out_000.mp3",  # 1st -> D
    "out_008.mp3",  # 2nd -> I  (small saw it, medium missed it, so we keep small's value)
    "out_009.mp3",  # 3rd -> O
    "out_025.mp3",  # 1st of keyword #2 (in small run)
    "out_026.mp3",  # 5th -> G
    "out_062.mp3",  # 6th -> I
]

model = WhisperModel("medium")

# accept "is" with comma or colon
re_main = re.compile(
    r"The\s+(?P<ord>(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))\s+letter"
    r"(?:\s+(?:in|of|and)\s+keyword)?\s+is[ ,:]+(?P<rest>.+)",
    re.IGNORECASE,
)

# accept the warped line: "The password and keyword is g-golf."
re_password = re.compile(
    r"The\s+password\s+(?:in|of|and)\s+keyword\s+is[ ,:]+(?P<rest>.+)",
    re.IGNORECASE,
)

ord_map = {
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
    # D-Delta / D, Delta / D – Delta
    m = re.match(r"([A-Z])\s*[-,–]\s*", rest)
    if m:
        return m.group(1).upper()
    # Starts with single capital
    m = re.match(r"([A-Z])\b", rest)
    if m:
        return m.group(1).upper()
    # First standalone capital
    m = re.search(r"\b([A-Z])\b", rest)
    if m:
        return m.group(1).upper()
    return None

results = []

for fname in FILES:
    path = os.path.join(CHUNK_DIR, fname)
    print(f"=== {fname} ===")
    segments, _ = model.transcribe(path, language="en")
    for seg in segments:
        text = seg.text.strip()
        print(f"  [{seg.start:6.2f}-{seg.end:6.2f}] {text!r}")

        m = re_main.search(text)
        if m:
            ord_raw = m.group("ord")
            rest = m.group("rest")
            if ord_raw.isdigit():
                idx = int(ord_raw)
            else:
                idx = ord_map[ord_raw.lower()]
            letter = extract_letter(rest)
            if letter:
                print(f"    -> HIT idx={idx} letter={letter} text={text!r}")
                results.append({
                    "file": fname,
                    "index": idx,
                    "letter": letter,
                    "raw": text,
                })
            continue  # processed

        m = re_password.search(text)
        if m:
            rest = m.group("rest")
            letter = extract_letter(rest)
            if letter:
                # treat "password" as 5th
                idx = 5
                print(f"    -> HIT idx={idx} letter={letter} text={text!r}")
                results.append({
                    "file": fname,
                    "index": idx,
                    "letter": letter,
                    "raw": text,
                })

# merge with the small-model hit:
# small said: out_008 -> "The second letter in keyword is I India."
results.append({
    "file": "out_008.mp3",
    "index": 2,
    "letter": "I",
    "raw": "The second letter in keyword is I India.  (from small run)"
})

results.sort(key=lambda x: (x["index"], x["file"]))

keyword1 = "".join([r["letter"] for r in results if r["index"] in (1,2,3)])
print("\nKEYWORD #1:", keyword1)

print("\nALL HITS:")
print(json.dumps(results, indent=2))
