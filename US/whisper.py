import os, re, json
from faster_whisper import WhisperModel

CHUNK_DIR = "chunks"

# only the files with possible letters
FILES = [
    "out_000.mp3",
    "out_008.mp3",
    "out_009.mp3",
    "out_025.mp3",
    "out_026.mp3",
    "out_062.mp3",
]

# use medium only here
model = WhisperModel("medium")  # will load ~1.5GB, but gives better text

p_num = re.compile(
    r"The\s+(\d+)(st|nd|rd|th)\s+letter(?:\s+(?:in|of|and)\s+keyword)?\s+is\s+(.+)",
    re.IGNORECASE,
)
p_word = re.compile(
    r"The\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+letter(?:\s+(?:in|of|and)\s+keyword)?\s+is\s+(.+)",
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
    # starts with single capital
    m = re.match(r"([A-Z])\b", rest)
    if m:
        return m.group(1).upper()
    # first standalone capital
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

        m1 = p_num.search(text)
        m2 = p_word.search(text)

        idx = None
        letter = None

        if m1:
            idx = int(m1.group(1))
            letter = extract_letter(m1.group(3))
        elif m2:
            idx = ord_map[m2.group(1).lower()]
            letter = extract_letter(m2.group(2))

        if idx is not None and letter is not None:
            print(f"    -> HIT idx={idx} letter={letter} text={text!r}")
            results.append({
                "file": fname,
                "index": idx,
                "letter": letter,
                "raw": text,
            })

results.sort(key=lambda x: x["index"])
flag = "".join(r["letter"] for r in results)

print("\nFLAG:", flag)
print(json.dumps(results, indent=2))
