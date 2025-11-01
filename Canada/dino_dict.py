import csv
import re
from collections import defaultdict, Counter

TRAIN_PATH = "participant_input/dinosaur_dataset.csv"
TEST_PATH = "participant_input/test-input.csv"
OUT_PATH = "output.csv"

# ---------------- helpers ----------------

PUNCTS = ["?", ".", ",", "!", ";", ":"]

def split_like_dino(s: str):
    # make punctuation separate tokens
    for p in PUNCTS:
        s = s.replace(p, f" {p}")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")

def join_dino(tokens):
    # dinosaur examples use space before punctuation
    out = []
    for t in tokens:
        out.append(t)
    return " ".join(out).strip()

# ---------------- learn dictionary ----------------

eng2dino_counts = defaultdict(Counter)

with open(TRAIN_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eng = row["english"].strip()
        dino = row["dinosaur"].strip()

        e_tokens = split_like_dino(eng)
        d_tokens = split_like_dino(dino)

        # skip weird rows where lengths differ a lot
        if len(e_tokens) != len(d_tokens):
            # try a small fix: many english tokens have punctuation attached
            # but dinosaur already had spaces
            # we already split, so if still unequal, better skip this row
            continue

        for e_tok, d_tok in zip(e_tokens, d_tokens):
            eng2dino_counts[e_tok][d_tok] += 1

# choose the most frequent dino token for each english token
eng2dino = {}
for e_tok, counter in eng2dino_counts.items():
    d_tok, _ = counter.most_common(1)[0]
    eng2dino[e_tok] = d_tok

# ---------------- translate test ----------------

outputs = []
with open(TEST_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sent = row["sentence"].strip()
        e_tokens = split_like_dino(sent)
        d_tokens = []
        for tok in e_tokens:
            if tok in eng2dino:
                d_tokens.append(eng2dino[tok])
            else:
                # unseen token -> leave as is
                # better to keep punctuation exact
                d_tokens.append(tok)
        dino_sent = join_dino(d_tokens)
        outputs.append(dino_sent)

# ---------------- write output.csv ----------------

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence"])
    for line in outputs:
        writer.writerow([line])

print("wrote", OUT_PATH)
print("rows:", len(outputs))
