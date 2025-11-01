import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Default hyperparameters
EMBED_DIM = 100
WINDOW = 5
SEED = 42

try:
    import importlib
    gensim_models = importlib.import_module("gensim.models")
    Word2Vec = getattr(gensim_models, "Word2Vec")
    USE_GENSIM = True
except Exception:
    USE_GENSIM = False

# Optional embedded fallback corpus (leave empty if no local file)
EMBEDDED_CORPUS = ""


def load_corpus(base: Path) -> str:
    p = base / "corpus.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return EMBEDDED_CORPUS


def sentence_tokenize(text: str) -> List[str]:
    # Simple split on punctuation/newlines
    return [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]


def word_tokenize(sent: str) -> List[str]:
    # Keep alphabetic, lowercase
    return re.findall(r"[a-z]+", sent.lower())


def build_sentences(text: str) -> List[List[str]]:
    return [word_tokenize(s) for s in sentence_tokenize(text)]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u) + 1e-12
    nv = np.linalg.norm(v) + 1e-12
    return float(np.dot(u, v) / (nu * nv))


def train_gensim_w2v(sents: List[List[str]], dim: int = EMBED_DIM, window: int = WINDOW):
    model = Word2Vec(
        sentences=sents,
        vector_size=dim,
        window=window,
        min_count=1,
        workers=1,
        sg=1,
        seed=SEED,
        epochs=200,
    )
    vocab = list(model.wv.key_to_index.keys())
    vecs = {w: model.wv[w] for w in vocab}
    return vecs


def train_ppmi_svd(sents: List[List[str]], dim: int = EMBED_DIM, window: int = WINDOW):
    # Build vocabulary
    vocab: Dict[str, int] = {}
    for sent in sents:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)
    cooc = np.zeros((V, V), dtype=np.float64)

    # Co-occurrence within symmetric window
    for sent in sents:
        n = len(sent)
        idxs = [vocab[w] for w in sent]
        for i, wi in enumerate(idxs):
            left = max(0, i - window)
            right = min(n, i + window + 1)
            for j in range(left, right):
                if j == i:
                    continue
                wj = idxs[j]
                cooc[wi, wj] += 1.0

    sum_total = cooc.sum()
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)

    # PPMI
    with np.errstate(divide="ignore"):
        PMI = np.log((cooc * sum_total + 1e-9) / (row_sum @ col_sum + 1e-9))
    PPMI = np.maximum(PMI, 0.0)

    # SVD
    U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    Uk = U[:, :dim]
    Sk = S[:dim]
    E = Uk * np.sqrt(Sk + 1e-9)

    inv_vocab = {i: w for w, i in vocab.items()}
    vecs = {inv_vocab[i]: E[i] for i in range(V)}
    return vecs


def most_similar(vecs: Dict[str, np.ndarray], query: np.ndarray, exclude: set, topn: int = 5):
    scores = []
    for w, v in vecs.items():
        if w in exclude:
            continue
        scores.append((w, cosine(query, v)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topn]


def solve_analogy(vecs: Dict[str, np.ndarray], a: str, b: str, c: str) -> str:
    for token in (a, b, c):
        if token not in vecs:
            raise KeyError(f"Token missing from vocab: {token}")
    query = vecs[a] - vecs[b] + vecs[c]
    exclude = {a, b, c}
    best = most_similar(vecs, query, exclude, topn=1)
    return best[0][0]


def main():
    base = Path(__file__).parent
    text = load_corpus(base)
    sents = build_sentences(text)

    if USE_GENSIM:
        vecs = train_gensim_w2v(sents)
        backend = "gensim"
    else:
        vecs = train_ppmi_svd(sents)
        backend = "ppmi-svd"

    # Required tokens (lowercase)
    a1, b1, c1 = "doctors", "medicine", "law"
    a2, b2, c2 = "teachers", "schools", "hospitals"

    try:
        d1 = solve_analogy(vecs, a1, b1, c1)
        d2 = solve_analogy(vecs, a2, b2, c2)
    except KeyError as e:
        print(f"Vocabulary error: {e}")
        return 1

    flag = f"SIGMOID_{d1.upper()}_{d2.upper()}"
    out = {
        "analogy_1": f"{a1} - {b1} + {c1} = {d1}",
        "analogy_2": f"{a2} - {b2} + {c2} = {d2}",
        "flag": flag,
    }
    print(json.dumps(out, indent=2))
    print(flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())