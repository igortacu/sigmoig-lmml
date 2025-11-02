# Mexico — Word Embeddings Analogy Challenge

**Difficulty:** Medium

## 📖 Task Overview

This challenge tests your understanding of word embeddings and vector space semantics. You'll train word vectors on a provided corpus and solve word analogies of the form:

**A is to B as C is to ?**

Using vector arithmetic: `vec(A) - vec(B) + vec(C) ≈ vec(D)`

## 🎯 Objective

Train word embeddings (Word2Vec or PPMI-SVD) on the provided corpus and solve two word analogies to generate the flag.

## 🚩 FLAG Format

`SIGMOID_{WORD1}_{WORD2}`

Where WORD1 and WORD2 are the solutions to the two analogies (uppercase).

## 📂 Files

### `corpus.txt`
A carefully crafted text corpus (83 lines) containing repetitive sentences about:
- **Doctors** and **medicine**
- **Engineers** and **law**
- **Teachers** and **schools**
- **Athletes** and **hospitals**

The corpus is designed to create semantic relationships through co-occurrence patterns.

### `vector.py`
Complete word embedding trainer and analogy solver with two backends:

1. **Gensim Word2Vec** (preferred if gensim is installed)
2. **PPMI-SVD** (fallback - no external dependencies)

## 🧠 How Word Embeddings Work

### The Core Idea

Words that appear in similar contexts should have similar vector representations. The corpus repeatedly associates:
- doctors ↔ medicine
- engineers ↔ law  
- teachers ↔ schools
- athletes ↔ hospitals

### Vector Arithmetic for Analogies

To solve "A is to B as C is to D":
1. Compute: `query_vector = vec(A) - vec(B) + vec(C)`
2. Find the word D whose vector is most similar (cosine similarity) to query_vector
3. Exclude A, B, C from candidates

**Example:** "king is to man as queen is to ?"
- `vec(king) - vec(man) + vec(queen) ≈ vec(woman)`
- The "is to" relationship is captured by the vector difference

## 🔧 Technical Details

### Hyperparameters
```python
EMBED_DIM = 100    # Vector dimensionality
WINDOW = 5         # Context window size
SEED = 42          # Random seed for reproducibility
```

### Word2Vec (Gensim Backend)
- **Algorithm:** Skip-gram (`sg=1`)
- **Training:** 200 epochs
- **Window:** 5-word context
- **Min count:** 1 (keep all words)

Captures semantic relationships through neural prediction of context words.

### PPMI-SVD (Fallback Backend)
1. **Co-occurrence matrix:** Count word pairs within window
2. **PPMI:** Positive Pointwise Mutual Information
   - `PMI(w1, w2) = log(P(w1, w2) / (P(w1) × P(w2)))`
   - `PPMI = max(0, PMI)` (keep only positive values)
3. **SVD:** Singular Value Decomposition reduces dimensionality
   - Extract top 100 singular vectors
   - Each word gets a 100-dimensional embedding

Both methods create a vector space where semantic relationships become geometric.

## 📋 The Two Analogies

### Analogy 1
```
doctors : medicine :: law : ?
```
- "Doctors are to medicine as law is to what?"
- The relationship is: professional → field of practice

### Analogy 2
```
teachers : schools :: hospitals : ?
```
- "Teachers are to schools as hospitals are to what?"
- The relationship is: professional → workplace

## 🚀 Usage

### Prerequisites
```powershell
# Optional: Install gensim for better results
pip install gensim numpy

# Minimum (uses PPMI-SVD fallback):
pip install numpy
```

### Run the Solver
```powershell
python vector.py
```

### Expected Output
```json
{
  "analogy_1": "doctors - medicine + law = engineers",
  "analogy_2": "teachers - schools + hospitals = athletes",
  "flag": "SIGMOID_ENGINEERS_ATHLETES"
}
SIGMOID_ENGINEERS_ATHLETES
```

## 🔍 How It Works (Step by Step)

1. **Load corpus:** Read `corpus.txt` (83 lines)

2. **Tokenize:**
   - Sentence split on `.!?\n`
   - Word tokenize: lowercase alphabetic only
   - Result: List of tokenized sentences

3. **Train embeddings:**
   - **If gensim available:** Train Word2Vec (200 epochs)
   - **Otherwise:** Build co-occurrence matrix → PPMI → SVD

4. **Solve analogy 1:**
   ```python
   query = vec("doctors") - vec("medicine") + vec("law")
   # Find word closest to query (excluding doctors, medicine, law)
   # Answer: "engineers"
   ```

5. **Solve analogy 2:**
   ```python
   query = vec("teachers") - vec("schools") + vec("hospitals")
   # Find word closest to query (excluding teachers, schools, hospitals)
   # Answer: "athletes"
   ```

6. **Generate flag:** `SIGMOID_{answer1.upper()}_{answer2.upper()}`

## 📊 Why This Corpus Works

The corpus is structured to create strong associations:

### Pattern 1: Professional → Practice Area
```
"Doctors practice medicine..."
"Engineers follow law..."
"Medicine forms the foundation of every doctor's expertise..."
"Law provides the essential framework that guides all engineering..."
```

This teaches: `vec(doctors) - vec(medicine) ≈ vec(engineers) - vec(law)`

### Pattern 2: Professional → Workplace
```
"Teachers work in schools..."
"Athletes recover in hospitals..."
"Schools provide the environment where teachers develop..."
"Hospitals offer the medical expertise athletes need..."
```

This teaches: `vec(teachers) - vec(schools) ≈ vec(athletes) - vec(hospitals)`

### Key Technique
Repeated co-occurrence strengthens vector associations. The corpus mentions each pair together 20+ times in similar syntactic contexts.

## 🧪 Debugging & Validation

### Check if words are in vocabulary:
```python
import vector
sents = vector.build_sentences(vector.load_corpus())
vecs = vector.train_gensim_w2v(sents)  # or train_ppmi_svd
print("Vocab size:", len(vecs))
print("Has 'doctors':", "doctors" in vecs)
```

### Test cosine similarity:
```python
print("doctors vs medicine:", vector.cosine(vecs["doctors"], vecs["medicine"]))
print("doctors vs engineers:", vector.cosine(vecs["doctors"], vecs["engineers"]))
```

You should see higher similarity between semantically related words.

### Manual analogy test:
```python
result = vector.solve_analogy(vecs, "doctors", "medicine", "law")
print(f"doctors - medicine + law = {result}")
```

## 💡 Common Issues & Solutions

**Problem:** "Token missing from vocab"
- **Cause:** Word not in corpus
- **Solution:** Check spelling, ensure lowercase, verify corpus.txt exists

**Problem:** Wrong analogy answer
- **Cause:** Insufficient training or corpus doesn't establish relationship
- **Solution:** 
  - Install gensim for better embeddings
  - Check corpus.txt wasn't modified
  - Increase EMBED_DIM or training epochs

**Problem:** Different results each run
- **Cause:** Random initialization (less common with SEED=42)
- **Solution:** Fixed seed should stabilize results; gensim backend more stable

## 📚 Background Concepts

### Word2Vec Skip-gram
Given a target word, predicts context words within window. Example:
```
Sentence: "doctors practice medicine daily"
Window: 2
Training pairs:
  practice → doctors
  practice → medicine
  practice → daily
```

The model learns to map words with similar contexts to nearby vectors.

### PPMI (Positive Pointwise Mutual Information)
Measures how much more often two words co-occur than expected by chance:
- High PPMI → words appear together more than random
- Zero PPMI → no association
- Negative PMI → clipped to zero (PPMI)

### SVD (Singular Value Decomposition)
Reduces sparse high-dimensional PPMI matrix to dense low-dimensional embeddings while preserving most variance.

## 🎓 Learning Outcomes

- Understanding distributional semantics ("You shall know a word by the company it keeps")
- Vector space representations of meaning
- Analogy solving through vector arithmetic
- Word2Vec and count-based embedding methods
- Cosine similarity for semantic comparison

## 🎁 Output Files

- **None** (flag printed to stdout)
- Optional: Add code to save embeddings as JSON/numpy if needed

## ⚙️ Customization

### Change embeddings dimension:
```python
EMBED_DIM = 200  # More dimensions = more nuanced representations
```

### Adjust context window:
```python
WINDOW = 3  # Smaller = focuses on immediate context
WINDOW = 10 # Larger = captures broader associations
```

### Use different corpus:
Replace `corpus.txt` with your own text. Ensure it contains the target words and establishes their relationships through repetition and context.

## 🏆 Success Criteria

When you run `python vector.py`, you should see:
1. JSON output with both analogy solutions
2. Flag in format `SIGMOID_WORD1_WORD2`
3. No errors about missing vocabulary

Submit the flag to complete the challenge!

---

**Note:** This challenge demonstrates how semantic relationships emerge from statistical patterns in text, forming the foundation of modern NLP systems like word embeddings, which power everything from search engines to language models.
