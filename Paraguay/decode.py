import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 1. load and sort
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# 2. normalize y
y = (y - y.min()) / (y.max() - y.min() + 1e-9)

def smooth(a, k):
    pad = k // 2
    a = np.pad(a, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="valid")

# 3. bin → same as your plot
N_BINS = 1200
edges = np.linspace(x.min(), x.max(), N_BINS + 1)
bix = np.digitize(x, edges) - 1

bx, by = [], []
for b in range(N_BINS):
    m = bix == b
    if m.any():
        bx.append(x[m].mean())
        by.append(y[m].mean())
bx = np.array(bx)
by = np.array(by)

sig = smooth(by, k=9)

# 4. find local maxima (for estimating grid step)
loc_x = []
loc_y = []
for i in range(1, len(sig) - 1):
    if sig[i] >= sig[i-1] and sig[i] >= sig[i+1]:
        loc_x.append(bx[i])
        loc_y.append(sig[i])
loc_x = np.array(loc_x)
loc_y = np.array(loc_y)

# 5. estimate base step from x-differences (use mode via kmeans)
dx = np.diff(loc_x).reshape(-1, 1)
# remove few big jumps first
dx_small = dx[dx < np.percentile(dx, 90)].reshape(-1, 1)
kdx = KMeans(n_clusters=1, n_init=5, random_state=0).fit(dx_small)
step = float(kdx.cluster_centers_[0, 0])

# 6. build regular grid
x0 = bx.min()
x1 = bx.max()
grid = np.arange(x0, x1 + step, step)

# 7. sample signal on this grid → take max in a small window around each slot
vals = []
half = step * 0.45  # window half-width
for gx in grid:
    m = (bx >= gx - half) & (bx <= gx + half)
    if m.any():
        vals.append(sig[m].max())
    else:
        vals.append(0.0)
vals = np.array(vals)

# 8. cluster heights into 3: gap / dot / dash
vh = vals.reshape(-1, 1)
# keep only nonzero for fitting
nz = vals > 0
fit_vals = vh[nz]

km = KMeans(n_clusters=3, n_init=12, random_state=0).fit(fit_vals)
cent = km.cluster_centers_.ravel()
order = np.argsort(cent)     # 0=gap,1=dot,2=dash
remap = {old: new for new, old in enumerate(order)}

labels = np.full(len(vals), 0, dtype=int)
labels[nz] = [remap[l] for l in km.predict(fit_vals)]

# 9. decode by Morse timing rule on grid cells
# labels: 0=gap cell, 1=dot, 2=dash
MORSE = {
    ".-":"A","-...":"B","-.-.":"C","-..":"D",".":"E","..-.":"F","--.":"G",
    "....":"H","..":"I",".---":"J","-.-":"K",".-..":"L","--":"M","-.":"N",
    "---":"O",".--.":"P","--.-":"Q",".-.":"R","...":"S","-":"T","..-":"U",
    "...-":"V",".--":"W","-..-":"X","-.--":"Y","--..":"Z",
    "-----":"0",".----":"1","..---":"2","...--":"3","....-":"4",
    ".....":"5","-....":"6","--...":"7","---..":"8","----.":"9",
}

symbols = []
for lab in labels:
    if lab == 0:
        symbols.append(" ")     # gap cell
    elif lab == 1:
        symbols.append(".")
    else:
        symbols.append("-")

# 10. convert runs to letters
letters = []
cur = []
gap_count = 0

for s in symbols:
    if s in ".-":
        # if we had gaps before this symbol
        if gap_count >= 7:
            # word gap
            if cur:
                letters.append(''.join(cur))
                cur = []
            letters.append("/")  # word
        elif gap_count >= 3:
            if cur:
                letters.append(''.join(cur))
                cur = []
        gap_count = 0
        cur.append(s)
    else:
        gap_count += 1

# flush
if cur:
    letters.append(''.join(cur))

# 11. map to text
words = []
curw = []
for item in letters:
    if item == "/":
        if curw:
            words.append(''.join(curw))
            curw = []
    else:
        curw.append(MORSE.get(item, "?"))
if curw:
    words.append(''.join(curw))

print("step:", step)
print("letters:", letters)
print("decoded words:", words)
