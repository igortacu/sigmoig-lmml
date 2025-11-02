import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load and process data exactly like original decode.py but more carefully

df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Loaded {len(x)} data points")

# Normalize y
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

# Helper function for smoothing
def smooth(a, k):
    pad = k // 2
    a = np.pad(a, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="valid")

# Try different bin sizes to find one that works
best_result = None

for N_BINS in [500, 600, 700, 800, 900, 1000, 1100]:
    print(f"\nTrying N_BINS={N_BINS}")
    
    # Bin the data
    edges = np.linspace(x.min(), x.max(), N_BINS + 1)
    bix = np.digitize(x, edges) - 1
    
    bx, by = [], []
    for b in range(N_BINS):
        m = bix == b
        if m.any():
            bx.append(x[m].mean())
            by.append(y_norm[m].mean())
    
    bx = np.array(bx)
    by = np.array(by)
    
    # Smooth
    sig = smooth(by, k=9)
    
    # Find local maxima
    loc_x = []
    loc_y = []
    for i in range(1, len(sig) - 1):
        if sig[i] >= sig[i-1] and sig[i] >= sig[i+1]:
            loc_x.append(bx[i])
            loc_y.append(sig[i])
    
    loc_x = np.array(loc_x)
    loc_y = np.array(loc_y)
    
    print(f"  Found {len(loc_x)} local maxima")
    
    if len(loc_x) < 10 or len(loc_x) > 100:
        print(f"  -> Too many/few peaks, skipping")
        continue
    
    # Estimate step from maxima
    dx = np.diff(loc_x)
    dx_small = dx[dx < np.percentile(dx, 90)].reshape(-1, 1)
    
    if len(dx_small) < 2:
        print(f"  -> Not enough gaps to estimate step, skipping")
        continue
    
    kdx = KMeans(n_clusters=1, n_init=5, random_state=0).fit(dx_small)
    step = float(kdx.cluster_centers_[0, 0])
    
    print(f"  Estimated step: {step:.6f}")
    
    # Build regular grid
    x0 = loc_x.min()
    x1 = loc_x.max()
    grid = np.arange(x0, x1 + step, step)
    
    # Sample signal on grid
    vals = []
    half = step * 0.45
    for gx in grid:
        m = (bx >= gx - half) & (bx <= gx + half)
        if m.any():
            vals.append(sig[m].max())
        else:
            vals.append(0.0)
    vals = np.array(vals)
    
    print(f"  Grid has {len(vals)} cells")
    
    # Cluster heights into gap/dot/dash
    vh = vals.reshape(-1, 1)
    nz = vals > 0
    fit_vals = vh[nz]
    
    if len(fit_vals) < 3:
        print(f"  -> Too few non-zero values, skipping")
        continue
    
    km = KMeans(n_clusters=3, n_init=12, random_state=0).fit(fit_vals)
    cent = km.cluster_centers_.ravel()
    order = np.argsort(cent)
    remap = {old: new for new, old in enumerate(order)}
    
    labels = np.full(len(vals), 0, dtype=int)
    labels[nz] = np.array([remap[l] for l in km.predict(fit_vals)])
    
    print(f"  Cluster centers: {np.sort(cent)}")
    print(f"  Labels distribution: {np.bincount(labels)}")
    
    # Decode
    symbols = []
    for lab in labels:
        if lab == 0:
            symbols.append(" ")
        elif lab == 1:
            symbols.append(".")
        else:
            symbols.append("-")
    
    symbol_str = ''.join(symbols)
    print(f"  Morse symbols: {symbol_str}")
    
    # Extract letters using gap logic
    letters = []
    cur = []
    gap_count = 0
    
    for s in symbols:
        if s in ".-":
            if gap_count >= 7:
                if cur:
                    letters.append(''.join(cur))
                    cur = []
                letters.append("/")
            elif gap_count >= 3:
                if cur:
                    letters.append(''.join(cur))
                    cur = []
            gap_count = 0
            cur.append(s)
        else:
            gap_count += 1
    
    if cur:
        letters.append(''.join(cur))
    
    print(f"  Morse letters: {letters}")
    
    # Decode to text
    MORSE = {
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
        "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
        "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
        "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
        "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
        ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    }
    
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
    
    result = ' '.join(words)
    print(f"  DECODED: {result}")
    
    # Check if result looks reasonable (has no ?'s and is a real word)
    num_unknowns = sum(1 for w in words for c in w if c == "?")
    
    if num_unknowns == 0 and len(words) > 0:
        best_result = (N_BINS, result)
        print(f"  ✓ GOOD CANDIDATE!")

if best_result:
    print(f"\n\nBEST RESULT (N_BINS={best_result[0]}): {best_result[1]}")
else:
    print("\n\nNo perfect decode found, trying best partial match...")
