import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.close("all")

# 1) load
data_dir = Path("task_25")
parts = sorted(data_dir.glob("dataset_part_*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

# 2) detect columns
series_cols = [c for c in df.columns
               if c.lower() in ("series", "unique_id", "id", "item_id")
               or c.lower().startswith(("series", "unique", "item"))]
if not series_cols:
    raise SystemExit(df.columns)
series_col = series_cols[0]

time_cols = [c for c in df.columns if c not in (series_col, "value", "noise_level")]
if time_cols:
    df = df.sort_values([series_col, time_cols[0]])
else:
    df = df.sort_values([series_col])

# 3) build matrix from noise_level  ← this is the thing that showed the staircase
series_ids = df[series_col].unique().tolist()
rows = []
for sid in series_ids:
    part = df[df[series_col] == sid]
    sig = part["noise_level"].to_numpy(float)
    rows.append(sig)

max_len = max(len(r) for r in rows)
M = len(rows)

mat = np.full((M, max_len), np.nan, dtype=float)
for i, r in enumerate(rows):
    mat[i, :len(r)] = r

# 4) align to row 0 with cross-corr
ref = mat[0]
ref = np.where(np.isnan(ref), np.nanmean(ref), ref)

aligned = np.full_like(mat, np.nan)

aligned[0] = ref
max_shift = 250  # increase if your series are longer

for i in range(1, M):
    row = mat[i]
    row = np.where(np.isnan(row), np.nanmean(row), row)

    best_shift = 0
    best_score = -1e9

    for sh in range(-max_shift, max_shift + 1):
        if sh < 0:
            r1 = ref[:sh]
            r2 = row[-sh:len(r1)-sh]
        elif sh > 0:
            r1 = ref[sh:]
            r2 = row[:len(r1)]
        else:
            r1 = ref
            r2 = row

        L = min(len(r1), len(r2))
        if L < 60:
            continue

        c = np.corrcoef(r1[:L], r2[:L])[0, 1]
        if c > best_score:
            best_score = c
            best_shift = sh

    # write aligned row: first fill with NaN, then drop the shifted slice
    newrow = np.full(max_len, np.nan, dtype=float)
    if best_shift < 0:
        part = row[-best_shift:]
        newrow[:len(part)] = part
    elif best_shift > 0:
        part = row[:max_len - best_shift]
        newrow[best_shift:best_shift + len(part)] = part
    else:
        newrow[:len(row)] = row

    aligned[i] = newrow

# 5) fill remaining NaNs columnwise
col_mean = np.nanmean(aligned, axis=0)
inds = np.where(np.isnan(aligned))
aligned[inds] = col_mean[inds[1]]

# 6) normalize
mn, mx = aligned.min(), aligned.max()
img = (aligned - mn) / (mx - mn + 1e-8)

# 7) pool rows and columns to thicken strokes
row_pool = 3          # 12 rows -> 4 pooled rows
col_pool = 3

H, W = img.shape
H2 = (H // row_pool) * row_pool
W2 = (W // col_pool) * col_pool

img2 = img[:H2, :W2].reshape(H2 // row_pool, row_pool, W2 // col_pool, col_pool).mean(axis=(1, 3))

# 8) adaptive threshold per column -> binary
med_per_col = np.median(img2, axis=0, keepdims=True)
binary = (img2 < med_per_col).astype(float)

plt.figure(figsize=(11, 4))
plt.imshow(1 - binary, cmap="gray", interpolation="nearest", aspect="auto")
plt.title("noise_level aligned → pooled → column-thresholded\nread left→right")
plt.axis("off")
plt.show()

# quick dump of first pooled row
bits = "".join("1" if x else "0" for x in binary[0])
print(bits[:240])
