import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("task_25")
files = sorted(data_dir.glob("dataset_part_*.csv"))
if not files:
    raise SystemExit("no csv files")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# detect id column
id_col = None
for name in ("unique_id", "series_id", "id", "item_id", "segment"):
    if name in df.columns:
        id_col = name
        break
if id_col is None:
    df["__id"] = 0
    id_col = "__id"

# detect time column
time_col = None
for name in ("ds", "date", "timestamp", "time"):
    if name in df.columns:
        time_col = name
        break
if time_col is None:
    time_col = df.columns[0]

# numeric columns only
num_cols = [c for c in df.columns if c not in (id_col, time_col)]
print("numeric columns:", num_cols)

def build_matrix(col):
    rows = []
    for sid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        rows.append(g[col].to_numpy(dtype=float))
    max_len = max(len(r) for r in rows)
    mat = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        mat[i, :len(r)] = r
    return mat

for col in num_cols:
    mat = build_matrix(col)
    mask = ~np.isnan(mat)
    vals = mat[mask]

    # 1) raw
    mn, mx = vals.min(), vals.max()
    norm = (mat - mn) / (mx - mn)
    norm[~mask] = 1.0

    plt.figure(figsize=(7,4))
    plt.imshow(norm, cmap="gray", aspect="auto", interpolation="nearest")
    plt.title(f"{col} – raw")
    plt.tight_layout()

    # 2) global binary
    thr = np.median(vals)
    b_global = (mat >= thr).astype(float)
    b_global[~mask] = 1.0

    plt.figure(figsize=(7,4))
    plt.imshow(b_global, cmap="gray", aspect="auto", interpolation="nearest")
    plt.title(f"{col} – binary global")
    plt.tight_layout()

    # 3) row binary
    b_row = np.ones_like(mat)
    for i in range(mat.shape[0]):
        row = mat[i]
        m = ~np.isnan(row)
        if m.sum() == 0:
            continue
        rthr = np.median(row[m])
        b_row[i, m] = (row[m] >= rthr).astype(float)
    b_row[~mask] = 1.0

    plt.figure(figsize=(7,4))
    plt.imshow(b_row, cmap="gray", aspect="auto", interpolation="nearest")
    plt.title(f"{col} – binary per row (look here)")
    plt.tight_layout()

    # 4) transposed
    plt.figure(figsize=(5,7))
    plt.imshow(b_row.T, cmap="gray", aspect="auto", interpolation="nearest")
    plt.title(f"{col} – binary per row – transposed")
    plt.tight_layout()

plt.show()
