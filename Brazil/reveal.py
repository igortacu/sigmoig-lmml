import pandas as pd
import numpy as np
from pathlib import Path
import argparse
def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("dataset_part_*.csv"))
    if not files:
        raise SystemExit(f"No csv files found in {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def detect_columns(df: pd.DataFrame):
    # id column
    id_col = None
    for name in ("unique_id", "series_id", "id", "item_id", "segment"):
        if name in df.columns:
            id_col = name
            break
    if id_col is None:
        df["__id"] = 0
        id_col = "__id"

    # time column
    time_col = None
    for name in ("ds", "date", "timestamp", "time"):
        if name in df.columns:
            time_col = name
            break
    if time_col is None:
        time_col = df.columns[0]

    # numeric columns only
    num_cols = [c for c in df.columns if c not in (id_col, time_col)]
    # keep only numeric dtypes
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
    return id_col, time_col, num_cols


def build_matrix(df: pd.DataFrame, id_col: str, time_col: str, value_col: str) -> np.ndarray:
    rows = []
    for sid, g in df.groupby(id_col):
        g = g.sort_values(time_col)
        rows.append(g[value_col].to_numpy(dtype=float))
    max_len = max(len(r) for r in rows)
    mat = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return mat


def downsample_binary(mat: np.ndarray, gh: int = 1, gw: int = 1) -> np.ndarray:
    """Downsample a binary (0/1) matrix by blocks using mean then threshold at 0.5."""
    h, w = mat.shape
    if gh > 1:
        h2 = h // gh
        mat = mat[: h2 * gh, :]
        mat = mat.reshape(h2, gh, w).mean(axis=1)
    if gw > 1:
        h, w = mat.shape
        w2 = w // gw
        mat = mat[:, : w2 * gw]
        mat = mat.reshape(h, w2, gw).mean(axis=2)
    return (mat >= 0.5).astype(float)


def ascii_art_from_binary(mat: np.ndarray) -> str:
    # Inverted mapping for readability: 1 -> block (█), 0 -> space
    lines = []
    for row in mat:
        line = []
        for v in row:
            line.append("█" if v >= 0.5 else " ")
        lines.append("".join(line))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Reveal hidden word via ASCII art from time series")
    parser.add_argument("--col", type=str, default="value", help="Numeric column to visualize")
    parser.add_argument("--mode", type=str, choices=["row", "global"], default="row", help="Thresholding mode: per-row median or global median")
    parser.add_argument("--gh", type=int, default=2, help="Row downsample factor (height)")
    parser.add_argument("--gw", type=int, default=4, help="Column downsample factor (width)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_dir = base / "task_25"
    df = load_dataset(data_dir)
    id_col, time_col, num_cols = detect_columns(df)

    if args.col not in num_cols:
        raise SystemExit(f"Column '{args.col}' not found in numeric columns {num_cols}")

    col = args.col
    mat = build_matrix(df, id_col, time_col, col)
    mask = ~np.isnan(mat)

    # Choose threshold mode
    if args.mode == "row":
        b = np.ones_like(mat)
        for i in range(mat.shape[0]):
            row = mat[i]
            m = ~np.isnan(row)
            if m.sum() == 0:
                continue
            rthr = np.median(row[m])
            b[i, m] = (row[m] >= rthr).astype(float)
        b[~mask] = 1.0
    else:
        vals = mat[mask]
        gthr = np.median(vals)
        b = (mat >= gthr).astype(float)
        b[~mask] = 1.0

    # Transpose to time x series and downsample for readability
    img = b.T
    img_ds = downsample_binary(img, gh=max(1, args.gh), gw=max(1, args.gw))

    sep = "=" * 80
    print(sep)
    print(f"Column: {col} — mode={args.mode} (transposed), downsample gh={args.gh}, gw={args.gw}")
    print(sep)
    print(ascii_art_from_binary(img_ds))
    print(sep)


if __name__ == "__main__":
    main()
