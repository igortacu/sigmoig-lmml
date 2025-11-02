import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
    return id_col, time_col, num_cols


def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "task_25"
    files = sorted(data_dir.glob("dataset_part_*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    # Load and concat
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    id_col, time_col, num_cols = detect_columns(df)
    # Parse timestamp
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Filter: noise_level < 0.3
    if "noise_level" in df.columns:
        df = df[df["noise_level"] < 0.3].copy()
        print(f"[INFO] Filtered to {len(df):,} rows with noise_level < 0.3")

    if not num_cols:
        raise SystemExit("No numeric columns to plot.")

    # Sort by timestamp then value
    sort_cols = [time_col]
    if "value" in df.columns:
        sort_cols.append("value")
    df = df.sort_values(sort_cols)

    # Create a subplot per numeric column
    n = len(num_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Scatter all points; alpha reduces overplotting
    for ax, col in zip(axes, num_cols):
        ax.scatter(df[time_col].values, df[col].values, s=3, alpha=0.3, edgecolors='none')
        ax.set_ylabel(col)
        ax.grid(True, linestyle=':', alpha=0.4)

    axes[-1].set_xlabel(str(time_col))
    fig.suptitle("Time series scatter by timestamp (all points)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save a copy and also show
    out_png = data_dir / "scatter_all.png"
    fig.savefig(out_png, dpi=150)
    print(f"[OK] Saved scatter to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
