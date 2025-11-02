import pandas as pd
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "task_25"
    files = sorted(data_dir.glob("dataset_part_*.csv"))
    if not files:
        raise SystemExit(f"No CSV parts found under {data_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            raise SystemExit(f"Failed to read {f}: {e}")

    total = pd.concat(dfs, ignore_index=True)
    out_path = data_dir / "total.csv"
    total.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(total):,} rows to {out_path}")


if __name__ == "__main__":
    main()
