import argparse
import random
import shutil
import sys
from pathlib import Path


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_class_folders(root: Path) -> dict[str, list[Path]]:
    """
    Discover class folders containing images under the given root.
    Accepts either of these layouts:
      - root/cola/*.jpg, root/fanta/*.jpg, root/sprite/*.jpg
      - root/source/cola/*.jpg, ...
    Returns: {class_name: [image_paths,...]}
    """
    candidates = []
    # direct children
    candidates.append(root)
    # common subdir name someone might use
    for name in ("source", "raw", "images", "all", "dataset"):
        p = root / name
        if p.is_dir():
            candidates.append(p)

    classes: dict[str, list[Path]] = {}
    for base in candidates:
        if not base.is_dir():
            continue
        subdirs = [d for d in base.iterdir() if d.is_dir()]
        # Heuristic: class folder must contain at least one image
        cls_map: dict[str, list[Path]] = {}
        for d in subdirs:
            imgs = [p for p in d.rglob("*") if p.is_file() and is_image(p)]
            if imgs:
                cls_map[d.name] = imgs
        # Need at least 2 classes to be a valid classification dataset
        if len(cls_map) >= 2:
            classes = cls_map
            break
    return classes


def create_split(classes: dict[str, list[Path]], out_root: Path, seed: int, tvt: tuple[float, float, float]):
    """Create train/val/test split by copying files into out_root/train|val|test/<class>.
    tvt: ratios that sum to 1.0
    """
    train_r, val_r, test_r = tvt
    rng = random.Random(seed)
    for split in ("train", "val", "test"):
        for cls in classes.keys():
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    for cls, items in classes.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        n_test = n - n_train - n_val
        splits = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }
        for split, paths in splits.items():
            dst_dir = out_root / split / cls
            for src in paths:
                dst = dst_dir / src.name
                if dst.exists():
                    # Avoid name collision by prefixing an index
                    base = src.stem
                    ext = src.suffix
                    i = 1
                    while (dst_dir / f"{base}_{i}{ext}").exists():
                        i += 1
                    dst = dst_dir / f"{base}_{i}{ext}"
                shutil.copy2(src, dst)


def _resolve_device(requested: str | None) -> str:
    """Return a safe device string for Ultralytics based on availability.
    - If CUDA is unavailable, always return 'cpu'.
    - Accepts 'cpu', '0', '0,1', etc. Falls back to 'cpu' if invalid for current env.
    """
    try:
        import torch  # local import to avoid hard dependency at module import time
    except Exception:
        # If torch import fails, ultralytics will raise later; default to cpu here
        return "cpu"

    if requested is None or str(requested).strip().lower() in {"", "auto"}:
        return "0" if torch.cuda.is_available() else "cpu"

    req = str(requested).strip()
    if req.lower() == "cpu":
        return "cpu"
    # GPU indices string like '0' or '0,1'
    parts = [p.strip() for p in req.split(",")]
    if all(p.isdigit() for p in parts):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return req
        else:
            print("[WARN] CUDA not available; falling back to CPU.")
            return "cpu"
    # Anything else: let Ultralytics try it, but if CUDA isn't available, force CPU
    return req if torch.cuda.is_available() else "cpu"


def train_yolov8_cls(dataset_root: Path, epochs: int, batch: int, imgsz: int, model_name: str, device: str | None):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERROR] Failed to import ultralytics. Did you install requirements?", file=sys.stderr)
        raise

    data_dir = dataset_root
    # Ultralytics classification expects: data_dir/{train,val,test}/<class>/*
    if not (data_dir / "train").is_dir():
        # Try to auto-locate a nested split root (handles cases like data/data/coke_fanta_sprite)
        try:
            nested_trains = [p for p in data_dir.rglob("train") if p.is_dir()]
        except Exception:
            nested_trains = []
        for tdir in nested_trains:
            cand_root = tdir.parent
            if (cand_root / "val").is_dir():
                data_dir = cand_root
                print(f"[INFO] Located existing train/val under {data_dir}; using it.")
                break

    if not (data_dir / "train").is_dir():
        # Attempt to discover classes and create a split
        print(f"[INFO] train/ not found under {data_dir}. Attempting to build a split from class folders...")
        classes = discover_class_folders(dataset_root)
        if not classes:
            raise SystemExit(
                f"No valid class folders with images were found under {dataset_root}.\n"
                "Expected structure like: data/cola/*.jpg, data/fanta/*.jpg, data/sprite/*.jpg\n"
                "Alternatively, place your images under one of: source/, raw/, images/, all/, dataset/"
            )
        split_root = dataset_root / "_split"
        if split_root.exists():
            shutil.rmtree(split_root)
        create_split(classes, split_root, seed=42, tvt=(0.8, 0.1, 0.1))
        data_dir = split_root
        print(f"[INFO] Created split at {data_dir}")

    resolved_device = _resolve_device(device)
    print(f"[INFO] Starting training with data at: {data_dir} (device={resolved_device})")
    model = YOLO(model_name)  # e.g., 'yolov8n-cls.pt'
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        lr0=0.01,
        lrf=0.01,
        patience=20,
        dropout=0.05,
        verbose=True,
        project=str(dataset_root / "runs"),
        name="cls",
        exist_ok=True,
    )

    # Find best weights and copy to Romania/model.pt
    weights = Path(results.save_dir) / "weights" / "best.pt"
    if not weights.exists():
        # Fallback to latest common path
        weights = (dataset_root / "runs" / "cls" / "weights" / "best.pt")
    out_model = Path(__file__).resolve().parent / "model.pt"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, out_model)
    print(f"[INFO] Saved best model to {out_model}")

    # Optional: evaluate on test split if present
    test_dir = Path(data_dir) / "test"
    if test_dir.is_dir():
        metrics = model.val(data=str(data_dir), split="test", imgsz=imgsz, device=resolved_device)
        print("[INFO] Test metrics:", metrics)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 classification on Cola/Fanta/Sprite")
    # Default to the current dataset location which already contains train/val/test
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "data" / "coke_fanta_sprite"),
        help="Dataset root directory (should contain train/val/test subfolders)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Pretrained YOLOv8 classification model")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id like '0' or 'cpu'")
    args = parser.parse_args()

    dataset_root = Path(args.data_dir)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_root}")

    train_yolov8_cls(dataset_root, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()
