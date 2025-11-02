#!/usr/bin/env python3
"""
Prep data for roman-numeral classifier used by train.py

Goal
----
- inputs for train.py: data_original/train and data_original/val
- 10 classes: i, ii, iii, iv, v, vi, vii, viii, ix, x
- train: 900 images / class  → 9,000
- val:   80 images / class   →   800
- total: 9,800  (< 10,000 in train.py)
- format: 32x32, RGB, white background
- source: data/train and data/val (original raw folders)

Run:
    python prep_fix.py
    python train.py
"""

from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
import shutil

SRC_TRAIN = Path("data/train")
SRC_VAL = Path("data/val")

OUT_ROOT = Path("data_original")
OUT_TRAIN = OUT_ROOT / "train"
OUT_VAL = OUT_ROOT / "val"

CLASSES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

IMG_SIZE = (32, 32)
TRAIN_PER_CLASS = 900
VAL_PER_CLASS = 80

# lower ink limit: drop blanks
MIN_INK = 0.008  # 0.8% dark pixels

# upper ink per class: drop too thick samples
INK_LIMIT = {
    "i": 0.30,
    "ii": 0.32,
    "iii": 0.34,
    "iv": 0.38,
    "v": 0.42,
    "vi": 0.42,
    "vii": 0.50,
    "viii": 0.55,
    "ix": 0.55,
    "x": 0.55,
}

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# light aug params
ROT_DEG = 6
BRIGHT_MIN, BRIGHT_MAX = 0.9, 1.05
CONTRAST_MIN, CONTRAST_MAX = 0.95, 1.05


def list_imgs(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT]


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def to_32_rgb_white(im: Image.Image) -> Image.Image:
    im = ImageOps.autocontrast(im)
    im = ImageOps.pad(im, IMG_SIZE, color=255)
    im = im.convert("RGB")
    return im


def ink_ratio(im: Image.Image, thr=240) -> float:
    if im.mode != "L":
        g = im.convert("L")
    else:
        g = im
    data = g.getdata()
    dark = sum(1 for px in data if px < thr)
    return dark / (IMG_SIZE[0] * IMG_SIZE[1])


def aug(im: Image.Image, thick: bool) -> Image.Image:
    im = im.convert("L")
    angle = random.randint(-ROT_DEG, ROT_DEG)
    im2 = im.rotate(angle, fillcolor=255)

    bf = random.uniform(BRIGHT_MIN, BRIGHT_MAX)
    im2 = ImageEnhance.Brightness(im2).enhance(bf)

    cf = random.uniform(CONTRAST_MIN, CONTRAST_MAX)
    im2 = ImageEnhance.Contrast(im2).enhance(cf)

    # slight shift
    if random.random() < 0.25:
        im2 = ImageOps.expand(im2, border=1, fill=255)
        im2 = im2.resize(IMG_SIZE)

    if thick and random.random() < 0.25:
        im2 = im2.filter(ImageFilter.MaxFilter(3))

    im2 = to_32_rgb_white(im2)
    return im2


def collect_sources(cls: str):
    paths = []
    tdir = SRC_TRAIN / cls
    vdir = SRC_VAL / cls
    if tdir.exists():
        paths.extend(list_imgs(tdir))
    if vdir.exists():
        paths.extend(list_imgs(vdir))
    return paths


def filter_seeds(paths, cls: str):
    seeds = []
    max_ink = INK_LIMIT.get(cls, 0.55)
    for p in paths:
        try:
            im = load_gray(p)
        except Exception:
            continue
        im = to_32_rgb_white(im)
        r = ink_ratio(im)
        if r < MIN_INK:
            # too blank
            continue
        if r > max_ink * 1.25:
            # too bold
            continue
        seeds.append(im)
    return seeds


def write_img(im: Image.Image, out_dir: Path, idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"img_{idx:04d}.png"
    im.save(out_path)


def build_for_class(cls: str):
    src = collect_sources(cls)
    seeds = filter_seeds(src, cls)
    if not seeds:
        raise SystemExit(f"no usable images for class {cls}")

    train_dir = OUT_TRAIN / cls
    val_dir = OUT_VAL / cls
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1) validation = only clean seeds, no heavy aug
    vidx = 0
    for im in seeds:
        if vidx >= VAL_PER_CLASS:
            break
        im_std = to_32_rgb_white(im)
        r = ink_ratio(im_std)
        max_ink = INK_LIMIT.get(cls, 0.55)
        if MIN_INK <= r <= max_ink * 1.10:
            write_img(im_std, val_dir, vidx)
            vidx += 1

    # if we still need more val → light aug, no thick
    a = 0
    while vidx < VAL_PER_CLASS:
        base = seeds[a % len(seeds)]
        im2 = base.convert("L")
        angle = random.randint(-3, 3)
        im2 = im2.rotate(angle, fillcolor=255)
        im2 = to_32_rgb_white(im2)
        r = ink_ratio(im2)
        max_ink = INK_LIMIT.get(cls, 0.55)
        if MIN_INK <= r <= max_ink * 1.05:
            write_img(im2, val_dir, vidx)
            vidx += 1
        a += 1

    # 2) training = all seeds + aug to 900
    thick = cls in ("v", "vi", "vii", "viii", "ix", "x")

    tidx = 0
    for im in seeds:
        if tidx >= TRAIN_PER_CLASS:
            break
        write_img(to_32_rgb_white(im), train_dir, tidx)
        tidx += 1

    aidx = 0
    max_ink = INK_LIMIT.get(cls, 0.55)
    while tidx < TRAIN_PER_CLASS:
        base = seeds[aidx % len(seeds)]
        im2 = aug(base, thick=thick)
        r = ink_ratio(im2)
        if MIN_INK <= r <= max_ink * 1.15:
            write_img(im2, train_dir, tidx)
            tidx += 1
        aidx += 1


def stats():
    print("[stats] train")
    total_train = 0
    for cls in CLASSES:
        d = OUT_TRAIN / cls
        n = len(list_imgs(d)) if d.exists() else 0
        total_train += n
        print(f"  {cls:>4}: {n}")
    print(f"  total train: {total_train}")

    print("[stats] val")
    total_val = 0
    for cls in CLASSES:
        d = OUT_VAL / cls
        n = len(list_imgs(d)) if d.exists() else 0
        total_val += n
        print(f"  {cls:>4}: {n}")
    print(f"  total val: {total_val}")
    print(f"  grand total: {total_train + total_val}")


def main():
    if not SRC_TRAIN.exists() and not SRC_VAL.exists():
        raise SystemExit("data/train or data/val missing")

    # wipe old output
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_VAL.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        print(f"[build] {cls}")
        build_for_class(cls)

    stats()
    print("\nNext:")
    print("  python train.py")


if __name__ == "__main__":
    main()
