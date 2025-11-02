#!/usr/bin/env python3
from pathlib import Path
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

TRAIN_DIR = Path("data/train")
IMG_SIZE = (32, 32)

TARGET_PER_CLASS = {
    "i": 700,
    "ii": 700,
    "iii": 700,
    "iv": 700,
    "v": 700,
    "vi": 700,
    "vii": 850,   # harder
    "viii": 850,
    "ix": 850,
    "x": 850,
}

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

ROT_DEG = 10
BRIGHT_MIN, BRIGHT_MAX = 0.85, 1.15
CONTRAST_MIN, CONTRAST_MAX = 0.9, 1.1

INK_LIMIT = {
    "i": 0.30,
    "ii": 0.32,
    "iii": 0.34,
    "iv": 0.40,
    "v": 0.45,
    "vi": 0.45,
    "vii": 0.55,
    "viii": 0.60,
    "ix": 0.60,
    "x": 0.60,
}

def list_class_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])

def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT]

def load_gray(path: Path):
    return Image.open(path).convert("L")

def to_32(im):
    im = ImageOps.autocontrast(im)
    im = ImageOps.pad(im, IMG_SIZE, color=255)
    return im

def ink_ratio(im, thr=240):
    data = im.getdata()
    dark = sum(1 for px in data if px < thr)
    return dark / (IMG_SIZE[0] * IMG_SIZE[1])

def aug(im, thick=False):
    angle = random.randint(-ROT_DEG, ROT_DEG)
    im2 = im.rotate(angle, fillcolor=255)

    bf = random.uniform(BRIGHT_MIN, BRIGHT_MAX)
    im2 = ImageEnhance.Brightness(im2).enhance(bf)

    cf = random.uniform(CONTRAST_MIN, CONTRAST_MAX)
    im2 = ImageEnhance.Contrast(im2).enhance(cf)

    if random.random() < 0.3:
        pad = random.randint(1, 2)
        im2 = ImageOps.expand(im2, border=pad, fill=255)
        im2 = im2.resize(IMG_SIZE)

    if thick:
        if random.random() < 0.5:
            im2 = im2.filter(ImageFilter.MaxFilter(3))

    im2 = to_32(im2)
    return im2

def save_im(im, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)

def process_train():
    for cls_dir in list_class_dirs(TRAIN_DIR):
        name = cls_dir.name.lower()
        target = TARGET_PER_CLASS.get(name, 700)
        max_ink = INK_LIMIT.get(name, 0.55)

        # drop old auto-aug
        for p in list_images(cls_dir):
            if p.name.startswith("aug_"):
                p.unlink(missing_ok=True)

        clean_paths = []
        for p in list_images(cls_dir):
            try:
                im = load_gray(p)
            except Exception:
                p.unlink(missing_ok=True)
                continue
            im = to_32(im)
            r = ink_ratio(im)
            if r < 0.01 or r > max_ink * 1.4:
                p.unlink(missing_ok=True)
                continue
            save_im(im, p)
            clean_paths.append(p)

        if not clean_paths:
            print(f"[warn] no images in {cls_dir}")
            continue

        src_imgs = [load_gray(p).convert("L") for p in clean_paths]
        idx = 0
        while len(list_images(cls_dir)) < target:
            base = src_imgs[idx % len(src_imgs)]
            thick = name in ("vii", "viii", "ix", "x", "v")
            out = aug(base, thick=thick)
            out_path = cls_dir / f"aug_{idx:05d}.png"
            save_im(out, out_path)
            idx += 1

        print(f"{name}: {len(list_images(cls_dir))} images")

def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("data/train missing")
    process_train()
    print("train ready")

if __name__ == "__main__":
    main()
