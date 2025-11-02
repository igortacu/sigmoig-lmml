#!/usr/bin/env python3
"""
Data prep for Roman numerals.

Steps
-----
1. detect classes in data/train and data/val
2. standardize every image to 128x128, grayscale, white background
3. remove images with zero content (pure white)
4. augment rare classes in train until target size
5. keep val close to one size per class
"""

import os
import random
from pathlib import Path

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# -----------------------------------------
# CONFIG
# -----------------------------------------
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

IMG_SIZE = (128, 128)
TRAIN_TARGET_PER_CLASS = 900       # adjust if storage is low
VAL_TARGET_PER_CLASS_LOWER = 60    # min per class in val
VAL_TARGET_PER_CLASS_UPPER = 120   # max per class in val

AUG_ROT_DEG = 10
AUG_BRIGHT_MIN = 0.8
AUG_BRIGHT_MAX = 1.2
AUG_CONTRAST_MIN = 0.9
AUG_CONTRAST_MAX = 1.1

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# -----------------------------------------
# helpers
# -----------------------------------------
def list_class_dirs(root: Path):
    """Return dirs in root, sorted by name."""
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT]


def is_blank(im: Image.Image, threshold: int = 250):
    """
    Quick filter for empty images.
    Returns True if image is almost white.
    """
    # downsample for speed
    small = im.resize((32, 32))
    pixels = small.getdata()
    white_like = sum(1 for px in pixels if px >= threshold)
    # 32*32 = 1024
    return white_like > 1000


def load_grayscale(path: Path) -> Image.Image:
    im = Image.open(path).convert("L")
    return im


def standardize(im: Image.Image, size=(128, 128)) -> Image.Image:
    """
    Pad to square, resize to target, white background.
    """
    # ensure white bg
    im = ImageOps.invert(ImageOps.invert(im))  # no-op but keeps idea clear
    # fit to box with padding
    im = ImageOps.pad(im, size, color=255)  # 255 = white
    return im


def random_aug(im: Image.Image) -> Image.Image:
    """
    Light augmentation.
    Rotation, brightness, contrast, tiny blur at times.
    """
    # rotation
    angle = random.randint(-AUG_ROT_DEG, AUG_ROT_DEG)
    im = im.rotate(angle, fillcolor=255)

    # brightness
    bright_factor = random.uniform(AUG_BRIGHT_MIN, AUG_BRIGHT_MAX)
    im = ImageEnhance.Brightness(im).enhance(bright_factor)

    # contrast
    contrast_factor = random.uniform(AUG_CONTRAST_MIN, AUG_CONTRAST_MAX)
    im = ImageEnhance.Contrast(im).enhance(contrast_factor)

    # small blur sometimes
    if random.random() < 0.1:
        im = im.filter(ImageFilter.GaussianBlur(radius=0.5))

    # after aug, standardize again to avoid black corners
    im = standardize(im, IMG_SIZE)
    return im


def save_image(im: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


# -----------------------------------------
# core steps
# -----------------------------------------
def preprocess_split(split_dir: Path):
    """
    Go through every image in split.
    Standardize format.
    Drop empty images.
    """
    print(f"[preprocess] {split_dir}")
    for cls_dir in list_class_dirs(split_dir):
        imgs = list_images(cls_dir)
        for img_path in imgs:
            try:
                im = load_grayscale(img_path)
            except Exception as e:
                print(f"  [warn] cannot open {img_path}: {e}")
                img_path.unlink(missing_ok=True)
                continue

            im = standardize(im, IMG_SIZE)

            # drop blank
            if is_blank(im):
                print(f"  [drop] blank {img_path}")
                img_path.unlink(missing_ok=True)
                continue

            # overwrite with cleaned version
            save_image(im, img_path)


def balance_train(train_dir: Path):
    """
    For each class in train:
    - count images
    - if less than target, synthesize new ones via aug
    """
    print("[balance train]")
    for cls_dir in list_class_dirs(train_dir):
        imgs = list_images(cls_dir)
        n = len(imgs)
        print(f"  class {cls_dir.name}: {n} images")

        if n == 0:
            print(f"    [warn] no images in {cls_dir}")
            continue

        # load originals once
        originals = []
        for img_path in imgs:
            try:
                originals.append(load_grayscale(img_path))
            except Exception as e:
                print(f"    [warn] cannot read {img_path}: {e}")

        idx = 0
        while len(list_images(cls_dir)) < TRAIN_TARGET_PER_CLASS:
            base_im = originals[idx % len(originals)]
            aug_im = random_aug(base_im)
            out_name = f"aug_{idx:05d}.png"
            out_path = cls_dir / out_name
            save_image(aug_im, out_path)
            idx += 1

        final_count = len(list_images(cls_dir))
        print(f"    -> {final_count} images after balance")


def tidy_val(val_dir: Path):
    """
    Keep val in a narrow range per class.
    Standardize format once more.
    """
    print("[tidy val]")
    for cls_dir in list_class_dirs(val_dir):
        imgs = list_images(cls_dir)
        print(f"  val class {cls_dir.name}: {len(imgs)} images")

        # standardize again
        for img_path in imgs:
            im = load_grayscale(img_path)
            im = standardize(im, IMG_SIZE)
            save_image(im, img_path)

        # if too many, trim oldest aug (names starting with aug_)
        imgs = sorted(list_images(cls_dir), key=lambda p: p.name)
        if len(imgs) > VAL_TARGET_PER_CLASS_UPPER:
            # remove from end
            extra = len(imgs) - VAL_TARGET_PER_CLASS_UPPER
            to_remove = imgs[-extra:]
            for p in to_remove:
                print(f"    [val drop] {p}")
                p.unlink(missing_ok=True)

        # if too few, expand via aug from existing ones
        imgs = list_images(cls_dir)
        if len(imgs) < VAL_TARGET_PER_CLASS_LOWER:
            originals = [load_grayscale(p) for p in imgs]
            idx = 0
            while len(list_images(cls_dir)) < VAL_TARGET_PER_CLASS_LOWER:
                base_im = originals[idx % len(originals)]
                aug_im = random_aug(base_im)
                out_path = cls_dir / f"aug_val_{idx:04d}.png"
                save_image(aug_im, out_path)
                idx += 1

        final_n = len(list_images(cls_dir))
        print(f"    -> {final_n} images in val/{cls_dir.name}")


def print_stats(root: Path, title: str):
    print(f"[stats] {title}")
    for cls_dir in list_class_dirs(root):
        n = len(list_images(cls_dir))
        print(f"  {cls_dir.name:>5}: {n}")


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("data/train not found")
    if not VAL_DIR.exists():
        raise SystemExit("data/val not found")

    # 1) preprocess train and val
    preprocess_split(TRAIN_DIR)
    preprocess_split(VAL_DIR)

    # 2) balance train
    balance_train(TRAIN_DIR)

    # 3) tidy val
    tidy_val(VAL_DIR)

    # 4) final stats
    print_stats(TRAIN_DIR, "train")
    print_stats(VAL_DIR, "val")
    print("\nDone. Run: python train.py")


if __name__ == "__main__":
    main()
