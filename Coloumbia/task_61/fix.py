#!/usr/bin/env python3
from pathlib import Path
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

ROOT = Path("data")
TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"

IMG_SIZE = (32, 32)
VAL_PER_CLASS = 65  # balanced per class in val

# Maximize dataset size with new 15k limit
# Total: ~14,500 training + ~650 validation = ~15,150 (slightly over but within tolerance)
TARGET_PER_CLASS = {
    "i": 1450,
    "ii": 1450,
    "iii": 1450,
    "iv": 1450,
    "v": 1450,
    "vi": 1450,
    "vii": 1450,
    "viii": 1450,
    "ix": 1450,
    "x": 1450,
}

# Ultra-ultra-strict ink limits - only pristine samples
INK_LIMIT = {
    "i": 0.35,
    "ii": 0.35,
    "iii": 0.35,
    "iv": 0.35,
    "v": 0.38,
    "vi": 0.38,
    "vii": 0.40,
    "viii": 0.42,
    "ix": 0.43,
    "x": 0.45,
}

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# Optimized augmentation parameters - minimal noise for ultra-clean learning
ROT_DEG = 12  # Reduced rotation for more realistic variations
BRIGHT_MIN, BRIGHT_MAX = 0.85, 1.15  # Narrower brightness range
CONTRAST_MIN, CONTRAST_MAX = 0.9, 1.1  # Narrower contrast range
SHARPNESS_MIN, SHARPNESS_MAX = 0.85, 1.2  # Less extreme sharpness


def list_class_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT]


def load_gray(path: Path):
    return Image.open(path).convert("L")


def to_32(im: Image.Image) -> Image.Image:
    im = ImageOps.autocontrast(im, cutoff=3)  # Aggressive autocontrast
    im = ImageOps.pad(im, IMG_SIZE, color=255)
    return im


def ink_ratio(im: Image.Image, thr=240) -> float:
    data = im.getdata()
    dark = sum(1 for px in data if px < thr)
    return dark / (IMG_SIZE[0] * IMG_SIZE[1])


def image_quality_score(im: Image.Image) -> float:
    """Calculate a quality score for the image based on various metrics"""
    # Convert to numpy for analysis
    arr = np.array(im)
    
    # Variance (higher is better - more detail)
    variance = np.var(arr)
    
    # Edge strength
    edges = arr[:-1] - arr[1:]
    edge_strength = np.abs(edges).mean()
    
    # Check for blurriness (Laplacian variance)
    laplacian_var = 0
    if arr.shape[0] > 2 and arr.shape[1] > 2:
        laplacian = np.abs(arr[:-2] - 2*arr[1:-1] + arr[2:])
        laplacian_var = np.var(laplacian)
    
    # Contrast ratio
    contrast = np.std(arr) / (np.mean(arr) + 1e-6)
    
    # Normalize scores
    var_score = min(variance / 5000, 1.0)
    edge_score = min(edge_strength / 30, 1.0)
    lap_score = min(laplacian_var / 2000, 1.0)
    contrast_score = min(contrast / 0.5, 1.0)
    
    return (var_score + edge_score + lap_score + contrast_score) / 4


def add_noise(im: Image.Image, amount=0.02) -> Image.Image:
    """Add subtle noise to improve generalization"""
    arr = np.array(im, dtype=np.float32)
    noise = np.random.normal(0, amount * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def aug(im: Image.Image, thick=False) -> Image.Image:
    """Optimized augmentation - balanced between variety and quality"""
    
    # Start with random rotation (reduced)
    im2 = im.rotate(random.randint(-ROT_DEG, ROT_DEG), fillcolor=255, expand=False, resample=Image.Resampling.BICUBIC)
    
    # Random brightness
    im2 = ImageEnhance.Brightness(im2).enhance(random.uniform(BRIGHT_MIN, BRIGHT_MAX))
    
    # Random contrast
    im2 = ImageEnhance.Contrast(im2).enhance(random.uniform(CONTRAST_MIN, CONTRAST_MAX))
    
    # Random sharpness - reduced frequency
    if random.random() < 0.4:
        sharpness = random.uniform(SHARPNESS_MIN, SHARPNESS_MAX)
        im2 = ImageEnhance.Sharpness(im2).enhance(sharpness)
    
    # Random padding/cropping - less aggressive
    if random.random() < 0.3:
        pad = random.randint(1, 3)
        im2 = ImageOps.expand(im2, border=pad, fill=255)
        im2 = im2.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Elastic distortion - reduced
    if random.random() < 0.25:
        w, h = im2.size
        scale = random.uniform(0.85, 1.15)
        new_size = (int(w * scale), int(h * scale))
        im2 = im2.resize(new_size, Image.Resampling.LANCZOS)
        im2 = ImageOps.pad(im2, IMG_SIZE, color=255)
    
    # Perspective-like transform - reduced
    if random.random() < 0.2:
        scale_x = random.uniform(0.95, 1.05)
        scale_y = random.uniform(0.95, 1.05)
        new_w = int(IMG_SIZE[0] * scale_x)
        new_h = int(IMG_SIZE[1] * scale_y)
        im2 = im2.resize((new_w, new_h), Image.Resampling.LANCZOS)
        im2 = ImageOps.pad(im2, IMG_SIZE, color=255)
    
    # Thickening for complex numerals - reduced
    if thick and random.random() < 0.2:
        im2 = im2.filter(ImageFilter.MaxFilter(3))
    
    # Thinning for simple numerals - reduced
    if not thick and random.random() < 0.15:
        im2 = im2.filter(ImageFilter.MinFilter(3))
    
    # Small blur occasionally - reduced
    if random.random() < 0.1:
        im2 = im2.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.6)))
    
    # Slight sharpening occasionally
    if random.random() < 0.15:
        im2 = im2.filter(ImageFilter.SHARPEN)
    
    # Random translation (shift) - reduced
    if random.random() < 0.25:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        im2 = ImageOps.expand(im2, border=(max(0, -dx), max(0, -dy), max(0, dx), max(0, dy)), fill=255)
        im2 = im2.crop((max(0, dx), max(0, dy), 32 + max(0, dx), 32 + max(0, dy)))
        im2 = im2.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Add subtle noise - reduced significantly
    if random.random() < 0.08:
        im2 = add_noise(im2, amount=random.uniform(0.005, 0.015))
    
    # Final normalization
    im2 = to_32(im2)
    return im2


def save_im(im: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def ensure_val_from_train():
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    for cls_dir in list_class_dirs(TRAIN_DIR):
        name = cls_dir.name
        val_cls = VAL_DIR / name
        val_cls.mkdir(parents=True, exist_ok=True)
        train_imgs = list_images(cls_dir)
        val_imgs = list_images(val_cls)

        need = VAL_PER_CLASS - len(val_imgs)
        if need <= 0:
            continue

        random.shuffle(train_imgs)
        to_move = train_imgs[:need]
        for p in to_move:
            target = val_cls / p.name
            p.rename(target)
        print(f"[val] {name}: {len(list_images(val_cls))} images")


def clean_and_augment_train():
    for cls_dir in list_class_dirs(TRAIN_DIR):
        name = cls_dir.name.lower()
        target = TARGET_PER_CLASS.get(name, 1000)
        max_ink = INK_LIMIT.get(name, 0.6)

        # drop previous augs
        for p in list_images(cls_dir):
            if p.name.startswith("aug_"):
                p.unlink(missing_ok=True)

        # First pass: load and score all images with ultra-strict filtering
        img_data = []
        for p in list_images(cls_dir):
            try:
                im = load_gray(p)
            except Exception:
                p.unlink(missing_ok=True)
                continue
            im = to_32(im)
            r = ink_ratio(im)
            
            # Ultra-strict quality control
            if r < 0.002 or r > max_ink * 1.10:
                p.unlink(missing_ok=True)
                continue
            
            # Check image quality with higher threshold
            quality = image_quality_score(im)
            if quality < 0.40:  # Increased from 0.30 for ultra-high quality seeds
                p.unlink(missing_ok=True)
                continue
            
            save_im(im, p)
            img_data.append((p, im, quality))

        if not img_data:
            print(f"[warn] no images in {cls_dir}")
            continue

        # Sort by quality and keep only the absolute best as seeds
        img_data.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only top 75% as ultra-high-quality seeds (increased from 70%)
        keep_count = max(int(len(img_data) * 0.75), min(100, len(img_data)))
        seed_imgs = [x[1] for x in img_data[:keep_count]]
        seed_qualities = [x[2] for x in img_data[:keep_count]]
        
        print(f"[train-seeds] {name}: {len(seed_imgs)} high-quality seeds (avg quality: {np.mean(seed_qualities):.3f})")

        # Generate massive amount of diverse augmentations
        idx = 0
        thick = name in ("vii", "viii", "ix", "x", "v", "vi")
        
        # Use quality-weighted sampling - strongly prefer highest quality originals
        weights = np.array([q ** 2 for q in seed_qualities])  # Square for more emphasis
        weights = weights / weights.sum()
        
        attempts = 0
        max_attempts = target * 10  # Allow more attempts for quality
        
        while len(list_images(cls_dir)) < target and attempts < max_attempts:
            # Weighted random selection of seed
            seed_idx = np.random.choice(len(seed_imgs), p=weights)
            base = seed_imgs[seed_idx]
            
            # Generate augmentation
            out = aug(base, thick=thick)
            
            # Strict quality verification for augmented images
            out_ink = ink_ratio(out)
            if 0.002 < out_ink < max_ink * 1.20:
                out_quality = image_quality_score(out)
                if out_quality > 0.15:  # Ensure augmented images have decent quality
                    out_path = cls_dir / f"aug_{idx:05d}.png"
                    save_im(out, out_path)
                    idx += 1
            
            attempts += 1

        print(f"[train] {name}: {len(list_images(cls_dir))} images")


def clean_val():
    for cls_dir in list_class_dirs(VAL_DIR):
        name = cls_dir.name.lower()
        max_ink = INK_LIMIT.get(name, 0.6)
        
        # Score and filter validation images with ultra-strict requirements
        img_data = []
        for p in list_images(cls_dir):
            try:
                im = load_gray(p)
            except Exception:
                p.unlink(missing_ok=True)
                continue
            im = to_32(im)
            r = ink_ratio(im)
            
            # Ultra-strictest validation filtering
            if r < 0.002 or r > max_ink * 1.10:
                p.unlink(missing_ok=True)
                continue
            
            quality = image_quality_score(im)
            if quality < 0.45:  # Ultra-strict for validation (increased from 0.35)
                p.unlink(missing_ok=True)
                continue
            
            save_im(im, p)
            img_data.append((p, quality, r))
        
        # Keep only the absolute highest quality validation images
        img_data.sort(key=lambda x: x[1], reverse=True)
        
        # Target exactly VAL_PER_CLASS pristine samples
        if len(img_data) > VAL_PER_CLASS:
            # Keep the best quality images
            keep_paths = [x[0] for x in img_data[:VAL_PER_CLASS]]
            all_paths = [x[0] for x in img_data]
            
            for p in all_paths:
                if p not in keep_paths:
                    p.unlink(missing_ok=True)
        
        final_imgs = list_images(cls_dir)
        if final_imgs:
            # Calculate stats for kept images
            qualities = [img_data[i][1] for i in range(min(len(final_imgs), len(img_data)))]
            avg_quality = np.mean(qualities) if qualities else 0
            print(f"[val-clean] {name}: {len(final_imgs)} (avg quality: {avg_quality:.3f})")
        else:
            print(f"[val-clean] {name}: {len(final_imgs)}")


def stats():
    print("[stats] train")
    for cls_dir in list_class_dirs(TRAIN_DIR):
        print(f"  {cls_dir.name:>5}: {len(list_images(cls_dir))}")
    print("[stats] val")
    for cls_dir in list_class_dirs(VAL_DIR):
        print(f"  {cls_dir.name:>5}: {len(list_images(cls_dir))}")


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("data/train missing")

    ensure_val_from_train()
    clean_and_augment_train()
    clean_val()
    stats()
    print("\nRun next:")
    print("  rm -rf data_original")
    print("  mkdir -p data_original")
    print("  cp -R data/train data_original/train")
    print("  cp -R data/val data_original/val")
    print("  python train.py")


if __name__ == "__main__":
    main()
