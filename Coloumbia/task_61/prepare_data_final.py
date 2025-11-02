import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path
import shutil
import cv2

SOURCE_TRAIN = Path("data_original/train")
SOURCE_VAL = Path("data_original/val")
OUTPUT_TRAIN = Path("data/train")
OUTPUT_VAL = Path("data/val")

CLASSES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


def compute_quality_score(img_path):
    """Advanced quality scoring - only filter truly blank images."""
    try:
        img = Image.open(img_path).convert('L')
        arr = np.array(img)
        
        # Simple check: is image mostly blank?
        non_white = np.sum(arr < 250)
        content_ratio = non_white / arr.size
        
        # Only reject if almost no content (< 2%)
        is_valid = content_ratio > 0.02
        
        metrics = {
            'content': content_ratio,
            'brightness': arr.mean()
        }
        
        return is_valid, metrics
        
    except Exception as e:
        return False, {}


def enhance_image(img):
    """Apply sophisticated image enhancement."""
    # Convert to numpy for processing
    arr = np.array(img)
    
    # Denoise
    arr = cv2.fastNlMeansDenoising(arr, h=10)
    
    # Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    arr = clahe.apply(arr)
    
    # Sharpen edges
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    arr = cv2.filter2D(arr, -1, kernel)
    
    # Threshold to ensure clean whites
    arr = np.where(arr > 240, 255, arr)
    
    img = Image.fromarray(arr)
    
    # Final enhancement
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    
    return img


def augment_smart(img):
   
    aug_choice = np.random.choice(['rotate', 'affine', 'brightness', 'combo'])
    
    arr = np.array(img)
    h, w = arr.shape
    
    if aug_choice == 'rotate':
        # Small rotation
        angle = np.random.uniform(-7, 7)
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        arr = cv2.warpAffine(arr, matrix, (w, h), borderValue=255)
    
    elif aug_choice == 'affine':
        # Slight perspective/shear
        pts1 = np.float32([[0,0], [w,0], [0,h]])
        shift = np.random.randint(-3, 4, (3, 2)).astype(np.float32)
        pts2 = pts1 + shift
        matrix = cv2.getAffineTransform(pts1, pts2)
        arr = cv2.warpAffine(arr, matrix, (w, h), borderValue=255)
    
    elif aug_choice == 'brightness':
        # Adjust brightness - fix overflow issue
        shift = np.random.randint(-10, 11)
        arr = arr.astype(np.int16) + shift  # Use int16 to avoid overflow
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    elif aug_choice == 'combo':
        # Combine multiple
        angle = np.random.uniform(-5, 5)
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        arr = cv2.warpAffine(arr, matrix, (w, h), borderValue=255)
        shift = np.random.randint(-8, 9)
        arr = arr.astype(np.int16) + shift
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(arr)
    
    # Always enhance after augmentation
    img = ImageEnhance.Contrast(img).enhance(1.05)
    
    return img


def process_dataset():
    """Main processing pipeline."""
    print("="*70)
    print("ULTIMATE DATA-CENTRIC PIPELINE")
    print("="*70)
    
    # Clean output
    for path in [OUTPUT_TRAIN, OUTPUT_VAL]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    
    stats = {'train_kept': 0, 'train_removed': 0, 'val_kept': 0, 'val_removed': 0}
    
    # Process training data
    print("\n[1/4] Filtering and enhancing training data...")
    for class_name in CLASSES:
        src_dir = SOURCE_TRAIN / class_name
        dst_dir = OUTPUT_TRAIN / class_name
        dst_dir.mkdir(exist_ok=True)
        
        images = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
        print(f"  {class_name}: {len(images)} -> ", end='')
        
        kept = 0
        for img_path in images:
            is_good, metrics = compute_quality_score(img_path)
            
            if is_good:
                # Load and enhance
                img = Image.open(img_path).convert('L')
                img = enhance_image(img)
                img.save(dst_dir / img_path.name)
                kept += 1
                stats['train_kept'] += 1
            else:
                stats['train_removed'] += 1
        
        print(f"{kept} kept")
    
    # Process validation data
    print("\n[2/4] Enhancing validation data...")
    for class_name in CLASSES:
        src_dir = SOURCE_VAL / class_name
        dst_dir = OUTPUT_VAL / class_name
        dst_dir.mkdir(exist_ok=True)
        
        images = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
        
        for img_path in images:
            is_good, _ = compute_quality_score(img_path)
            
            if is_good:
                img = Image.open(img_path).convert('L')
                img = enhance_image(img)
                img.save(dst_dir / img_path.name)
                stats['val_kept'] += 1
            else:
                stats['val_removed'] += 1
    
    # Balance training set
    print("\n[3/4] Balancing training set with augmentation...")
    target = 900  # Target per class
    
    for class_name in CLASSES:
        class_dir = OUTPUT_TRAIN / class_name
        current_images = list(class_dir.glob("*"))
        current_count = len(current_images)
        
        if current_count == 0:
            print(f"  {class_name}: WARNING - No images found after filtering!")
            continue
        
        if current_count >= target:
            print(f"  {class_name}: {current_count} (no augmentation needed)")
            continue
        
        needed = target - current_count
        print(f"  {class_name}: {current_count} -> {target} (+{needed})")
        
        # Generate augmentations
        for i in range(needed):
            src_img_path = current_images[i % len(current_images)]
            src_img = Image.open(src_img_path).convert('L')
            aug_img = augment_smart(src_img)
            aug_img.save(class_dir / f"aug_{i:04d}.png")
    
    # Final stats
    print("\n[4/4] Final statistics...")
    train_total = sum(len(list((OUTPUT_TRAIN / c).glob("*"))) for c in CLASSES)
    val_total = sum(len(list((OUTPUT_VAL / c).glob("*"))) for c in CLASSES)
    total = train_total + val_total
    
    print(f"\nTrain: {train_total} images")
    print(f"Val:   {val_total} images")
    print(f"Total: {total} images")
    print(f"Removed: {stats['train_removed']} train, {stats['val_removed']} val")
    
    if total <= 10000:
        print(f"\n✓ Within limit ({10000 - total} remaining)")
        print("\n" + "="*70)
        print("✓ Ready to train! Run: python train.py")
        print("="*70)
        return True
    else:
        print(f"\n✗ Exceeds limit by {total - 10000}")
        return False


if __name__ == "__main__":
    import sys
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV required for advanced filtering")
        print("Install with: pip install opencv-python")
        sys.exit(1)
    
    process_dataset()
