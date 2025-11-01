"""
Clean only speckle/noise on a scanned document while preserving the original page
(grayscale/tones/text). This avoids binarization or background flattening.

Pipeline:
1) Build a robust text mask so we never alter text pixels.
2) Detect tiny dark/bright speckles with black-hat/top-hat.
3) Filter to keep only very small connected components (true speckles).
4) Inpaint those speckles.
5) Optionally, lightly denoise ONLY the background (not the text).

Usage:
  python clean_noise_only.py --input image1.png --output cleaned.png

Optional tuning:
  --bh-thresh 12        # black-hat threshold for dark speckles
  --th-thresh 12        # top-hat threshold for bright speckles
  --max-area 40         # max speckle area (in pixels)
  --max-dim 9           # max speckle width/height
  --no-bg-denoise       # disable background-only light denoising
  --save-debug          # save intermediate images for inspection

Requires:
  pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# Repository-relative default input directories (visible in help)
BASE = Path(__file__).parent
DEFAULT_NOISY_DIR = str(BASE / 'noisy')
DEFAULT_ORIGINAL_DIR = str(BASE / 'original')


def make_text_mask(gray: np.ndarray) -> np.ndarray:
    """
    Create a conservative mask of text/graphics (foreground).
    We dilate slightly to protect edges from any smoothing.
    """
    # Adaptive threshold -> text as white (255)
    bin_inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 15
    )
    # Remove isolated dots from the mask itself so they don't get treated as text
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # Slight dilation to protect text edges
    text_mask = cv2.dilate(bin_inv, np.ones((3, 3), np.uint8), iterations=1)
    return text_mask  # 255 = text, 0 = background


def detect_speckles(gray: np.ndarray, bh_thresh: int, th_thresh: int) -> np.ndarray:
    """
    Detect tiny dark and bright speckles using black-hat and top-hat.
    Returns a binary mask where speckles are 255.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # dark specks positive
    tophat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   kernel)  # bright specks positive

    _, bh_bin = cv2.threshold(blackhat, bh_thresh, 255, cv2.THRESH_BINARY)
    _, th_bin = cv2.threshold(tophat,   th_thresh, 255, cv2.THRESH_BINARY)

    speckle_mask = cv2.bitwise_or(bh_bin, th_bin)
    # Clean tiny single pixels and thin lines
    speckle_mask = cv2.morphologyEx(speckle_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return speckle_mask


def filter_small_components(mask: np.ndarray, max_area: int, max_dim: int) -> np.ndarray:
    """
    Keep only small, blob-like components typical of speckles.
    """
    if mask.max() == 0:
        return mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w    = int(stats[i, cv2.CC_STAT_WIDTH])
        h    = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area <= max_area and w <= max_dim and h <= max_dim:
            out[labels == i] = 255
    return out


def inpaint_speckles(img: np.ndarray, speckle_mask: np.ndarray) -> np.ndarray:
    """
    Inpaint speckles using Telea method. Works for 1 or 3 channels.
    """
    # Inpaint requires 8-bit single mask with 255 as inpaint region
    mask = (speckle_mask > 0).astype(np.uint8) * 255
    # cv2.inpaint supports 1-channel or 3-channel 8-bit images
    inpainted = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    return inpainted


def background_only_denoise(img: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """
    Light denoising only on background pixels, leaving text untouched.
    """
    bg_mask = (text_mask == 0).astype(np.uint8)  # 1 where background
    if img.ndim == 2:
        den = cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)
        out = img.copy()
        out[bg_mask.astype(bool)] = den[bg_mask.astype(bool)]
        return out
    else:
        den = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)
        out = img.copy()
        # Apply mask per channel
        for c in range(3):
            ch = out[..., c]
            ch_den = den[..., c]
            ch[bg_mask.astype(bool)] = ch_den[bg_mask.astype(bool)]
            out[..., c] = ch
        return out


def save_debug(outdir, **images):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, im in images.items():
        p = os.path.join(outdir, f"{name}.png")
        cv2.imwrite(p, im)
        paths.append(p)
    return paths


def _process_single_file(in_path: Path, out_path: Path, args):
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: could not read {in_path}, skipping")
        return

    # Work in color; keep a grayscale for masks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Text mask (protect text from any changes)
    text_mask = make_text_mask(gray)

    # 2) Detect speckles
    speckles = detect_speckles(gray, bh_thresh=args.bh_thresh, th_thresh=args.th_thresh)

    # 3) Remove any speckle that overlaps text (we don't want to remove punctuation/dots)
    # Also, filter by small size/shape
    protected = cv2.bitwise_and(speckles, text_mask)
    speckles_no_text = cv2.bitwise_and(speckles, cv2.bitwise_not(text_mask))
    speckles_small = filter_small_components(speckles_no_text, max_area=args.max_area, max_dim=args.max_dim)

    # 4) Inpaint just those small background speckles
    inpainted = inpaint_speckles(img, speckles_small)

    # 5) Optional: light NLMeans denoise only on background
    if args.no_bg_denoise:
        cleaned = inpainted
    else:
        cleaned = background_only_denoise(inpainted, text_mask)

    # Save final
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cleaned)
    print(f"Saved cleaned image to: {out_path}")

    if args.save_debug:
        dbg = {
            "00_input": img,
            "01_gray": gray,
            "02_text_mask": text_mask,
            "03_speckles_raw": speckles,
            "04_speckles_protected_overlap": protected,
            "05_speckles_small_bg_only": speckles_small,
            "06_inpainted": inpainted,
            "07_final_cleaned": cleaned,
        }
        dbg_dir = Path(args.debug_dir) / in_path.stem
        paths = save_debug(str(dbg_dir), **dbg)
        print("Debug images:")
        for p in paths:
            print(" -", p)


def main():
    ap = argparse.ArgumentParser(description="Remove only speckle/noise from a scanned page without altering text/background tones.")
    ap.add_argument("--input", "-i", required=False, help="Path to input image (file)")
    ap.add_argument("--indir", "-d", required=False, help=f"Directory containing input images (optional). Example: {DEFAULT_NOISY_DIR}")
    ap.add_argument("--outdir", default="cleaned_out", help="Directory to save outputs when processing a directory")
    ap.add_argument("--output", "-o", default="cleaned.png", help="Path to save cleaned image (when processing single file)")
    ap.add_argument("--bh-thresh", type=int, default=12, help="Black-hat threshold for dark speckles")
    ap.add_argument("--th-thresh", type=int, default=12, help="Top-hat threshold for bright speckles")
    ap.add_argument("--max-area", type=int, default=40, help="Max connected component area to treat as speckle")
    ap.add_argument("--max-dim", type=int, default=9, help="Max width/height of a speckle component")
    ap.add_argument("--no-bg-denoise", action="store_true", help="Disable background-only light denoising")
    ap.add_argument("--save-debug", action="store_true", help="Save intermediate images for inspection")
    ap.add_argument("--debug-dir", default="debug_noise_clean", help="Directory for debug images")
    ap.add_argument("--glob", default="*.png", help="Glob pattern to match images in directory mode")
    args = ap.parse_args()

    # If directory mode, iterate files
    if args.indir:
        indir = Path(args.indir)
        if not indir.exists():
            raise FileNotFoundError(f"Input directory not found: {indir}")
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in indir.glob(args.glob) if p.is_file()])
        if not files:
            print(f"No files found in {indir} matching {args.glob}")
            return
        for f in files:
            out_path = outdir / f"{f.stem}_cleaned{f.suffix}"
            _process_single_file(f, out_path, args)
        return

    # Single-file mode (input required)
    if not args.input:
        raise SystemExit("No input specified. Use --input <file> or --indir <directory>.")

    in_path = Path(args.input)
    out_path = Path(args.output)
    _process_single_file(in_path, out_path, args)


if __name__ == "__main__":
    main()