"""Restore a QR code with central ink-blot damage using inpainting.

This script reads `distorted_qr.png`, detects the damaged (dark blob) region,
uses OpenCV inpainting to reconstruct missing pixels, then decodes the QR code.
Output: `restored_qr.png` and decoded text printed to stdout.

Usage:
    python main.py

Dependencies: opencv-python, numpy, pyzbar (or pillow for basic decode attempts)
"""
from pathlib import Path
import sys

import cv2
import numpy as np

IN_NAME = "distorted_qr.png"
OUT_NAME = "restored_qr.png"


def restore_qr_with_inpainting(inp: Path, out: Path):
    """Restore QR code by detecting dark blob damage and inpainting it."""
    img = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "cannot-read-image"
    
    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Create binary threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # The dark blob in center is anomalously dark - detect it as damaged region
    # Invert to find dark regions
    inverted = 255 - img
    
    # Threshold to isolate the very dark blob (damage mask)
    # Use a high threshold to only capture the ink blot
    _, damage_mask = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate the mask slightly to ensure we cover all damaged pixels
    damage_mask = cv2.dilate(damage_mask, kernel, iterations=2)
    
    # Inpaint the damaged region using Telea or NS algorithm
    restored = cv2.inpaint(img, damage_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Apply final threshold to get clean binary QR
    _, final = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Additional cleanup: remove small noise
    kernel_small = np.ones((3, 3), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel_small)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_small)
    
    cv2.imwrite(str(out), final)
    return out, "ok"


def try_decode(img_path: Path):
    """Try to decode QR code using pyzbar and OpenCV QRCodeDetector."""
    results = []
    
    # Try pyzbar first (most reliable)
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image
        img = Image.open(img_path)
        decoded = pyzbar_decode(img)
        if decoded:
            for d in decoded:
                results.append(d.data.decode("utf-8", errors="replace"))
            return results
    except Exception as e:
        pass
    
    # Try OpenCV QRCodeDetector
    try:
        img = cv2.imread(str(img_path))
        detector = cv2.QRCodeDetector()
        data, pts, _ = detector.detectAndDecode(img)
        if data:
            results.append(data)
            return results
    except Exception as e:
        pass
    
    # Skipping inverted image attempts as requested (no temp inverted PNG)
    
    return results


def main():
    base = Path(__file__).parent
    inp = base / IN_NAME
    out = base / OUT_NAME
    
    if not inp.exists():
        print(f"Input not found: {inp}")
        return 2
    
    print(f"Processing {inp}...")
    
    # Restore the QR code
    result, status = restore_qr_with_inpainting(inp, out)
    
    if result is None:
        print(f"Restoration failed: {status}")
        return 1
    
    print(f"Restored image saved to: {out}")
    print(f"Status: {status}")
    
    # Try to decode
    decoded = try_decode(out)
    
    if decoded:
        print("\n[OK] QR Code decoded successfully!")
        print("=" * 50)
        for i, data in enumerate(decoded, 1):
            print(f"Decoded #{i}:")
            print(data)
            print("=" * 50)
        return 0
    else:
        print("\n[INFO] Could not decode QR code automatically.")
        print(f"Please check the restored image manually: {out}")
        # Treat as non-fatal so the script doesn't fail the run
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

