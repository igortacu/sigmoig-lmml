#!/usr/bin/env python3
"""
Legend of Restoration – brute version

Input:  shuffled_puzzle.png
Output: several PNGs:
  - step_crop.png
  - step_unskew.png
  - restored_t16_py_pc.png   → tile=16, python RNG, per-channel noise
  - restored_t16_py_px.png   → tile=16, python RNG, per-pixel noise
  - restored_t16_np_pc.png   → tile=16, numpy RNG, per-channel noise
  - ...
Also tries tile 8 and 32.
Pick the cleanest one.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import random

<<<<<<< HEAD

# ------------------- helpers -------------------
=======
pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
>>>>>>> origin/main

def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def auto_crop_bright(img: Image.Image, thr: int = 80) -> Image.Image:
    arr = np.array(img)
    gray = arr.mean(axis=2)
    ys, xs = np.where(gray > thr)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def trim_to_mul16(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    w16 = (w // 16) * 16
    h16 = (h // 16) * 16
    arr = arr[:h16, :w16, :]
    return Image.fromarray(arr, mode="RGB")


def step_unskew(img: Image.Image) -> Image.Image:
    w, h = img.size
    # inverse of (1, 0.2, 0, 0, 1, 0)
    affine_inv = (1, -0.2, 0,
                  0,  1.0, 0)
    img2 = img.transform((w, h), Image.AFFINE, affine_inv, resample=Image.BICUBIC)
    return img2


def step_swap_rb(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    arr = arr[..., [2, 1, 0]]  # swap R and B
    return Image.fromarray(arr, mode="RGB")


def step_flip_v(img: Image.Image) -> Image.Image:
    return ImageOps.flip(img)


def step_flip_h(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def step_invert(img: Image.Image) -> Image.Image:
    return ImageOps.invert(img)


def step_remove_noise(img: Image.Image, per_channel: bool) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    h, w, c = arr.shape
    rng = np.random.default_rng(42)
    if per_channel:
        noise = rng.integers(-50, 51, size=(h, w, c), dtype=np.int16)
    else:
        noise_px = rng.integers(-50, 51, size=(h, w), dtype=np.int16)
        noise = noise_px[:, :, None]
    out = arr - noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def unshuffle_numpy(img: Image.Image, tile: int = 16, seed: int = 42) -> Image.Image:
    arr = np.array(img)
    h, w, c = arr.shape
    tx = w // tile
    ty = h // tile
    n = tx * ty

    ids = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)  # this is the *forward* order

    invp = np.empty_like(ids)
    invp[ids] = np.arange(n)

    out = np.zeros_like(arr)
    for scr_idx in range(n):
        orig_idx = invp[scr_idx]
        sx = (scr_idx % tx) * tile
        sy = (scr_idx // tx) * tile
        dx = (orig_idx % tx) * tile
        dy = (orig_idx // tx) * tile
        out[dy:dy+tile, dx:dx+tile, :] = arr[sy:sy+tile, sx:sx+tile, :]
    return Image.fromarray(out, mode="RGB")


def unshuffle_python(img: Image.Image, tile: int = 16, seed: int = 42) -> Image.Image:
    arr = np.array(img)
    h, w, c = arr.shape
    tx = w // tile
    ty = h // tile
    n = tx * ty

    ids = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(ids)  # this is the *forward* order

    invp = [0] * n
    for new_pos, old_pos in enumerate(ids):
        invp[new_pos] = old_pos

    out = np.zeros_like(arr)
    for scr_idx in range(n):
        orig_idx = invp[scr_idx]
        sx = (scr_idx % tx) * tile
        sy = (scr_idx // tx) * tile
        dx = (orig_idx % tx) * tile
        dy = (orig_idx // tx) * tile
        out[dy:dy+tile, dx:dx+tile, :] = arr[sy:sy+tile, sx:sx+tile, :]
    return Image.fromarray(out, mode="RGB")


# ------------------- main -------------------

def main():
    if len(sys.argv) < 2:
        print("usage: python restore_brutal.py input.png")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = in_path.parent

    img0 = load_rgb(str(in_path))

    # 1. remove dark frame
    img1 = auto_crop_bright(img0, thr=80)
    img1.save(out_dir / "step_crop.png")

    # 2. unskew right away (before trimming)
    img2 = step_unskew(img1)
    img2.save(out_dir / "step_unskew.png")

    # 3. trim to 16x16 grid
    img3 = trim_to_mul16(img2)
    img3.save(out_dir / "step_trim16.png")

    # 4. fixed-order core steps (the ones from the story)
    img4 = step_swap_rb(img3)
    img5 = step_flip_v(img4)
    img6 = step_flip_h(img5)
    img7 = step_invert(img6)

    # now branch: noise variant × RNG variant × tile size
    noise_modes = [
        ("pc", True),
        ("px", False),
    ]
    rng_modes = [
        ("py", unshuffle_python),
        ("np", unshuffle_numpy),
    ]
    tile_sizes = [16, 8, 32]

    for noise_name, per_ch in noise_modes:
        den = step_remove_noise(img7, per_channel=per_ch)
        # save intermediate in case you want to check
        den.save(out_dir / f"step_denoise_{noise_name}.png")

        for rng_name, unshuf_fn in rng_modes:
            for t in tile_sizes:
                try:
                    res = unshuf_fn(den, tile=t, seed=42)
                    out_name = f"restored_t{t}_{rng_name}_{noise_name}.png"
                    res.save(out_dir / out_name)
                    print("wrote", out_name)
                except Exception as e:
                    print("skip", rng_name, noise_name, t, "->", e)


if __name__ == "__main__":
    main()
