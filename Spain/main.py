<<<<<<< HEAD
import cv2
import numpy as np
from PIL import Image

# === LEGEND OF RESTORATION ===
# Step-by-step reversal of each “Age” in the Chronicle of the Shattered Realm

# Load the cursed image
img = cv2.imread("initial_to_be_given.jpg")

# 1️⃣ Undo “Pact of Crimson and Azure” — Swap red and blue
img = img[:, :, [2, 1, 0]]
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shattered Realm — robust reassembler (seed = 42)

Reverse order:
  A) inverse shear (1, -0.2, 0, 0, 1, 0)
  B) swap R/B
  C) vertical flip
  D) horizontal flip
  E) invert (255 - v)
  F) remove noise in [-50,+50] with seed=42 (subtract the same noise)
  G) unshatter with seed=42 (search tile sizes, perm directions, D8 tile-orient, grid sym/rolls)

Usage:
  pip install pillow numpy
  python main.py
"""

import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# ---------------- Config ----------------
CANDIDATE_NAMES = ["initial_to_be_given.png","initial_to_be_given.jpg","initial_to_be_given.jpeg"]
OUTPUT_BASENAME = "restored"

# Search space
TILE_SIZES_PX: List[int] = [4, 8, 16, 32]   # pixel tile sizes to try
TRY_GRID_RC = (16, 16)                      # 16x16 grid interpretation
TRY_PERM_DIRECTIONS = ["pull", "push"]      # new[i]=old[perm[i]]  or  new[perm[i]]=old[i]
TRY_TILE_ORIENT = ["none", "after_unshuffle", "before_unshuffle"]  # per-tile D8 inverse
SEED = 42
STORM_MAGNITUDE = 50
SHEAR_X = 0.2

# Grid-search knobs (auto disabled on huge grids)
ENABLE_SHIFT_SEARCH = True
ENABLE_GRID_D8_SEARCH = True
MAX_GRID_FOR_SEARCH = 128*128   # if rows*cols exceeds this, skip grid D8/shift search

SAVE_VARIANTS = True     # save candidate winners per hypothesis
SAVE_STEPS = True        # save A..F intermediates for the final winner
HELPERS_DIR = "helpers_out"
# ----------------------------------------
>>>>>>> c2aaf727a7e3c41aa7df2491c612be6f4d253009

# 2️⃣ Undo “Mirror of Night” — Invert all colors
img = cv2.bitwise_not(img)

<<<<<<< HEAD
# 3️⃣ Undo “Eastward Turn” — Horizontal flip
img = cv2.flip(img, 1)
=======
def _here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def resolve_input_path() -> Path:
    base = _here()
    env_name = os.getenv("SHATTERED_REALM_INPUT", "").strip()
    if env_name:
        p = Path(env_name)
        if not p.is_absolute():
            p_script = base / p
            if p_script.exists():
                return p_script
        if p.exists():
            return p
    for name in CANDIDATE_NAMES:
        p = base / name
        if p.exists():
            return p
    for name in CANDIDATE_NAMES:
        p = Path.cwd() / name
        if p.exists():
            return p
    raise FileNotFoundError("Place one of: " + ", ".join(CANDIDATE_NAMES) + " next to this script.")
>>>>>>> c2aaf727a7e3c41aa7df2491c612be6f4d253009

# 4️⃣ Undo “Southward Turn” — Vertical flip
img = cv2.flip(img, 0)

<<<<<<< HEAD
# 5️⃣ Undo “Skewwright’s Sigil” — Reverse the affine skew
rows, cols, ch = img.shape
M = np.float32([[1, -0.2, 0], [0, 1, 0]])  # inverse of (1, 0.2)
img = cv2.warpAffine(img, M, (cols, rows))

# 6️⃣ Undo “Storm of Fifty Winds” — Normalize contrast & brightness
img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# 7️⃣ Undo “Seed of 42” — Median blur correction
img = cv2.medianBlur(img, 3)
=======
# ---------- Global reverse ops (A..F) ----------
def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

def inverse_shear(img: Image.Image, shear_x: float) -> Image.Image:
    w, h = img.size
    coeffs = (1.0, -shear_x, 0.0, 0.0, 1.0, 0.0)
    return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=0)

def swap_red_blue(arr: np.ndarray) -> np.ndarray:
    return arr[..., [2, 1, 0]]

def flip_vertical(arr: np.ndarray) -> np.ndarray:
    return arr[::-1, :, :]

def flip_horizontal(arr: np.ndarray) -> np.ndarray:
    return arr[:, ::-1, :]

def invert_colors(arr: np.ndarray) -> np.ndarray:
    return (255 - arr).astype(np.uint8)

def remove_storm_noise(arr: np.ndarray, seed: int, magnitude: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randint(-magnitude, magnitude + 1, size=arr.shape, dtype=np.int16)
    base = arr.astype(np.int16) - noise
    return np.clip(base, 0, 255).astype(np.uint8)

def common_reverse_prefix(img: Image.Image) -> np.ndarray:
    img = _ensure_rgb(img)
    a = inverse_shear(img, shear_x=SHEAR_X)
    arr = np.array(a)
    arr = swap_red_blue(arr)
    arr = flip_vertical(arr)
    arr = flip_horizontal(arr)
    arr = invert_colors(arr)
    arr = remove_storm_noise(arr, seed=SEED, magnitude=STORM_MAGNITUDE)
    return arr
# ----------------------------------------------
>>>>>>> c2aaf727a7e3c41aa7df2491c612be6f4d253009

# Save the restored base
cv2.imwrite("1_restored_realm.jpg", img)

<<<<<<< HEAD
# === SECOND PHASE: The Revealing Ritual ===
# Convert to grayscale for easier text detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("2_gray_realm.jpg", gray)
=======
# --------------- Tiling utils -----------------
def crop_to_multiple(arr: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    H, W, _ = arr.shape
    Hc = (H // tile_h) * tile_h
    Wc = (W // tile_w) * tile_w
    return arr[:Hc, :Wc, :]

def tiles_reshape(arr: np.ndarray, tile_h: int, tile_w: int) -> Tuple[np.ndarray, int, int]:
    arr = crop_to_multiple(arr, tile_h, tile_w)
    H, W, C = arr.shape
    rows, cols = H // tile_h, W // tile_w
    grid = arr.reshape(rows, tile_h, cols, tile_w, C).swapaxes(1, 2)  # (rows, cols, th, tw, C)
    return grid, rows, cols

def flatten_tiles(grid: np.ndarray) -> np.ndarray:
    rows, cols, th, tw, C = grid.shape
    return grid.reshape(rows * cols, th, tw, C)

def grid_from_tiles(tiles: np.ndarray, rows: int, cols: int) -> np.ndarray:
    th, tw, C = tiles.shape[1], tiles.shape[2], tiles.shape[3]
    grid = tiles.reshape(rows, cols, th, tw, C).swapaxes(1, 2)
    return grid.reshape(rows * th, cols * tw, C)

# Two possible forward permutation styles; provide inverses for both
def unshuffle_tiles_from_pull(tiles_new: np.ndarray, seed: int) -> np.ndarray:
    """
    Forward (pull): new[i] = old[perm[i]]
    Inverse: old[perm[i]] = new[i]
    """
    N = tiles_new.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    old = np.empty_like(tiles_new)
    for i in range(N):
        j = perm[i]
        old[j] = tiles_new[i]
    return old

def unshuffle_tiles_from_push(tiles_new: np.ndarray, seed: int) -> np.ndarray:
    """
    Forward (push): new[perm[i]] = old[i]
    Inverse: old[i] = new[perm[i]]   (i.e., apply argsort)
    """
    N = tiles_new.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(N)
    old = tiles_new[inv]
    return old
# ----------------------------------------------
>>>>>>> c2aaf727a7e3c41aa7df2491c612be6f4d253009

# Edge detection — “You are looking but can’t see”
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("3_edges_revealed.jpg", edges)

<<<<<<< HEAD
# Invert & enhance edges
inverted = cv2.bitwise_not(edges)
blurred = cv2.GaussianBlur(inverted, (3,3), 0)
enhanced = cv2.convertScaleAbs(blurred, alpha=2.5, beta=40)
cv2.imwrite("4_enhanced_hidden_text.jpg", enhanced)
=======
# ------------- Per-tile D8 orientation -------------
def d8_apply_tile(tile: np.ndarray, code: int) -> np.ndarray:
    if code == 0: return tile
    if code == 1: return np.rot90(tile, 1, axes=(0, 1)).copy()
    if code == 2: return np.rot90(tile, 2, axes=(0, 1)).copy()
    if code == 3: return np.rot90(tile, 3, axes=(0, 1)).copy()
    if code == 4: return tile[:, ::-1, :].copy()        # flipH
    if code == 5: return tile[::-1, :, :].copy()        # flipV
    if code == 6: return tile.transpose(1, 0, 2).copy() # transpose
    if code == 7: return tile.transpose(1, 0, 2)[::-1, :, :].copy()  # anti-diagonal
    raise ValueError("D8 code must be 0..7")

def d8_inverse(code: int) -> int:
    return {0:0, 1:3, 2:2, 3:1, 4:4, 5:5, 6:6, 7:7}[code]

def allowed_d8_codes_for_tile(tile_h: int, tile_w: int) -> List[int]:
    # If non-square tiles, restrict to ops that preserve (h,w)
    return [0,2,4,5] if tile_h != tile_w else list(range(8))

def undo_tile_orientations(tiles: np.ndarray, seed: int, tile_h: int, tile_w: int) -> np.ndarray:
    N = tiles.shape[0]
    allowed = allowed_d8_codes_for_tile(tile_h, tile_w)
    rng = np.random.RandomState(seed)
    out = tiles.copy()
    ops_idx = rng.randint(0, len(allowed), size=N)
    for i in range(N):
        code = allowed[ops_idx[i]]
        out[i] = d8_apply_tile(out[i], d8_inverse(code))
    return out
# ----------------------------------------------------
>>>>>>> c2aaf727a7e3c41aa7df2491c612be6f4d253009

# Apply XOR with the sacred Seed of 42
xor42 = cv2.bitwise_xor(gray, 42)
cv2.imwrite("5_xor42_reveal.jpg", xor42)

<<<<<<< HEAD
# Combine all visual layers for maximum contrast discovery
combined = cv2.addWeighted(enhanced, 0.7, xor42, 0.3, 0)
cv2.imwrite("6_final_reveal.jpg", combined)

print("✅ Restoration complete!")
print("Check the generated files (1_restored_realm.jpg … 6_final_reveal.jpg)")
print("The flag or hidden text should appear in one of them.")
=======
# ---------------- Grid-level D8 (positions) ----------
def d8_apply_grid(grid5: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        g = grid5
    elif code == 1:   # rot90
        g = np.rot90(grid5, 1, axes=(0, 1)).copy()
    elif code == 2:   # rot180
        g = np.rot90(grid5, 2, axes=(0, 1)).copy()
    elif code == 3:   # rot270
        g = np.rot90(grid5, 3, axes=(0, 1)).copy()
    elif code == 4:   # flipH
        g = grid5[:, ::-1, ...].copy()
    elif code == 5:   # flipV
        g = grid5[::-1, :, ...].copy()
    elif code == 6:   # transpose
        g = grid5.transpose(1, 0, 2, 3, 4).copy()
    elif code == 7:   # anti-diagonal
        g = grid5.transpose(1, 0, 2, 3, 4)[::-1, ...].copy()
    else:
        raise ValueError("Grid D8 code must be 0..7")
    return g
# ----------------------------------------------------


# ---------------- Scoring & helpers -----------------
def seam_error(arr: np.ndarray, tile_h: int, tile_w: int) -> float:
    H, W, _ = arr.shape
    vs = []
    for x in range(tile_w, W, tile_w):
        left = arr[:, x-1, :].astype(np.int16)
        right = arr[:, x, :].astype(np.int16)
        vs.append(np.abs(left - right).mean())
    hs = []
    for y in range(tile_h, H, tile_h):
        top = arr[y-1, :, :].astype(np.int16)
        bot = arr[y, :, :].astype(np.int16)
        hs.append(np.abs(top - bot).mean())
    return float((np.mean(vs) if vs else 0.0) + (np.mean(hs) if hs else 0.0))

def _gray(arr: np.ndarray) -> np.ndarray:
    return (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32)

def edge_score(arr: np.ndarray) -> float:
    g = _gray(arr)
    dx = np.abs(g[:,1:] - g[:,:-1]).mean()
    dy = np.abs(g[1:,:] - g[:-1,:]).mean()
    return float(dx + dy)

def combined_score(arr: np.ndarray, tile_h: int, tile_w: int) -> float:
    return -seam_error(arr, tile_h, tile_w) + 0.05 * edge_score(arr)
# ----------------------------------------------------


# ---------------- Variant runner --------------------
def run_grid_search_over_positions(tiles: np.ndarray, rows: int, cols: int, th: int, tw: int,
                                   do_shift: bool, do_gridd8: bool) -> tuple[str, np.ndarray, float]:
    best_name, best_img, best_score = None, None, -1e9
    base_grid = tiles.reshape(rows, cols, th, tw, tiles.shape[3])
    grid_codes = range(8) if do_gridd8 else [0]
    for gc in grid_codes:
        g = d8_apply_grid(base_grid, gc)
        riter = range(rows) if do_shift else [0]
        citer = range(cols) if do_shift else [0]
        for rs in riter:
            for cs in citer:
                gr = np.roll(np.roll(g, rs, axis=0), cs, axis=1)
                arr = gr.swapaxes(1, 2).reshape(rows*th, cols*tw, tiles.shape[3])
                sc = combined_score(arr, th, tw)
                name = f"gridD8_{gc}.shift_r{rs}_c{cs}"
                if sc > best_score:
                    best_name, best_img, best_score = name, arr, sc
    return best_name, best_img, best_score

def unshatter_try(arrF: np.ndarray, tile_h: int, tile_w: int, tag: str, base_dir: Path):
    grid, rows, cols = tiles_reshape(arrF, tile_h, tile_w)
    tiles = flatten_tiles(grid)

    # decide if we can afford grid search
    do_shift = ENABLE_SHIFT_SEARCH and (rows*cols <= MAX_GRID_FOR_SEARCH)
    do_gridd8 = ENABLE_GRID_D8_SEARCH and (rows*cols <= MAX_GRID_FOR_SEARCH)

    best_name, best_img, best_score = None, None, -1e9

    for perm_dir in TRY_PERM_DIRECTIONS:
        if perm_dir == "pull":
            unshuffle = unshuffle_tiles_from_pull
        else:
            unshuffle = unshuffle_tiles_from_push

        for orient_mode in TRY_TILE_ORIENT:
            # Build candidate tiles according to orient hypothesis
            if orient_mode == "none":
                tiles_cur = unshuffle(tiles, SEED)
            elif orient_mode == "after_unshuffle":
                tiles_unshuf = unshuffle(tiles, SEED)
                tiles_cur = undo_tile_orientations(tiles_unshuf, seed=SEED, tile_h=tile_h, tile_w=tile_w)
            elif orient_mode == "before_unshuffle":
                tiles_orient = undo_tile_orientations(tiles, seed=SEED, tile_h=tile_h, tile_w=tile_w)
                tiles_cur = unshuffle(tiles_orient, SEED)
            else:
                continue

            # Optional grid search (positions)
            name_pos, img_pos, score_pos = run_grid_search_over_positions(
                tiles_cur, rows, cols, tile_h, tile_w, do_shift, do_gridd8
            )

            label = f"{tag}.{perm_dir}.{orient_mode}.{name_pos}"
            if SAVE_VARIANTS:
                Image.fromarray(img_pos).save(base_dir / f"{OUTPUT_BASENAME}.{label}.png")

            if score_pos > best_score:
                best_name, best_img, best_score = label, img_pos, score_pos

    return best_name, best_img, best_score
# ----------------------------------------------------


# ---------------- Helpers (visuals) ----------------
def helpers_for(img: Image.Image, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ImageOps.grayscale(img).save(out_dir / "restored_gray.png")
    sharp = ImageOps.autocontrast(img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=130, threshold=2)), cutoff=1)
    sharp.save(out_dir / "restored_sharp_autocontrast.png")
    img.filter(ImageFilter.FIND_EDGES).save(out_dir / "restored_edges.png")
    small = img.resize((max(1, img.width//4), max(1, img.height//4)), Image.BILINEAR)
    small.resize(img.size, Image.NEAREST).save(out_dir / "restored_squint.png")
# ----------------------------------------------------


# ---------------- Main ------------------------------
def main():
    base = _here()
    inp = resolve_input_path()
    print(f"[i] Opening image: {inp}")

    src = Image.open(str(inp))
    arrF = common_reverse_prefix(src)

    candidates = []

    # Pixel tile sizes (4/8/16/32 px)
    for ts in TILE_SIZES_PX:
        name, img, score = unshatter_try(arrF, ts, ts, f"px{ts}", base)
        candidates.append((name, img, score))

    # 16x16 grid interpretation (may be non-square tiles)
    H, W, _ = arrF.shape
    tr, tc = TRY_GRID_RC
    th, tw = max(1, H // tr), max(1, W // tc)
    arrF2 = crop_to_multiple(arrF, th, tw)
    name, img, score = unshatter_try(arrF2, th, tw, "grid16x16", base)
    candidates.append((name, img, score))

    # Pick best by score
    winner_name, winner_img, winner_score = max(candidates, key=lambda x: x[2])
    out_final = base / f"{OUTPUT_BASENAME}.png"
    Image.fromarray(winner_img).save(out_final)
    print(f"[✓] Winner: {winner_name}  (score={winner_score:.3f})")
    print(f"[✓] Saved:  {out_final}")

    # Save A..F intermediates for transparency
    if SAVE_STEPS:
        a = inverse_shear(_ensure_rgb(src), shear_x=SHEAR_X)
        sA = np.array(a); sB = swap_red_blue(sA); sC = flip_vertical(sB)
        sD = flip_horizontal(sC); sE = invert_colors(sD); sF = remove_storm_noise(sE, seed=SEED, magnitude=STORM_MAGNITUDE)
        base_noext = out_final.with_suffix("")
        Image.fromarray(sA).save(base_noext.with_suffix(".A_inverse_shear.png"))
        Image.fromarray(sB).save(base_noext.with_suffix(".B_swapRB.png"))
        Image.fromarray(sC).save(base_noext.with_suffix(".C_vflip.png"))
        Image.fromarray(sD).save(base_noext.with_suffix(".D_hflip.png"))
        Image.fromarray(sE).save(base_noext.with_suffix(".E_invert.png"))
        Image.fromarray(sF).save(base_noext.with_suffix(".F_remove_noise.png"))
        print("    Saved A..F intermediate steps.")

    if HELPERS_DIR:
        helpers_for(Image.fromarray(winner_img), base / HELPERS_DIR)
        print(f"    Helper images in: {base / HELPERS_DIR}")

if __name__ == "__main__":
    main()
>>>>>>> c2aaf727a7e3c41aa7df2491c612be6f4d253009
