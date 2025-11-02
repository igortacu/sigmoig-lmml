import os
import numpy as np
from PIL import Image, ImageFilter
from seed_labels import SEED

TEST_DIR = "data/task_32/images"
OUT_DIR = "data/seed_chars"
os.makedirs(OUT_DIR, exist_ok=True)

ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def preprocess(img: Image.Image, target_h=64) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    new_w = int(w * (target_h / h))
    img = img.resize((new_w, target_h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return img

def segment_8(img_np: np.ndarray):
    H, W = img_np.shape
    thr = 200
    binm = (img_np < thr).astype(np.uint8)
    proj = binm.sum(axis=0)
    smooth = np.convolve(proj, np.ones(5)/5, mode="same")

    boxes = []
    step = W / 8.0
    for i in range(8):
        x0 = int(i * step)
        x1 = int((i + 1) * step)
        if x1 <= x0:
            x1 = x0 + 1
        sl = smooth[x0:x1]
        if sl.max() == 0:
            boxes.append((x0, 0, x1, H))
            continue
        nz = np.where(sl > 0)[0]
        lx = nz[0] + x0
        rx = nz[-1] + x0 + 1
        lx = max(0, lx - 1)
        rx = min(W, rx + 1)
        boxes.append((lx, 0, rx, H))
    return boxes

idx = 0
for fname, label in SEED.items():
    path = os.path.join(TEST_DIR, fname)
    assert os.path.exists(path), f"missing {path}"
    pil = Image.open(path)
    pil = preprocess(pil, 64)
    img_np = np.array(pil)

    boxes = segment_8(img_np)
    if len(label) < 8:
        label = label + "_" * (8 - len(label))

    for pos, box in enumerate(boxes):
        ch = label[pos]
        if ch == "_" or ch not in ALPH:
            continue
        x0, y0, x1, y1 = box
        crop = pil.crop((x0, y0, x1, y1)).resize((64, 64), Image.BILINEAR)

        out_dir = os.path.join(OUT_DIR, ch)
        os.makedirs(out_dir, exist_ok=True)
        crop.save(os.path.join(out_dir, f"{idx:07d}.png"))
        idx += 1

print(f"[ok] saved {idx} char crops in {OUT_DIR}")
