import os, csv
from PIL import Image, ImageFilter
import numpy as np

CSV_CANDIDATES = [
    "data/task_32/train_labels.csv",
    "data/train_labels.csv",
    "train_labels.csv",
    "data/task_32/images/train/labels.csv",
]

IMG_DIRS = [
    "data/task_32/train",
    "data/task_32/images/train/images",
    "data/train",
]

OUT_DIR = "data/char_ds"
os.makedirs(OUT_DIR, exist_ok=True)

ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def find_csv():
    for p in CSV_CANDIDATES:
        if os.path.exists(p):
            print("[info] csv:", p)
            return p
    raise FileNotFoundError("no train_labels.csv found")


def find_img(fname):
    for d in IMG_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    if fname.startswith("train_"):
        alt = fname.replace("train_", "test_", 1)
        for d in IMG_DIRS:
            p = os.path.join(d, alt)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"no image for {fname}")


def preprocess(pil_img, target_h=64):
    img = pil_img.convert("L")
    w, h = img.size
    new_w = int(w * (target_h / h))
    img = img.resize((new_w, target_h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return img


def seg_8(img_np):
    H, W = img_np.shape
    thr = 200
    binm = (img_np < thr).astype(np.uint8)
    proj = binm.sum(axis=0)
    smooth = np.convolve(proj, np.ones(5) / 5, mode="same")

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
        boxes.append((lx, 0, rx, H))
    return boxes


def save_crop(char_label, pil_crop, idx):
    sub = os.path.join(OUT_DIR, char_label)
    os.makedirs(sub, exist_ok=True)
    pil_crop.save(os.path.join(sub, f"{idx:07d}.png"))


def main():
    csv_path = find_csv()
    idx = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            label = row["answer"].strip()
            img_path = find_img(fname)

            pil = Image.open(img_path)
            pil = preprocess(pil, 64)
            img_np = np.array(pil)

            boxes = seg_8(img_np)
            if len(label) < 8:
                label = label + "_" * (8 - len(label))

            for pos, box in enumerate(boxes):
                ch = label[pos]
                if ch == "_" or ch not in ALPH:
                    continue
                x0, y0, x1, y1 = box
                crop = pil.crop((x0, y0, x1, y1)).resize((64, 64), Image.BILINEAR)
                save_crop(ch, crop, idx)
                idx += 1

    print("[done] total crops:", idx)


if __name__ == "__main__":
    main()
