# infer.py  (consistent with train_char.py)
import os, csv
from PIL import Image, ImageFilter
import numpy as np
import torch
import torchvision.transforms as T

TEST_DIR = "data/task_32/images"
CKPT_PATH = "char_cnn.pt"
OUT_CSV = "submission.csv"

# ----- model definition must match train_char.py -----
class CharNet(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(128 * 16 * 16, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# -----------------------------------------------------

ckpt = torch.load(CKPT_PATH, map_location="cpu")
classes = ckpt["classes"]
model = CharNet(len(classes))
model.load_state_dict(ckpt["model"])
model.eval()

tf = T.Compose([
    T.Grayscale(),
    T.Resize((64, 64)),
    T.ToTensor(),
])


def preprocess(pil_img, h=64):
    img = pil_img.convert("L")
    w, old_h = img.size
    new_w = int(w * (h / old_h))
    img = img.resize((new_w, h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return img


def segment_8(pil_img):
    img_np = np.array(pil_img)
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
        # small padding
        lx = max(lx - 1, 0)
        rx = min(rx + 1, W)
        boxes.append((lx, 0, rx, H))
    return boxes


rows = [("filename", "answer")]

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.endswith(".png"):
        continue

    pil = Image.open(os.path.join(TEST_DIR, fname))
    pil = preprocess(pil, 64)
    boxes = segment_8(pil)

    chars = []
    for (x0, y0, x1, y1) in boxes:
        crop = pil.crop((x0, y0, x1, y1)).resize((64, 64))
        tens = tf(crop).unsqueeze(0)
        with torch.no_grad():
            out = model(tens)
        idx = out.argmax(1).item()
        chars.append(classes[idx])

    rows.append((fname, "".join(chars)))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

print("submission.csv written.")
