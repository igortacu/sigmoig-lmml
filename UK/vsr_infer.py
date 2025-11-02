# vsr_infer.py
from pathlib import Path
import torch
import cv2
import numpy as np
from config import WORK_DIR, MOUTH_SIZE, TARGET_SEGMENTS

def load_avhubert_model():
    # placeholder
    # load your real model here
    class Dummy(torch.nn.Module):
        def forward(self, x):
            # x: [1, T, 1, H, W]
            return ["DUMMYFLAG"]
    return Dummy()

def frames_to_tensor(frames):
    # frames: list of (H,W) gray
    arr = np.stack(frames, axis=0)  # [T, H, W]
    arr = arr.astype(np.float32) / 255.0
    arr = arr[:, None, :, :]       # [T, 1, H, W]
    arr = torch.from_numpy(arr).unsqueeze(0)  # [1, T, 1, H, W]
    return arr

def read_segment_frames(seg_dir: Path):
    files = sorted(seg_dir.glob("*.png"))
    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames.append(img)
    return frames

def infer_all():
    model = load_avhubert_model()
    model.eval()

    results = []
    seg_dirs = sorted([d for d in WORK_DIR.iterdir() if d.is_dir() and d.name.startswith("segment_")])
    for idx, seg_dir in enumerate(seg_dirs[:TARGET_SEGMENTS]):
        frames = read_segment_frames(seg_dir)
        if len(frames) == 0:
            results.append((idx, ""))
            continue
        x = frames_to_tensor(frames)
        with torch.no_grad():
            out = model(x)
        # out is placeholder
        text = out[0] if isinstance(out, list) else "SEG"
        results.append((idx, text))

    # build flag
    results.sort(key=lambda x: x[0])
    flag = "".join([t for _, t in results])
    flag_path = WORK_DIR / "flag.txt"
    flag_path.write_text(flag, encoding="utf-8")
    print(f"[vsr_infer] FLAG -> {flag}")

if __name__ == "__main__":
    infer_all()
