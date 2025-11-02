# extract_mouth_roi.py
from pathlib import Path
import cv2
import time

VIDEO_PATH = Path("data/input.mp4")
WORK_DIR = Path("work_lipread")
MOUTH_SIZE = 112

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def mouth_box(face_box):
    x, y, w, h = face_box
    top = int(y + 0.55 * h)
    bottom = int(y + 0.95 * h)
    left = x
    right = x + w
    return left, top, right, bottom

def pick_biggest_face(faces):
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces[0]

def read_segments():
    seg_file = WORK_DIR / "segments.txt"
    segs = []
    with seg_file.open() as f:
        for line in f:
            s, e, *_ = line.strip().split(",")
            segs.append((int(s), int(e)))
    segs.sort(key=lambda x: x[0])
    return segs

def progress_bar(curr, total, prefix=""):
    width = 30
    ratio = curr / total if total else 0
    done = int(ratio * width)
    bar = "[" + "#" * done + "-" * (width - done) + "]"
    print(f"\r{prefix} {bar} {curr}/{total}", end="", flush=True)

def extract():
    segs = read_segments()
    if len(segs) == 0:
        print("[extract_mouth_roi] no segments to extract")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    for i, _ in enumerate(segs):
        (WORK_DIR / f"segment_{i:03d}").mkdir(parents=True, exist_ok=True)

    print("[extract_mouth_roi] start")
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        biggest = pick_biggest_face(faces)

        for i, (s, e) in enumerate(segs):
            if frame_idx < s or frame_idx > e:
                continue

            if biggest is None:
                continue

            mx1, my1, mx2, my2 = mouth_box(biggest)
            mx1 = max(mx1, 0)
            my1 = max(my1, 0)
            mx2 = min(mx2, gray.shape[1])
            my2 = min(my2, gray.shape[0])

            mouth = gray[my1:my2, mx1:mx2]
            if mouth.size == 0:
                continue
            mouth = cv2.resize(mouth, (MOUTH_SIZE, MOUTH_SIZE))
            out_dir = WORK_DIR / f"segment_{i:03d}"
            out_path = out_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), mouth)

        now = time.time()
        if now - last_print > 0.3:
            progress_bar(frame_idx, total_frames, prefix="[extract_mouth_roi]")
            last_print = now

        frame_idx += 1

    cap.release()
    progress_bar(total_frames, total_frames, prefix="[extract_mouth_roi]")
    print()
    print("[extract_mouth_roi] done")

if __name__ == "__main__":
    extract()
