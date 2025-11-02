# detect_and_track.py
from pathlib import Path
import cv2
import numpy as np
import sys
import time

# ----- config -----
VIDEO_PATH = Path("data/input.mp4")
WORK_DIR = Path("work_lipread")
WORK_DIR.mkdir(parents=True, exist_ok=True)

FPS_ANALYSIS = 5
MIN_SEG_FRAMES = 20
MOTION_THRESH = 6.0
MAX_SEGMENTS_TO_KEEP = 10
# -------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def pick_biggest_face(faces):
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces[0]

def mouth_box(face_box):
    x, y, w, h = face_box
    top = int(y + 0.55 * h)
    bottom = int(y + 0.95 * h)
    left = x
    right = x + w
    return left, top, right, bottom

def progress_bar(curr, total, prefix=""):
    width = 30
    ratio = curr / total if total else 0
    done = int(ratio * width)
    bar = "[" + "#" * done + "-" * (width - done) + "]"
    print(f"\r{prefix} {bar} {curr}/{total}", end="", flush=True)

def process_video():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(base_fps // FPS_ANALYSIS))

    frame_idx = 0
    prev_mouth = None

    raw_segments = []
    active = False
    seg_start = None
    scores_for_seg = []

    print("[detect_and_track] start")
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # progress print (once per 0.3s)
        now = time.time()
        if now - last_print > 0.3:
            progress_bar(frame_idx, total_frames, prefix="[detect_and_track]")
            last_print = now

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        mouth_motion = 0.0
        biggest = pick_biggest_face(faces)
        if biggest is not None:
            mx1, my1, mx2, my2 = mouth_box(biggest)
            mx1 = max(mx1, 0)
            my1 = max(my1, 0)
            mx2 = min(mx2, gray.shape[1])
            my2 = min(my2, gray.shape[0])
            mouth_roi = gray[my1:my2, mx1:mx2]

            if mouth_roi.size > 0:
                if prev_mouth is None:
                    prev_mouth = mouth_roi
                prev_mouth_r = cv2.resize(prev_mouth, (mouth_roi.shape[1], mouth_roi.shape[0]))
                diff = cv2.absdiff(mouth_roi, prev_mouth_r)
                mouth_motion = float(np.mean(diff))
                prev_mouth = mouth_roi

        speaking = mouth_motion > MOTION_THRESH

        if speaking and not active:
            active = True
            seg_start = frame_idx
            scores_for_seg = [mouth_motion]
        elif speaking and active:
            scores_for_seg.append(mouth_motion)
        elif (not speaking) and active:
            seg_end = frame_idx
            length = seg_end - seg_start
            if length >= MIN_SEG_FRAMES:
                avg_score = float(np.mean(scores_for_seg))
                raw_segments.append((seg_start, seg_end, avg_score))
            active = False
            seg_start = None
            scores_for_seg = []

        frame_idx += 1

    cap.release()
    progress_bar(total_frames, total_frames, prefix="[detect_and_track]")
    print()  # newline

    raw_segments.sort(key=lambda x: x[2], reverse=True)
    raw_segments = raw_segments[:MAX_SEGMENTS_TO_KEEP]

    out_path = WORK_DIR / "segments.txt"
    with out_path.open("w") as f:
        for s, e, score in raw_segments:
            f.write(f"{s},{e},{score:.3f}\n")

    print(f"[detect_and_track] saved {len(raw_segments)} segments to {out_path}")

if __name__ == "__main__":
    process_video()
