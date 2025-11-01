#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

IMG_SIZE = 400
FRAME_STEP = 5          # read every 5th frame
MIN_FRAMES = 2000       # process at least this
IDLE_WINDOW = 800       # how many processed frames to watch for new pixels
IDLE_NEW_PX = 15        # if in the window we add less than this → stop
MAX_FRAMES = 40000      # hard upper bound per video

def detect_cyan_car_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cyan / turquoise
    lower = np.array([80, 40, 40], dtype=np.uint8)
    upper = np.array([110, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 3)
    return mask

def get_center(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

def process_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"fail open {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or IMG_SIZE)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or IMG_SIZE)
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[{video_path.name}] reported frames: {total_reported}")

    visited = np.zeros((h, w), dtype=np.uint8)
    positions = []

    frame_idx = 0         # source frames
    proc_idx = 0          # processed frames
    last_counts = []      # for idle detection
    stopped_by_idle = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # skip frames
        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        mask = detect_cyan_car_mask(frame)
        center = get_center(mask)

        if center is None:
            if positions:
                cx, cy = positions[-1]
            else:
                cx, cy = w // 2, h // 2
        else:
            cx, cy = center

        positions.append((cx, cy))

        # clamp
        cx = max(0, min(w-1, cx))
        cy = max(0, min(h-1, cy))
        # mark visited
        visited[cy, cx] = 1

        proc_idx += 1
        frame_idx += 1

        if proc_idx % 300 == 0:
            visited_count = int(visited.sum())
            last_counts.append(visited_count)
            # keep last N
            if len(last_counts) > IDLE_WINDOW // 300 + 2:
                last_counts.pop(0)

            if proc_idx >= MIN_FRAMES:
                if len(last_counts) >= 2:
                    delta = last_counts[-1] - last_counts[0]
                    if delta < IDLE_NEW_PX:
                        print(f"\n[{video_path.name}] no new area → stop (Δpx={delta})")
                        stopped_by_idle = True
                        break

        if proc_idx >= MAX_FRAMES:
            print(f"\n[{video_path.name}] reached MAX_FRAMES={MAX_FRAMES}")
            break

        # progress line
        if total_reported > 0:
            pct = frame_idx / total_reported * 100
            print(f"\r[{video_path.name}] src {frame_idx}/{total_reported} ({pct:5.1f}%) processed {proc_idx}", end="")
        else:
            print(f"\r[{video_path.name}] processed {proc_idx}", end="")

    print()
    cap.release()

    # draw
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, '-', color='yellow', linewidth=1)
    plt.plot(xs[0], ys[0], 'go', markersize=10, label='start')
    plt.plot(xs[-1], ys[-1], 'ro', markersize=10, label='end')
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Car Trajectory - {video_path.name}")
    out_img = video_path.parent / f"{video_path.stem}_trajectory.png"
    plt.savefig(out_img, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[{video_path.name}] saved {out_img}  | points: {len(positions)}  | idle_stop={stopped_by_idle}")

def process_folder(folder: Path):
    vids = sorted(folder.glob("*.mp4"))
    if not vids:
        print(f"no videos in {folder}")
        return
    for v in vids:
        print("\n" + "="*50)
        print(f"processing {v.name}")
        print("="*50)
        process_video(v)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_folder(Path(sys.argv[1]))
    else:
        process_folder(Path("."))
