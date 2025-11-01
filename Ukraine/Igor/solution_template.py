# yolo_solution.py
#!/usr/bin/env python3
"""
Car Tracking Challenge - YOLO-based solution
- detect vehicle with YOLO
- track center
- detect letters from frame
- choose closest letter
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO


def detect_letters_from_frame(frame, debug_path=None):
    import pytesseract

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 3)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 200:
            continue

        roi = gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(
            roi,
            config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ).strip()

        if len(text) == 1:
            letters.append({
                "text": text,
                "bbox": (x, y, w, h),
                "center": (x + w // 2, y + h // 2),
            })

    if debug_path is not None:
        dbg = frame.copy()
        for L in letters:
            x, y, w, h = L["bbox"]
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(dbg, L["text"], (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(debug_path), dbg)

    return letters


class YOLOCarTracker:
    def __init__(self, video_path, model_name="yolov8n.pt"):
        self.video_path = Path(video_path)
        self.model = YOLO(model_name)
        self.cap = cv2.VideoCapture(str(self.video_path))

        self.frames = []
        self.positions = []
        self.bboxes = []

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def load_video(self):
        self.frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        self.cap.release()
        print(f"[{self.video_path.name}] frames: {len(self.frames)}")

    def track(self, conf_th=0.25):
        vehicles = [2, 3, 5, 7]  # COCO ids
        for idx, frame in enumerate(self.frames):
            res = self.model(frame, verbose=False)[0]
            best_box = None
            best_conf = 0.0

            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in vehicles and conf > conf_th and conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                self.positions.append((cx, cy))
                self.bboxes.append((int(x1), int(y1), int(x2), int(y2)))
            else:
                if self.positions:
                    self.positions.append(self.positions[-1])
                    self.bboxes.append(self.bboxes[-1])
                else:
                    self.positions.append((self.width // 2, self.height // 2))
                    self.bboxes.append((0, 0, 0, 0))

            if (idx + 1) % 10 == 0:
                print(f"{idx+1}/{len(self.frames)} frames")

        print(f"[{self.video_path.name}] tracking done, points: {len(self.positions)}")

    def visualize_trajectory(self, save_path=None):
        if not self.positions:
            print("no positions")
            return

        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]

        plt.figure(figsize=(8, 8))
        if self.frames:
            plt.imshow(cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2RGB), alpha=0.25)

        for i in range(len(xs) - 1):
            alpha = (i + 1) / len(xs)
            plt.plot(xs[i:i+2], ys[i:i+2], "c-", linewidth=2, alpha=alpha)

        plt.plot(xs[0], ys[0], "go", markersize=12, label="start")
        plt.plot(xs[-1], ys[-1], "ro", markersize=12, label="end")
        plt.xlim(0, self.width)
        plt.ylim(self.height, 0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f"Car Trajectory - {self.video_path.name}")
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
            print(f"saved: {save_path}")
        plt.show()

    def deduce_letter(self):
        if not self.frames or not self.positions:
            return "?"
        first_frame = self.frames[0]
        dbg_path = self.video_path.parent / f"{self.video_path.stem}_letters.png"
        letters = detect_letters_from_frame(first_frame, debug_path=dbg_path)

        if not letters:
            print("no letters detected")
            return "?"

        end_x, end_y = self.positions[-1]
        best = None
        best_d = 1e9
        for L in letters:
            lx, ly = L["center"]
            d = (end_x - lx) ** 2 + (end_y - ly) ** 2
            if d < best_d:
                best_d = d
                best = L["text"]

        print(f"[{self.video_path.name}] endpoint → {best}  (dist={best_d**0.5:.1f})")
        return best


def process_all_videos(video_dir="./videos", model_name="yolov8n.pt"):
    video_dir = Path(video_dir)
    vids = sorted(video_dir.glob("*.mp4"))
    if not vids:
        print("no videos")
        return

    letters_out = []
    for v in vids:
        print("\n" + "="*60)
        print(f"processing {v.name}")
        print("="*60)
        tr = YOLOCarTracker(v, model_name=model_name)
        tr.load_video()
        tr.track(conf_th=0.25)
        traj_img = v.parent / f"{v.stem}_traj.png"
        tr.visualize_trajectory(save_path=traj_img)
        L = tr.deduce_letter()
        letters_out.append(L)

    flag = "".join(letters_out)
    print("\n--- RESULT ---")
    for v, L in zip(vids, letters_out):
        print(f"{v.name:20s} -> {L}")
    print("\nFlag options:")
    print(f"FLAG{{{flag}}}")
    print(f"CTF{{{flag}}}")
    print(f"SIGMOID{{{flag}}}")
    print(flag.lower())


if __name__ == "__main__":
    process_all_videos("./videos", model_name="yolov8n.pt")
