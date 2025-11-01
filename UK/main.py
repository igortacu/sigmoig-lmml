import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Optional OCR pass (only if pytesseract is available on the system)
try:
    import pytesseract  # Requires Tesseract installed on the system (optional)
    OCR_OK = True
except Exception:
    OCR_OK = False


def find_video_file(base: Path) -> Path:
    for ext in ("*.mp4"):
        files = sorted(base.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError("No video file found in the UK folder.")


def detect_faces_haar(gray: np.ndarray, face_cascade: cv2.CascadeClassifier):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(80, 80),
    )
    # return largest face first
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces


def mouth_roi_from_face(face: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x, y, w, h = face
    # Heuristic: bottom-central region of the face
    mx = x + int(0.2 * w)
    my = y + int(0.60 * h)
    mw = int(0.60 * w)
    mh = int(0.35 * h)
    mx = max(0, mx)
    my = max(0, my)
    x2 = min(W, mx + mw)
    y2 = min(H, my + mh)
    return mx, my, x2 - mx, y2 - my


def detect_speaking_segments(energies: List[float], fps: float, thr_scale: float = 2.0,
                             min_dur: float = 0.5, max_gap: float = 0.2) -> List[Tuple[int, int]]:
    if not energies:
        return []
    arr = np.array(energies, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-9
    thr = med + thr_scale * mad

    active = arr > thr
    segments = []
    start = None
    gap = 0
    max_gap_frames = int(max_gap * fps)
    min_len_frames = int(min_dur * fps)

    for i, is_on in enumerate(active):
        if is_on:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > max_gap_frames:
                    end = i - gap + 1
                    if end - start >= min_len_frames:
                        segments.append((start, end))
                    start, gap = None, 0
    if start is not None:
        end = len(active)
        if end - start >= min_len_frames:
            segments.append((start, end))
    return segments


def try_ocr_flag(img: np.ndarray) -> str | None:
    if not OCR_OK:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        try:
            text = pytesseract.image_to_string(gray, config="--psm 6")
        except Exception:
            # Tesseract not installed or runtime OCR error; skip OCR
            return None
        m = re.search(r"FLAG\{[^}]+\}", text, re.IGNORECASE)
        return m.group(0) if m else None
    except Exception:
        # Any unexpected image processing error -> skip OCR
        return None


def process_video(video_path: Path, out_dir: Path, detect_every: int = 1) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {face_cascade_path}")

    energies: List[float] = []
    rois = []  # (frame_idx, roi_bgr, full_frame_bgr)
    prev_roi_small = None
    roi_size = (96, 64)  # (w, h)

    frame_idx = 0
    found_flag = None
    last_face = None  # (x,y,w,h)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face periodically or if we have none
        if last_face is None or (frame_idx % detect_every == 0):
            faces = detect_faces_haar(gray_full, face_cascade)
            last_face = faces[0] if faces else None

        if last_face is not None:
            mx, my, mw, mh = mouth_roi_from_face(last_face, W, H)
            roi = frame[my:my + mh, mx:mx + mw].copy()
            if roi.size == 0:
                energies.append(0.0)
            else:
                roi_small = cv2.resize(roi, roi_size, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
                if prev_roi_small is None:
                    e = 0.0
                else:
                    e = float(np.mean(cv2.absdiff(gray, prev_roi_small)))
                prev_roi_small = gray
                energies.append(e)
                rois.append((frame_idx, roi, frame))

                if found_flag is None:
                    maybe = try_ocr_flag(frame)
                    if maybe:
                        found_flag = maybe
        else:
            energies.append(0.0)

        frame_idx += 1

    cap.release()
    segments = detect_speaking_segments(energies, fps)

    saved = []
    for si, (s, e) in enumerate(segments, 1):
        seg_dir = out_dir / f"segment_{si:02d}"
        seg_dir.mkdir(exist_ok=True, parents=True)

        chosen = [(idx, roi, full) for (idx, roi, full) in rois if s <= idx < e]
        if not chosen:
            continue

        h, w = chosen[0][1].shape[:2]
        vid_path = seg_dir / "mouth_preview.mp4"
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for j, (idx, roi, full) in enumerate(chosen):
            img_path = seg_dir / f"roi_{j:05d}.png"
            cv2.imwrite(str(img_path), roi)
            writer.write(roi)
        writer.release()

        if OCR_OK and found_flag is None:
            for _, _, full in chosen:
                maybe = try_ocr_flag(full)
                if maybe:
                    found_flag = maybe
                    break

        saved.append({"index": si, "start_frame": int(s), "end_frame": int(e),
                      "start_time": round(s / fps, 3), "end_time": round(e / fps, 3),
                      "dir": str(seg_dir)})

    meta = {
        "video": str(video_path),
        "fps": fps,
        "segments": saved,
        "energies_len": len(energies),
        "flag": found_flag,
    }
    (out_dir / "segments.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Lip-motion segmenter without MediaPipe (UK video).")
    parser.add_argument("--video", type=str, default=None, help="Path to video. If omitted, first video in folder is used.")
    parser.add_argument("--out", type=str, default="out_segments", help="Output directory.")
    parser.add_argument("--detect-every", type=int, default=1, help="Redetect face every N frames (default 1).")
    args = parser.parse_args()

    base = Path(__file__).parent
    video = Path(args.video) if args.video else find_video_file(base)
    if not video.is_absolute():
        video = (base / video).resolve()
    out_dir = (base / args.out).resolve()

    meta = process_video(video, out_dir, detect_every=args.detect_every)

    print(f"Processed: {video}")
    print(f"Segments saved in: {out_dir}")
    if meta.get("flag"):
        print(f"Detected FLAG (via OCR): {meta['flag']}")
    else:
        print("No FLAG detected automatically. Inspect the segment folders and reconstruct manually.")


if __name__ == "__main__":
    raise SystemExit(main())