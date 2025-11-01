import cv2
import numpy as np
import glob
import os

CANVAS_PATH = "input_flag.png"
APPLE_DIR = "apple_images"

canvas = cv2.imread(CANVAS_PATH)
if canvas is None:
    raise RuntimeError("input_flag.png not found")

H, W = canvas.shape[:2]

apple_files = sorted(glob.glob(os.path.join(APPLE_DIR, "*.*")))
if not apple_files:
    raise RuntimeError("no images in apple_images/")

os.makedirs("out_runs", exist_ok=True)

# 1) different hsv configs (tight → loose)
HSV_CONFIGS = [
    # (low1, high1, low2, high2)
    (np.array([0, 90, 90]),  np.array([10, 255, 255]),
     np.array([170, 90, 90]), np.array([180, 255, 255])),
    (np.array([0, 70, 70]),  np.array([12, 255, 255]),
     np.array([168, 70, 70]), np.array([180, 255, 255])),
    (np.array([0, 50, 50]),  np.array([12, 255, 255]),
     np.array([168, 50, 50]), np.array([180, 255, 255])),
]

# 2) area ranges (fraction of image)
AREA_CONFIGS = [
    (0.002, 0.05),
    (0.002, 0.035),
    (0.003, 0.03),
    (0.004, 0.025),
]

# 3) merge modes
MERGE_MODES = ["or", "and"]  # union vs intersection

run_id = 0

for hsv_idx, (l1, h1, l2, h2) in enumerate(HSV_CONFIGS):
    for area_idx, (min_frac, max_frac) in enumerate(AREA_CONFIGS):
        for merge_mode in MERGE_MODES:

            global_mask = None  # init later

            for path in apple_files:
                img = cv2.imread(path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                img_area = h * w

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv, l1, h1)
                m2 = cv2.inRange(hsv, l2, h2)
                mask = cv2.bitwise_or(m1, m2)

                # clean
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # per-component overlay
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < min_frac * img_area:
                        continue
                    if area > max_frac * img_area:
                        continue

                    comp_mask = np.zeros((h, w), np.uint8)
                    cv2.drawContours(comp_mask, [c], -1, 255, -1)

                    # small grow
                    comp_mask = cv2.dilate(comp_mask, np.ones((3, 3), np.uint8), iterations=1)

                    comp_big = cv2.resize(comp_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                    if global_mask is None:
                        global_mask = comp_big
                    else:
                        if merge_mode == "or":
                            global_mask = cv2.bitwise_or(global_mask, comp_big)
                        else:  # "and"
                            global_mask = cv2.bitwise_and(global_mask, comp_big)

            if global_mask is None:
                continue

            result = canvas.copy()
            result[global_mask == 255] = (0, 0, 0)

            out_name = f"out_runs/flag_h{hsv_idx}_a{area_idx}_{merge_mode}.png"
            cv2.imwrite(out_name, result)
            print("saved", out_name)
            run_id += 1
