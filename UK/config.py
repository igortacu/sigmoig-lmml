# config.py
from pathlib import Path

VIDEO_PATH = Path("data/input.mp4")     # 50-minute clip
WORK_DIR = Path("work_lipread")
WORK_DIR.mkdir(parents=True, exist_ok=True)

FPS_ANALYSIS = 5               # light pass
MIN_SEG_FRAMES = 20            # drop micro-movements
LIP_Z_THRESH = 0.7
MOTION_THRESH = 4.0

MOUTH_SIZE = 112
FINAL_FPS = 25
MAX_SEGMENTS_TO_KEEP = 10
TARGET_SEGMENTS = 3
