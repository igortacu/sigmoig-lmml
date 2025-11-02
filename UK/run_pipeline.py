# run_pipeline.py
import subprocess
import sys

steps = [
    ("[1/3] detect & track", ["python", "detect_and_track.py"]),
    ("[2/3] extract mouth ROI", ["python", "extract_mouth_roi.py"]),
    ("[3/3] VSR infer", ["python", "vsr_infer.py"]),
]

for label, cmd in steps:
    print(f"\n{label}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"{label} failed with code {r.returncode}")
        sys.exit(r.returncode)

print("\nAll steps finished.")
