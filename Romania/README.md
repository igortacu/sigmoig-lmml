# Romania — Cola/Fanta/Sprite Classification (YOLOv8)

This folder trains a YOLOv8 classification model to recognize three soft drink brands: Cola, Fanta, and Sprite.

Output artifact: `Romania/model.pt` (YOLOv8 format), suitable for evaluation.

## Dataset

Place your images under `Romania/data/coke_fanta_sprite/` using one of these layouts:

- Flat class folders:
  - `Romania/data/coke_fanta_sprite/cola/*.jpg`
  - `Romania/data/coke_fanta_sprite/fanta/*.jpg`
  - `Romania/data/coke_fanta_sprite/sprite/*.jpg`

- Or inside a subdirectory like `source/`, `raw/`, `images/`, `all/`, or `dataset/`:
  - `Romania/data/coke_fanta_sprite/source/cola/*.jpg`
  - `Romania/data/coke_fanta_sprite/source/fanta/*.jpg`
  - `Romania/data/coke_fanta_sprite/source/sprite/*.jpg`

If `train/`, `val/`, `test/` splits are not already present, the training script will automatically create an 80/10/10 split.

## Setup

- Requirements (Python 3.10+): see `Romania/requirements.txt`.
- Recommended to use CPU or a CUDA GPU if available.

Install dependencies (Windows PowerShell):

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r Romania/requirements.txt
```

## Train

Run training (will save best weights to `Romania/model.pt`):

```powershell
python Romania/train_cls.py --epochs 50 --batch 32 --imgsz 224 --device 0
```

Notes:
- Change `--device` to `cpu` if you don't have a GPU.
- You can switch the backbone via `--model yolov8s-cls.pt` or larger if resources allow.

## Quick smoke test

Run a very short training (sanity check):

```powershell
python Romania/train_cls.py --epochs 1 --batch 8 --imgsz 224 --device cpu
```

## Predict on New Images

Use the trained model to classify a single image:

```powershell
python Romania/predict.py path/to/image.jpg --model model.pt --device cpu
```

**Output:** Single lowercase word: `cola`, `fanta`, or `sprite`

The prediction script automatically normalizes class names to lowercase regardless of how the training folders were named (handles Coke/Cola/coke variations).

## Results

- Training artifacts: `Romania/data/coke_fanta_sprite/runs/cls/` (metrics, confusion matrices, etc.).
- Best model copied to: `Romania/model.pt`.
- **Output format:** The model predicts one of three classes: `cola`, `fanta`, `sprite` (lowercase)

## Tips for >90% accuracy

- Use diverse images (angles, lighting, backgrounds, partial occlusions).
- Include cans and bottles; close-ups and context shots.
- Balance classes; aim for 200–500 images per class to start.
- Consider larger backbones (e.g., `yolov8s-cls.pt`) and longer training (100+ epochs) once the pipeline works.
- Mix basic augmentations (random crop/flip/brightness) — enabled by default in Ultralytics classification.
