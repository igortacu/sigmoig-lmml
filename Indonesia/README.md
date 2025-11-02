# Indonesia — Adversarial Perturbation Attack

This folder contains scripts to generate an adversarial perturbation for fooling an image classifier into predicting "giant panda" on any input image.

## Task Overview

Create a `perturbation.npy` file of shape `(224, 224, 3)` with dtype `float32` that, when added to any image, causes a classifier to misclassify it as a panda. The perturbation is constrained to a maximum L∞ epsilon of 0.2.

## Files

- **`make.py`** - Main script using MobileNetV2 as surrogate model to generate targeted adversarial perturbation
- **`generate_perturbation.py`** - Alternative flexible script with white-box PGD attack or fallback mode
- **`input.jpg`** - Source image for perturbation generation
- **`perturbation.npy`** - Output adversarial perturbation (generated)

## Quick Start

### Method 1: Using make.py (Recommended)

This uses MobileNetV2 from TensorFlow/Keras as a surrogate model:

```powershell
python make.py
```

**Parameters in the script:**
- `IMG_SIZE = 224` - Image dimensions
- `MAX_EPS = 0.2` - Maximum L∞ perturbation bound
- `STEP = 0.01` - PGD step size
- `ITERS = 120` - Number of optimization iterations
- `PANDA_INDEX = 388` - ImageNet class index for giant panda

### Method 2: Using generate_perturbation.py

**White-box mode** (with model and image):
```powershell
python generate_perturbation.py --image input.jpg --model pd_model.h5 --target-index 388 --eps 0.2 --steps 100 --step-size 0.01 --out perturbation.npy
```

**Fallback mode** (no model needed):
```powershell
python generate_perturbation.py --out perturbation.npy
```

## How It Works

1. **Load surrogate model** - Uses a pre-trained ImageNet classifier (MobileNetV2)
2. **Initialize perturbation** - Start with zero perturbation
3. **Targeted PGD attack** - Iteratively update perturbation to maximize probability of target class (panda)
4. **Project to constraints** - Clip perturbation to [-eps, eps] and ensure final image is in [0, 1]
5. **Save output** - Export as `perturbation.npy`

## Output Format

- **File:** `perturbation.npy`
- **Shape:** `(224, 224, 3)`
- **Dtype:** `float32`
- **Value range:** `[-0.2, 0.2]` (or as specified by `--eps`)

## Requirements

```
tensorflow>=2.0
numpy
opencv-python (for generate_perturbation.py)
```

Install with:
```powershell
pip install tensorflow numpy opencv-python
```

## Notes

- The perturbation is added element-wise to evaluation images: `adversarial_image = original_image + perturbation`
- Final images are clipped to `[0, 1]` during evaluation
- For best results, use a surrogate model similar to the evaluation model
- ImageNet class 388 = "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca"
- Higher iteration counts generally improve attack success rate but take longer
- Adjust `MAX_EPS` to balance imperceptibility vs effectiveness
