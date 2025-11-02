# Portugal — CAPTCHA Recognition Challenge

This folder contains a complete pipeline for training and deploying a CAPTCHA recognition system using Convolutional Recurrent Neural Networks (CRNN) with CTC loss.

## Task Overview

Build an OCR system to recognize distorted text in CAPTCHA images. The system must handle:
- Variable-length text sequences (typically 8 characters)
- Character set: lowercase letters (a-z) and digits (0-9)
- Noise, distortion, and line interference
- Varying image dimensions

## Files

### Training & Dataset Generation
- **`generate_dataset.py`** - Generates synthetic CAPTCHA training data with noise, distortion, and lines
- **`train_crnn.py`** - Trains CRNN model with CTC loss for sequence recognition
- **`trainy.py`** - Alternative training script
- **`train_char.py`** - Character-level CNN training (fallback approach)

### Data Processing
- **`build.py`** - Preprocesses training images and builds character-level dataset
- **`build_seed.py`** - Creates seed dataset from labeled examples
- **`seed_labels.py`** - Generates initial labels for bootstrapping
- **`label_rest.py`** - Labels remaining unlabeled images

### Inference Scripts
- **`infer_crnn.py`** - Main CRNN inference for test set
- **`infer.py`** - Character-level CNN inference
- **`infer_tesseract.py`** - Tesseract OCR fallback
- **`infer_easyocr.py`** - EasyOCR fallback
- **`infer_paddleocr.py`** - PaddleOCR fallback

### Utilities
- **`debug.py`** - Debugging and visualization tools
- **`run_pipeline.sh`** - Complete automated pipeline (Linux/Mac)
- **`submission.csv`** - Output predictions

## Quick Start

### Complete Pipeline (Bash - Linux/Mac)

```bash
bash run_pipeline.sh
```

This runs all three steps automatically:
1. Generate synthetic training data
2. Train CRNN model
3. Create submission file

### Step-by-Step (Windows PowerShell)

#### Step 1: Generate Training Data
```powershell
python generate_dataset.py
```
Creates 5,000 synthetic CAPTCHA images in `data/synthetic_train/`

#### Step 2: Train Model
```powershell
python train_crnn.py
```
Trains CRNN model and saves checkpoint to `crnn_model.pt`

#### Step 3: Run Inference
```powershell
python infer_crnn.py
```
Generates `submission.csv` with predictions

### Alternative Approaches

**Character-level approach:**
```powershell
python build.py           # Build character dataset
python train_char.py      # Train character CNN
python infer.py           # Run inference
```

**Using pre-trained OCR:**
```powershell
python infer_tesseract.py   # Requires Tesseract installed
python infer_easyocr.py     # Requires easyocr package
python infer_paddleocr.py   # Requires paddleocr package
```

## Model Architecture

### CRNN (Main Approach)
```
Input (H×W×1) 
  ↓ CNN Feature Extractor
  ↓ Sequential Features
  ↓ Bidirectional LSTM
  ↓ CTC Loss
Output (Sequence)
```

**Key components:**
- **CNN backbone:** Extracts visual features from image
- **RNN layers:** Captures sequential dependencies
- **CTC loss:** Handles variable-length sequences without alignment

### Configuration (train_crnn.py)
```python
IMG_HEIGHT = 64
IMG_WIDTH = 270
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
```

## Dataset Structure

```
data/
├── task_32/
│   ├── train/              # Original training images
│   ├── test/               # Test images
│   └── train_labels.csv    # Training labels
├── synthetic_train/        # Generated synthetic data
│   ├── *.png
│   └── labels.csv
├── char_ds/                # Character-level dataset
└── seed_chars/             # Seed characters for bootstrapping
```

## Requirements

```
torch>=1.9.0
torchvision
Pillow
numpy
opencv-python (optional, for some preprocessing)
```

### Additional for inference alternatives:
```
pytesseract              # For Tesseract OCR
easyocr                  # For EasyOCR
paddleocr paddlepaddle   # For PaddleOCR
```

Install with:
```powershell
pip install torch torchvision Pillow numpy
```

## Synthetic Data Generation

The `generate_dataset.py` script creates realistic CAPTCHA images with:
- **Random text:** 8-character sequences from charset
- **Noise:** Gaussian noise with random intensity
- **Lines:** Random diagonal/curved lines
- **Distortion:** Character spacing, rotation, warping
- **Brightness variation:** Random brightness adjustments

Adjust parameters in the script:
```python
NUM_SAMPLES = 5000        # Number of images to generate
IMG_WIDTH = 270
IMG_HEIGHT = 80
CAPTCHA_LENGTH = 8
```

## Training Process

1. **Load synthetic dataset** from `data/synthetic_train/`
2. **Preprocess images:** Grayscale, resize to 64×270
3. **Encode labels:** Convert characters to indices
4. **Train with CTC loss:** Handles variable-length sequences
5. **Save checkpoint:** Model saved to `crnn_model.pt`

**Monitor training:**
- Loss should decrease steadily
- Character accuracy reported per epoch
- Early stopping if validation loss plateaus

## Inference

The inference script:
1. Loads trained model from checkpoint
2. Processes test images (resize, normalize)
3. Runs forward pass through CRNN
4. Decodes CTC output to text
5. Saves predictions to `submission.csv`

**Output format:**
```csv
filename,answer
test_0001.png,abc123de
test_0002.png,xyz789gh
...
```

## Troubleshooting

**Low accuracy on real CAPTCHAs:**
- Increase synthetic dataset size
- Add more augmentation variations
- Fine-tune on real labeled examples if available

**Training too slow:**
- Reduce `BATCH_SIZE` if GPU memory is limited
- Use fewer `EPOCHS` for quick experiments
- Enable GPU acceleration (CUDA)

**CTC decoding errors:**
- Increase model capacity (more channels/layers)
- Adjust learning rate
- Ensure character set matches exactly

## Pipeline Workflow

```
[Generate] → [Train] → [Infer]
    ↓           ↓         ↓
synthetic   model.pt  submission.csv
  data
```

1. **Generate:** Creates training data mimicking target CAPTCHAs
2. **Train:** Learns sequence-to-sequence mapping with CTC
3. **Infer:** Predicts text for all test images

## Advanced: Bootstrapping from Real Data

If you have some labeled real CAPTCHA images:

```powershell
python seed_labels.py      # Create initial seed labels
python build_seed.py       # Build seed dataset
python label_rest.py       # Semi-supervised labeling
python train_crnn.py       # Train on real + synthetic
```

This approach combines synthetic data with real examples for better domain adaptation.

## Notes

- CRNN with CTC is the standard approach for sequence OCR tasks
- Synthetic data generation is key when labeled data is scarce
- Character-level approach (`build.py` + `train_char.py`) works for fixed-length CAPTCHAs
- Pre-trained OCR tools (Tesseract, EasyOCR) can be used as fallbacks but may require fine-tuning
- Increase synthetic dataset size and variety for better generalization
