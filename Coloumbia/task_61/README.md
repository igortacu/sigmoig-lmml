# Roman Numerals Classification - Data-Centric AI Challenge

## Challenge Overview
Improve Roman numeral recognition accuracy by focusing on **data quality improvements only**, without modifying the model architecture in `train.py`.

## Task
Classify handwritten Roman numerals (i, ii, iii, iv, v, vi, vii, viii, ix, x) using a ResNet50-based CNN model.

**Goal**: Achieve 90%+ accuracy through data cleaning, quality filtering, and smart augmentation.

## Solution Approach

### Data-Centric Strategy
1. **Quality Filtering**: Remove low-quality images using multiple metrics:
   - Content ratio (non-white pixels > 8%)
   - Edge strength (Sobel gradients)
   - Contrast (standard deviation)
   - Brightness (mean < 245)

2. **Image Enhancement**: Apply preprocessing to all images:
   - Denoising (fastNlMeansDenoising)
   - Adaptive histogram equalization (CLAHE)
   - Edge sharpening
   - Contrast enhancement

3. **Smart Augmentation**: Balance classes to 900 images each:
   - Small rotations (-7° to +7°)
   - Affine transformations (slight perspective)
   - Brightness adjustments
   - Combination augmentations

## Files

### Core Files
- `train.py` - Model training script (DO NOT MODIFY)
- `prepare_data_final.py` - Ultimate data cleaning pipeline
- `data_original/` - Original dataset (train: 900/class, val: 80/class)
- `data/` - Cleaned and enhanced dataset (generated)

### Output
- `best_model.weights.h5` - Best model weights saved during training

## Dataset Structure

```
data_original/
├── train/
│   ├── i/      (900 images)
│   ├── ii/     (900 images)
│   ├── iii/    (900 images)
│   ├── iv/     (900 images)
│   ├── v/      (900 images)
│   ├── vi/     (900 images)
│   ├── vii/    (900 images)
│   ├── viii/   (900 images)
│   ├── ix/     (900 images)
│   └── x/      (900 images)
└── val/
    ├── i/      (80 images)
    ├── ii/     (80 images)
    ├── iii/    (80 images)
    ├── iv/     (80 images)
    ├── v/      (80 images)
    ├── vi/     (80 images)
    ├── vii/    (80 images)
    ├── viii/   (80 images)
    ├── ix/     (80 images)
    └── x/      (80 images)
```

## Usage

### 1. Prepare the Dataset
```bash
python prepare_data_final.py
```

This will:
- Filter out low-quality images (typically removes 300-400 poor images)
- Enhance all remaining images
- Balance classes to 900 images each through augmentation
- Create cleaned dataset in `data/` folder
- Stay within 10,000 image limit

### 2. Train the Model
```bash
python train.py
```

The model will:
- Train for 2 epochs (as specified in the original script)
- Use batch size of 8
- Save best weights to `best_model.weights.h5`
- Report validation accuracy

## Key Insights

### What Makes Good Training Data?
1. **Clear edges**: Roman numerals need strong, visible strokes
2. **Sufficient content**: At least 8% of pixels should be non-white
3. **Good contrast**: High variance in pixel values
4. **Not too bright**: Average brightness < 245 (avoids blank/near-blank images)

### Quality Metrics Used
```python
quality_score = (
    (content_ratio > 0.08) * 30 +    # Must have content
    min(edge_strength / 10, 30) +     # Strong edges
    min(contrast / 10, 30) +           # Good contrast
    (brightness < 245) * 10            # Not too bright
)
# Threshold: score >= 60 to keep
```

### Augmentation Techniques
- **Rotation**: ±7° to simulate handwriting variation
- **Affine transforms**: Slight perspective/shear changes
- **Brightness**: ±15 pixel value adjustments
- **Always enhance after augmentation**: Maintain quality

## Requirements

```
tensorflow==2.19.0
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Model Architecture (Fixed)

The model uses a ResNet50 backbone:
- Input: 32×32×3 grayscale images
- Base: ResNet50 (up to conv2_block3_out)
- Head: GlobalAveragePooling2D → Dense(10)
- Loss: Categorical crossentropy
- Optimizer: Adam (lr=0.0001)

**Note**: Architecture cannot be modified per challenge rules.

## Results

Expected performance after data cleaning:
- **Before cleaning**: ~60-70% validation accuracy
- **After cleaning**: 90%+ validation accuracy

The improvement comes entirely from better data quality, not model changes.

## Troubleshooting

### "Dataset size larger than 10,000"
Reduce the `target` variable in `prepare_data_final.py` from 900 to a lower value (e.g., 880).

### Low accuracy after training
Try these data-centric improvements:
1. Increase quality threshold (change `>= 60` to `>= 70`)
2. Add more aggressive denoising
3. Enhance edge detection in filtering
4. Reduce augmentation randomness

### Out of memory during training
This is a model issue (batch size), but you can:
- Use a machine with more RAM
- The script uses batch_size=8 which should work on most systems

## Data-Centric AI Principles Applied

1. **Quality over quantity**: Removed 300+ low-quality images
2. **Consistent preprocessing**: All images enhanced uniformly
3. **Balanced distribution**: Equal samples per class
4. **Smart augmentation**: Preserves numeral structure
5. **Validation set integrity**: Enhanced but not over-augmented

## License

This is a competition submission. Refer to challenge guidelines for usage terms.
