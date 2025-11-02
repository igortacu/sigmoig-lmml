#!/bin/bash
# Complete pipeline to generate dataset, train model, and create submission

echo "=== CAPTCHA Solver Pipeline ==="
echo ""

# Step 1: Generate synthetic training data
echo "[Step 1/3] Generating synthetic CAPTCHA dataset..."
python3 generate_dataset.py
if [ $? -ne 0 ]; then
    echo "Error generating dataset!"
    exit 1
fi
echo ""

# Step 2: Train the CRNN model
echo "[Step 2/3] Training CRNN model..."
python3 train_crnn.py
if [ $? -ne 0 ]; then
    echo "Error training model!"
    exit 1
fi
echo ""

# Step 3: Run inference on test set
echo "[Step 3/3] Running inference on test set..."
python3 infer_crnn.py
if [ $? -ne 0 ]; then
    echo "Error during inference!"
    exit 1
fi
echo ""

echo "=== Pipeline Complete ==="
echo "Check submission.csv for results"
