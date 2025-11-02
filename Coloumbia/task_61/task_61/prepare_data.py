#!/usr/bin/env python3
"""Prepare optimal 10k dataset from parent's ultra-high-quality data"""
import shutil
from pathlib import Path
import random

# Clean slate
shutil.rmtree('data_original', ignore_errors=True)
Path('data_original/train').mkdir(parents=True, exist_ok=True)
Path('data_original/val').mkdir(parents=True, exist_ok=True)

# Copy ultra-high-quality data from parent, selecting best samples to fit 10k limit
parent_data = Path('../data_original')
classes = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

# Target: 9,400 train + 600 val = 10,000 total
# Per class: 940 train + 60 val
train_per_class = 940
val_per_class = 60

random.seed(42)  # For reproducibility

print("Copying ultra-high-quality data...")
for cls in classes:
    # Training data
    src_train = parent_data / 'train' / cls
    dst_train = Path('data_original/train') / cls
    dst_train.mkdir(exist_ok=True)
    
    if src_train.exists():
        files = list(src_train.glob('*.png')) + list(src_train.glob('*.jpg'))
        # Randomly select to ensure diversity
        selected = random.sample(files, min(train_per_class, len(files)))
        for f in selected:
            shutil.copy2(f, dst_train / f.name)
    
    # Validation data
    src_val = parent_data / 'val' / cls
    dst_val = Path('data_original/val') / cls
    dst_val.mkdir(exist_ok=True)
    
    if src_val.exists():
        files = list(src_val.glob('*.png')) + list(src_val.glob('*.jpg'))
        selected = random.sample(files, min(val_per_class, len(files)))
        for f in selected:
            shutil.copy2(f, dst_val / f.name)
    
    train_count = len(list(dst_train.glob('*')))
    val_count = len(list(dst_val.glob('*')))
    print(f'{cls}: {train_count} train, {val_count} val')

# Calculate total
train_total = sum(len(list((Path('data_original/train') / c).glob('*'))) for c in classes)
val_total = sum(len(list((Path('data_original/val') / c).glob('*'))) for c in classes)
print(f'\n✅ Total: {train_total} train + {val_total} val = {train_total + val_total}')
print(f'✅ Ready for training with ultra-high-quality data (original quality: 0.676-0.849)')
