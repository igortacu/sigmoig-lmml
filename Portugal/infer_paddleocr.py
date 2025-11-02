"""
CAPTCHA inference using PaddleOCR - a powerful pre-trained OCR model
"""
import os
import csv
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
import numpy as np

TEST_DIR = "data/task_32/images"
OUTPUT_CSV = "submission.csv"

print("Initializing PaddleOCR...")
# Initialize PaddleOCR (English, use_angle_cls=True for rotated text detection)
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

def preprocess_image(img_path):
    """Preprocess image for better OCR"""
    img = Image.open(img_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    return img

def clean_text(text):
    """Clean and normalize OCR output"""
    if not text:
        return ""
    # Remove spaces and special characters
    text = text.replace(' ', '').replace('\n', '').replace('\t', '')
    # Convert to lowercase
    text = text.lower()
    # Keep only alphanumeric
    text = ''.join(c for c in text if c.isalnum())
    return text

print(f"Processing test images from {TEST_DIR}...")

results = []
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.png')])

for i, filename in enumerate(test_files):
    img_path = os.path.join(TEST_DIR, filename)
    
    # Try with original image first
    result = ocr.ocr(img_path, cls=False)
    
    texts = []
    if result and result[0]:
        for line in result[0]:
            if len(line) >= 2:
                texts.append(line[1][0])  # Get the text
    
    # Combine all detected text
    text = clean_text(''.join(texts))
    
    # If not good enough, try with preprocessing
    if len(text) != 8:
        img = preprocess_image(img_path)
        # Save to temp file
        temp_path = "temp_processed.png"
        img.save(temp_path)
        result = ocr.ocr(temp_path, cls=False)
        
        texts = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    texts.append(line[1][0])
        
        text2 = clean_text(''.join(texts))
        
        # Use whichever is closer to 8 characters
        if abs(len(text2) - 8) < abs(len(text) - 8):
            text = text2
    
    # Ensure exactly 8 characters
    if len(text) > 8:
        text = text[:8]
    elif len(text) < 8:
        # Pad with most common missing chars or 'x'
        text = text.ljust(8, 'x')
    
    results.append((filename, text))
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_files)} images")

# Clean up temp file
if os.path.exists("temp_processed.png"):
    os.remove("temp_processed.png")

# Save results
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'answer'])
    writer.writerows(results)

print(f"\nDone! Results saved to {OUTPUT_CSV}")
print("\nFirst 10 predictions:")
for filename, text in results[:10]:
    print(f"  {filename}: {text}")
print("\nLast 5 predictions:")
for filename, text in results[-5:]:
    print(f"  {filename}: {text}")
