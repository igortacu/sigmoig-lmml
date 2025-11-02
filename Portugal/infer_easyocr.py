"""
CAPTCHA inference using EasyOCR pre-trained model
"""
import os
import csv
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

TEST_DIR = "data/task_32/images"
OUTPUT_CSV = "submission.csv"

print("Initializing EasyOCR reader...")
# Initialize EasyOCR reader (English only)
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(img_path):
    """Preprocess image for better OCR"""
    img = Image.open(img_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Apply threshold to make text clearer
    threshold = 140
    img = img.point(lambda x: 0 if x < threshold else 255, '1')
    img = img.convert('L')
    
    # Slight blur to remove noise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img

def clean_text(text):
    """Clean and normalize OCR output"""
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
    
    # Preprocess image
    img = preprocess_image(img_path)
    
    # Convert to numpy array for EasyOCR
    img_array = np.array(img)
    
    # Run OCR
    result = reader.readtext(img_array, detail=0, paragraph=False)
    
    # Combine all detected text
    text = ''.join(result)
    text = clean_text(text)
    
    # Ensure we have 8 characters (pad or truncate)
    if len(text) < 8:
        # If less than 8, try with original image
        img_original = Image.open(img_path)
        result_original = reader.readtext(np.array(img_original), detail=0, paragraph=False)
        text_original = clean_text(''.join(result_original))
        if len(text_original) > len(text):
            text = text_original
    
    # Ensure exactly 8 characters
    if len(text) > 8:
        text = text[:8]
    elif len(text) < 8:
        # Pad with most common character or try different preprocessing
        text = text.ljust(8, 'x')
    
    results.append((filename, text))
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_files)} images")

# Save results
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'answer'])
    writer.writerows(results)

print(f"\nDone! Results saved to {OUTPUT_CSV}")
print("\nFirst 10 predictions:")
for filename, text in results[:10]:
    print(f"  {filename}: {text}")
