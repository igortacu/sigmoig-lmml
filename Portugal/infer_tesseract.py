"""
CAPTCHA inference using Tesseract OCR and pytesseract
"""
import os
import csv
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

TEST_DIR = "data/task_32/images"
OUTPUT_CSV = "submission.csv"

def preprocess_for_ocr(img_path):
    """Preprocess image for better OCR"""
    img = Image.open(img_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    
    # Apply threshold
    threshold = 140
    img = img.point(lambda x: 0 if x < threshold else 255, '1')
    img = img.convert('L')
    
    return img

def clean_text(text):
    """Clean OCR output"""
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

# Configure pytesseract for single line, alphanumeric
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz'

for i, filename in enumerate(test_files):
    img_path = os.path.join(TEST_DIR, filename)
    
    # Try multiple preprocessing approaches
    texts = []
    
    # Approach 1: Preprocessed image
    img = preprocess_for_ocr(img_path)
    text1 = pytesseract.image_to_string(img, config=custom_config)
    texts.append(clean_text(text1))
    
    # Approach 2: Original image
    img_original = Image.open(img_path)
    text2 = pytesseract.image_to_string(img_original, config=custom_config)
    texts.append(clean_text(text2))
    
    # Approach 3: Inverted image
    img_inv = Image.open(img_path).convert('L')
    img_inv = Image.eval(img_inv, lambda x: 255 - x)
    text3 = pytesseract.image_to_string(img_inv, config=custom_config)
    texts.append(clean_text(text3))
    
    # Choose the best result (closest to 8 characters)
    text = min(texts, key=lambda t: abs(len(t) - 8))
    
    # Ensure exactly 8 characters
    if len(text) > 8:
        text = text[:8]
    elif len(text) < 8:
        # Try to find a better match
        for t in texts:
            if len(t) == 8:
                text = t
                break
        else:
            # Pad if needed
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
