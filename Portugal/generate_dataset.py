"""
Generate synthetic CAPTCHA dataset for training
"""
import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUTPUT_DIR = "data/synthetic_train"
NUM_SAMPLES = 5000
IMG_WIDTH = 270
IMG_HEIGHT = 80
CAPTCHA_LENGTH = 8

# Character set (lowercase letters and digits)
CHARSET = string.ascii_lowercase + string.digits

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_noise(img):
    """Add various types of noise to the image"""
    img_array = np.array(img, dtype=np.float32)
    
    # Add Gaussian noise
    noise = np.random.normal(0, random.randint(5, 15), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255)
    
    # Random brightness adjustment
    brightness = random.uniform(0.7, 1.3)
    img_array = np.clip(img_array * brightness, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))


def add_lines(draw, width, height):
    """Add random lines to image"""
    num_lines = random.randint(2, 5)
    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = random.randint(100, 200)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 2))


def generate_captcha(text):
    """Generate a CAPTCHA image with the given text"""
    # Create white background
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font, fallback to default
    try:
        # Try different font sizes
        font_size = random.randint(35, 45)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw each character with random positioning and rotation
    char_width = IMG_WIDTH // len(text)
    
    for i, char in enumerate(text):
        # Random position within character slot
        x_offset = random.randint(-3, 3)
        y_offset = random.randint(-5, 5)
        x = i * char_width + char_width // 4 + x_offset
        y = IMG_HEIGHT // 3 + y_offset
        
        # Random color (dark)
        color = random.randint(0, 100)
        
        # Create a temporary image for the character
        char_img = Image.new('L', (char_width + 20, IMG_HEIGHT), color=255)
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((10, IMG_HEIGHT // 3), char, font=font, fill=0)
        
        # Random rotation
        angle = random.randint(-15, 15)
        char_img = char_img.rotate(angle, expand=False, fillcolor=255)
        
        # Paste the character
        img.paste(char_img, (i * char_width, 0), char_img)
    
    # Redraw with actual colors
    img = img.convert('L')
    draw = ImageDraw.Draw(img)
    
    for i, char in enumerate(text):
        x_offset = random.randint(-3, 3)
        y_offset = random.randint(-5, 5)
        x = i * char_width + char_width // 4 + x_offset
        y = IMG_HEIGHT // 3 + y_offset
        color = random.randint(0, 100)
        draw.text((x, y), char, font=font, fill=color)
    
    # Add noise lines
    add_lines(draw, IMG_WIDTH, IMG_HEIGHT)
    
    # Convert back to RGB
    img = img.convert('RGB')
    
    # Add noise
    img = add_noise(img)
    
    # Apply blur
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    return img


def generate_text():
    """Generate random CAPTCHA text"""
    return ''.join(random.choices(CHARSET, k=CAPTCHA_LENGTH))


def main():
    print(f"Generating {NUM_SAMPLES} synthetic CAPTCHA images...")
    
    labels = []
    for i in range(NUM_SAMPLES):
        text = generate_text()
        img = generate_captcha(text)
        
        filename = f"synth_{i:05d}.png"
        img.save(os.path.join(OUTPUT_DIR, filename))
        labels.append((filename, text))
        
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} images")
    
    # Save labels
    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w") as f:
        f.write("filename,answer\n")
        for filename, text in labels:
            f.write(f"{filename},{text}\n")
    
    print(f"Done! Generated {NUM_SAMPLES} images in {OUTPUT_DIR}")
    print(f"Labels saved to {OUTPUT_DIR}/labels.csv")


if __name__ == "__main__":
    main()
