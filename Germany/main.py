import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
import requests 

GERMANY_PATH = os.path.dirname(os.path.abspath(__file__))

def clean_qr(binary_img):
    # Remove tiny noise dots
    cleaned = cv2.medianBlur(binary_img, 5)

    # Close small holes and reconnect broken squares
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sharpen blocks to a crisp grid
    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(closed, -1, sharpen_kernel)

    return sharpened

def restore_qr_code(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strong adaptive threshold (works better on damaged images)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 3
    )

    cleaned = clean_qr(adaptive)
    return cleaned

def decode_qr_code(image):
    decoded_objects = decode(image)
    if decoded_objects:
        for obj in decoded_objects:
            if obj.type == 'QRCODE':
                return obj.data.decode('utf-8')
    return None

def decode_with_zxing(image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    response = requests.post(
        "https://zxing.org/w/decode",
        files={"f": ("qr.png", img_data, "image/png")}
    )
    text = response.text
    if "Parsed Result" in text:
        start = text.find("<pre>") + 5
        end = text.find("</pre>")
        return text[start:end].strip()
    return None

def main():
    if os.path.exists(GERMANY_PATH):
        os.chdir(GERMANY_PATH)
    else:
        print(f"Path not found: {GERMANY_PATH}")
        return

    input_file = 'distorted_qr.png'
    if not os.path.exists(input_file):
        print(f"{input_file} not found!")
        return

    # Restore QR
    restored_qr = restore_qr_code(input_file)
    cv2.imwrite('restored_qr.png', restored_qr)
    print("Restored QR code saved as restored_qr.png")

    # Try decoding
    flag = decode_qr_code(restored_qr)
    if flag:
        print(f"SUCCESS! Decoded locally: {flag}")
        return

    print("Local decode failed, trying ZXing online...")
    flag = decode_with_zxing('restored_qr.png')
    if flag:
        print(f"✅ SUCCESS via ZXing API: {flag}")
        return

    # Try enlarging and rescanning
    enlarged = cv2.resize(restored_qr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('restored_large.png', enlarged)
    flag = decode_qr_code(enlarged)
    if flag:
        print(f"SUCCESS with enlarged version: {flag}")
        return

    print("All methods failed. The QR may be too damaged, or needs manual patching.")

if __name__ == "__main__":
    main()