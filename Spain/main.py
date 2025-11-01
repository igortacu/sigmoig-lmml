import cv2
import numpy as np
import pytesseract
import random
import os

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BLOCK = 16
SEED = 42

def unshuffle_blocks(img, block=BLOCK, seed=SEED):
    h, w = img.shape[:2]
    bh, bw = h // block, w // block

    blocks = []
    for y in range(bh):
        for x in range(bw):
            blocks.append(img[y*block:(y+1)*block, x*block:(x+1)*block])

    random.seed(seed)
    order = list(range(len(blocks)))
    random.shuffle(order)

    restored = [None]*len(blocks)
    for i, shuffled_i in enumerate(order):
        restored[i] = blocks[shuffled_i]   # FIX HERE

    out = np.zeros_like(img)
    idx = 0
    for y in range(bh):
        for x in range(bw):
            out[y*block:(y+1)*block, x*block:(x+1)*block] = restored[idx]
            idx += 1

    return out

def restore_realm(image_path):
    img = cv2.imread(image_path)
    print("[+] Image loaded!")

    # Reverse Skew
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0.2, 0], [0, 1, 0]])
    Minv = cv2.invertAffineTransform(M)
    img = cv2.warpAffine(img, Minv, (cols, rows))

    # Swap R/B
    img = img[:, :, [2,1,0]]

    # Flip Vertical
    img = cv2.flip(img, 0)

    # Flip Horizontal
    img = cv2.flip(img, 1)

    # Invert
    img = 255 - img

    # Light Denoise
    img = cv2.medianBlur(img, 3)

    # **NOW** Unshuffle
    img = unshuffle_blocks(img)

    cv2.imwrite("restored.png", img)
    print("[+] Saved corrected restored.png")
    return img

def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,11,2)

    cv2.imwrite("final_text_ready.png", gray)

    text = pytesseract.image_to_string(gray, config='--psm 6')
    print("\n==== OCR RESULT ====\n")
    print(text)
    print("\n====================\n")
    return text

if __name__ == "__main__":
    for f in os.listdir('.'):
        if f.lower().endswith(('.png', '.jpg')):
            print(f"[+] Using image: {f}")
            img = restore_realm(f)
            extract_text(img)
            break
