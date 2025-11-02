import cv2
import numpy as np
import sys
import os

def enhanced_nutrient_denoise(input_path, output_path):
    """
    Enhanced with clearer text contours
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot load image: {input_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"Processing: {input_path}")
    
    # Gentle noise removal (preserves edges)
    denoised = cv2.medianBlur(gray, 3)
    
    #Enhance edges and text contours
    # Use bilateral filter to preserve edges while denoising
    bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # Sharpen the image to enhance text contours
    kernel_sharpen = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(bilateral, -1, kernel_sharpen)
    
    #threshold for clean binarization
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up with minimal morphological operations
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(output_path, cleaned)
    print(f"Enhanced image saved: {output_path}")
    
    return cleaned

def process_all_documents():
    noisy_folder = "noisy"
    output_folder = "enhanced_cleaned"
    
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(noisy_folder):
        print(f"Noisy folder not found: {noisy_folder}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    print(f"Processing folder: {noisy_folder}")
    
    # Process each image
    for filename in os.listdir(noisy_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(noisy_folder, filename)
            output_path = os.path.join(output_folder, f"enhanced_{filename}")
            
            print(f"\nEnhancing: {filename}")
            try:
                enhanced_nutrient_denoise(input_path, output_path)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        enhanced_nutrient_denoise(sys.argv[1], sys.argv[2])
    else:
        process_all_documents()
        print("\nENHANCED DENOISING COMPLETE!")
        print("Use: python enhanced.py <input> <output> for hidden test set")