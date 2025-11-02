import cv2
import numpy as np
from PIL import Image

# === LEGEND OF RESTORATION ===
# Step-by-step reversal of each “Age” in the Chronicle of the Shattered Realm

# Load the cursed image
img = cv2.imread("initial_to_be_given.jpg")

# 1️⃣ Undo “Pact of Crimson and Azure” — Swap red and blue
img = img[:, :, [2, 1, 0]]

# 2️⃣ Undo “Mirror of Night” — Invert all colors
img = cv2.bitwise_not(img)

# 3️⃣ Undo “Eastward Turn” — Horizontal flip
img = cv2.flip(img, 1)

# 4️⃣ Undo “Southward Turn” — Vertical flip
img = cv2.flip(img, 0)

# 5️⃣ Undo “Skewwright’s Sigil” — Reverse the affine skew
rows, cols, ch = img.shape
M = np.float32([[1, -0.2, 0], [0, 1, 0]])  # inverse of (1, 0.2)
img = cv2.warpAffine(img, M, (cols, rows))

# 6️⃣ Undo “Storm of Fifty Winds” — Normalize contrast & brightness
img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# 7️⃣ Undo “Seed of 42” — Median blur correction
img = cv2.medianBlur(img, 3)

# Save the restored base
cv2.imwrite("1_restored_realm.jpg", img)

# === SECOND PHASE: The Revealing Ritual ===
# Convert to grayscale for easier text detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("2_gray_realm.jpg", gray)

# Edge detection — “You are looking but can’t see”
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("3_edges_revealed.jpg", edges)

# Invert & enhance edges
inverted = cv2.bitwise_not(edges)
blurred = cv2.GaussianBlur(inverted, (3,3), 0)
enhanced = cv2.convertScaleAbs(blurred, alpha=2.5, beta=40)
cv2.imwrite("4_enhanced_hidden_text.jpg", enhanced)

# Apply XOR with the sacred Seed of 42
xor42 = cv2.bitwise_xor(gray, 42)
cv2.imwrite("5_xor42_reveal.jpg", xor42)

# Combine all visual layers for maximum contrast discovery
combined = cv2.addWeighted(enhanced, 0.7, xor42, 0.3, 0)
cv2.imwrite("6_final_reveal.jpg", combined)

print("✅ Restoration complete!")
print("Check the generated files (1_restored_realm.jpg … 6_final_reveal.jpg)")
print("The flag or hidden text should appear in one of them.")
