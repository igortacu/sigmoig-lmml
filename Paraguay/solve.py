import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# Load and sort data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Data loaded: {len(x)} points")
print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")

# Normalize y
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

# Create smooth spline to fit the signal
# Using high-degree polynomial to capture details
spl = UnivariateSpline(x, y_norm, k=3, s=5.0)
x_smooth = np.linspace(x.min(), x.max(), 5000)
y_smooth = spl(x_smooth)

# Find peaks (dots/dashes are peaks, gaps are valleys)
peaks, peak_props = find_peaks(y_smooth, height=0.3, distance=50, prominence=0.1)
valleys, valley_props = find_peaks(-y_smooth, height=-0.7, distance=50, prominence=0.1)

print(f"\nPeaks found: {len(peaks)}")
print(f"Valleys found: {len(valleys)}")

# Extract peak heights and positions
peak_heights = y_smooth[peaks]
peak_x = x_smooth[peaks]

# Sort by x position
sort_idx = np.argsort(peak_x)
peak_x_sorted = peak_x[sort_idx]
peak_heights_sorted = peak_heights[sort_idx]

print(f"\nPeak positions (x):")
for i, (px, ph) in enumerate(zip(peak_x_sorted, peak_heights_sorted)):
    print(f"  {i}: x={px:.4f}, height={ph:.4f}")

# Calculate gaps between peaks
gaps = np.diff(peak_x_sorted)
print(f"\nGaps between peaks (statistics):")
print(f"  Min: {gaps.min():.4f}, Max: {gaps.max():.4f}, Mean: {gaps.mean():.4f}")
print(f"  Sorted unique gaps (first 10): {np.sort(np.unique(np.round(gaps, 4)))[:10]}")

# Cluster heights into dot/dash
from sklearn.cluster import KMeans
heights_2d = peak_heights_sorted.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
kmeans.fit(heights_2d)
height_labels = kmeans.labels_

# Determine which cluster is dot and which is dash
centers = np.sort(kmeans.cluster_centers_.ravel())
dot_height = centers[0]
dash_height = centers[1]
height_to_label = {centers[i]: i for i in range(2)}

print(f"\nHeight clustering:")
print(f"  Dot height: {dot_height:.4f}")
print(f"  Dash height: {dash_height:.4f}")

# Interpret as Morse code
# Gaps of 1 unit = between symbols
# Gaps of 3+ units = between letters
symbols = []
for i, (px, ph, label) in enumerate(zip(peak_x_sorted, peak_heights_sorted, height_labels)):
    if label == height_to_label[dot_height]:
        symbols.append(".")
    else:
        symbols.append("-")
    
    # Check gap to next peak
    if i < len(gaps):
        gap = gaps[i]
        if gap > 0.05:  # Threshold for letter gap
            symbols.append(" ")

print(f"\nSymbols (raw): {''.join(symbols)}")

# Morse code dictionary
MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
    "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
    "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
    "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    "/.": ".", "/": " "
}

# Decode
text = ''.join(symbols)
letters = text.split()
decoded = []
for letter_code in letters:
    if letter_code in MORSE:
        decoded.append(MORSE[letter_code])
    else:
        decoded.append("?")

result = ''.join(decoded)
print(f"\nDecoded: {result}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot original data and smooth fit
axes[0].scatter(x, y_norm, alpha=0.3, s=5, label='Data')
axes[0].plot(x_smooth, y_smooth, 'r-', linewidth=1, label='Smooth fit')
axes[0].scatter(peak_x_sorted, peak_heights_sorted, color='green', s=50, label='Peaks')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Normalized y')
axes[0].set_title('Signal with detected peaks')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot gap analysis
axes[1].plot(peak_x_sorted[:-1], gaps, 'o-')
axes[1].axhline(y=0.02, color='r', linestyle='--', label='Symbol gap (~0.02)')
axes[1].axhline(y=0.06, color='b', linestyle='--', label='Letter gap (~0.06)')
axes[1].set_xlabel('x position')
axes[1].set_ylabel('Gap to next peak')
axes[1].set_title('Gaps between peaks')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decode_analysis.png', dpi=100)
print("\nPlot saved to decode_analysis.png")
