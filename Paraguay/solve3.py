import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, argrelextrema

# Load and sort data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Data loaded: {len(x)} points")
print(f"x range: [{x.min():.6f}, {x.max():.6f}]")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")

# Create smooth spline with high-degree polynomial fit (for detail capture)
spl = UnivariateSpline(x, y, k=3, s=1000.0)

# Sample at very high resolution
x_fine = np.linspace(x.min(), x.max(), 20000)
y_fine = spl(x_fine)

# Find local maxima and minima
# Peaks (maxima)
peaks = argrelextrema(y_fine, np.greater, order=50)[0]
# Valleys (minima)
valleys = argrelextrema(y_fine, np.less, order=50)[0]

print(f"\nPeaks found: {len(peaks)}")
print(f"Valleys found: {len(valleys)}")

# Combine and sort by position
extrema = sorted(list(zip(peaks, [y_fine[i] for i in peaks], [1]*len(peaks))) + 
                 list(zip(valleys, [y_fine[i] for i in valleys], [0]*len(valleys))))

x_ext = np.array([x_fine[i[0]] for i in extrema])
y_ext = np.array([i[1] for i in extrema])
is_peak = np.array([i[2] for i in extrema])

print(f"Total extrema: {len(extrema)}")

# Heights of peaks
peak_heights = y_ext[is_peak == 1]
valley_depths = y_ext[is_peak == 0]

print(f"\nPeak statistics:")
print(f"  Min: {peak_heights.min():.4f}, Max: {peak_heights.max():.4f}, Mean: {peak_heights.mean():.4f}")
print(f"Valley statistics:")
print(f"  Min: {valley_depths.min():.4f}, Max: {valley_depths.max():.4f}, Mean: {valley_depths.mean():.4f}")

# Cluster peak heights into dot vs dash
from sklearn.cluster import KMeans
if len(peak_heights) > 0:
    peak_heights_2d = peak_heights.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
    kmeans.fit(peak_heights_2d)
    peak_labels = kmeans.labels_
    
    centers_sorted = np.sort(kmeans.cluster_centers_.ravel())
    print(f"\nPeak clustering:")
    print(f"  Dot cluster: {centers_sorted[0]:.4f}")
    print(f"  Dash cluster: {centers_sorted[1]:.4f}")
    
    # Create mapping
    height_mapping = {}
    for i, center in enumerate(centers_sorted):
        height_mapping[center] = i
    
    # Convert peaks to dots and dashes
    symbols = []
    peak_idx = 0
    for i, (x_e, y_e, is_p) in enumerate(extrema):
        if is_p == 1:  # peak
            label = kmeans.predict([[y_e]])[0]
            if label == 0:
                symbols.append(".")
            else:
                symbols.append("-")

print(f"\nExtracted symbols: {''.join(symbols)}")

# Now extract gaps between consecutive peaks to separate letters
peak_x_positions = x_ext[is_peak == 1]
gaps = np.diff(peak_x_positions)

print(f"\nGap statistics (between consecutive peaks):")
print(f"  Min: {gaps.min():.6f}, Max: {gaps.max():.6f}, Mean: {gaps.mean():.6f}")
print(f"  Unique gaps (sorted): {np.sort(np.unique(np.round(gaps, 5)))}")

# Use gap clustering to find symbol vs letter gaps
if len(gaps) > 1:
    gaps_2d = gaps.reshape(-1, 1)
    kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
    kmeans_gaps.fit(gaps_2d)
    gap_labels = kmeans_gaps.labels_
    
    gap_centers = np.sort(kmeans_gaps.cluster_centers_.ravel())
    symbol_gap = gap_centers[0]
    letter_gap = gap_centers[1]
    
    print(f"\nGap clustering:")
    print(f"  Symbol gap: {symbol_gap:.6f}")
    print(f"  Letter gap: {letter_gap:.6f}")
    
    # Decode letters
    letters = []
    current_letter = []
    
    for i, (x_e, y_e, is_p) in enumerate(extrema):
        if is_p == 1:  # peak
            label = kmeans.predict([[y_e]])[0]
            if label == 0:
                current_letter.append(".")
            else:
                current_letter.append("-")
            
            # Check gap to next peak
            peak_count = np.sum(is_peak[:i+1])
            if peak_count - 1 < len(gaps):
                gap = gaps[peak_count - 1]
                if gap > (symbol_gap + letter_gap) / 2:  # Letter boundary
                    if current_letter:
                        letters.append(''.join(current_letter))
                        current_letter = []
    
    if current_letter:
        letters.append(''.join(current_letter))
    
    print(f"\nExtracted letters: {letters}")
    
    # Decode Morse
    MORSE = {
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
        "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
        "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
        "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
        "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
        ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    }
    
    decoded = []
    for letter in letters:
        if letter in MORSE:
            decoded.append(MORSE[letter])
        else:
            decoded.append("?")
    
    result = ''.join(decoded)
    print(f"\nDecoded word: {result}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Data with smooth fit
axes[0].scatter(x, y, alpha=0.3, s=5, label='Data', color='blue')
axes[0].plot(x_fine, y_fine, 'r-', linewidth=1, label='Smooth fit')
axes[0].scatter(x_ext[is_peak == 1], y_ext[is_peak == 1], color='green', s=50, marker='^', label='Peaks')
axes[0].scatter(x_ext[is_peak == 0], y_ext[is_peak == 0], color='orange', s=30, marker='v', label='Valleys')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Signal with detected peaks and valleys')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Peaks only with labels
if len(peak_heights) > 0:
    for i, (xp, yp) in enumerate(zip(peak_x_positions, peak_heights)):
        label = kmeans.predict([[yp]])[0]
        symbol = "." if label == 0 else "-"
        axes[1].scatter(xp, yp, s=100, marker='^')
        axes[1].text(xp, yp + 0.5, symbol, ha='center', fontsize=8)

if len(gaps) > 0:
    for i, gap in enumerate(gaps):
        gap_label = kmeans_gaps.predict([[gap]])[0]
        if gap_label == 1:  # Letter gap
            axes[1].axvline(x=peak_x_positions[i] + gap/2, color='red', alpha=0.3, linestyle='--')

axes[1].set_xlabel('x')
axes[1].set_ylabel('Peak height')
axes[1].set_title('Peaks with symbol labels and letter boundaries')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('extrema_analysis.png', dpi=100)
print("\nPlot saved to extrema_analysis.png")
