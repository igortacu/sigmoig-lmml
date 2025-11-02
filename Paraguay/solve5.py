import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Data loaded: {len(x)} points")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# Fit a smooth spline (with minimal smoothing to capture oscill ations)
spl = UnivariateSpline(x, y, k=3, s=50.0)

# Sample at high resolution
x_fine = np.linspace(x.min(), x.max(), 10000)
y_fine = spl(x_fine)

# Find peaks (local maxima)
peaks_idx, peaks_props = find_peaks(y_fine, distance=100, prominence=1.0)
# Find valleys (local minima)
valleys_idx, valleys_props = find_peaks(-y_fine, distance=100, prominence=1.0)

peaks_x = x_fine[peaks_idx]
peaks_y = y_fine[peaks_idx]
valleys_x = x_fine[valleys_idx]
valleys_y = y_fine[valleys_idx]

print(f"\nPeaks: {len(peaks_x)}")
print(f"Valleys: {len(valleys_x)}")

if len(peaks_x) > 0:
    print(f"Peak y range: [{peaks_y.min():.2f}, {peaks_y.max():.2f}]")
    print(f"Valley y range: [{valleys_y.min():.2f}, {valleys_y.max():.2f}]")
    
    # Cluster peak heights into dot vs dash
    from sklearn.cluster import KMeans
    peaks_y_2d = peaks_y.reshape(-1, 1)
    
    try:
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
        kmeans.fit(peaks_y_2d)
        
        centers = np.sort(kmeans.cluster_centers_.ravel())
        print(f"\nHeight clusters: {centers}")
        
        # Map peaks to dots/dashes
        labels = kmeans.predict(peaks_y_2d)
        morse = []
        for label in labels:
            if label == 0:
                morse.append(".")
            else:
                morse.append("-")
        
        morse_str = ''.join(morse)
        print(f"Morse symbols ({len(morse)}): {morse_str}")
        
        # Analyze gaps between peaks
        gaps = np.diff(peaks_x)
        print(f"\nGap analysis:")
        print(f"  Min: {gaps.min():.6f}, Max: {gaps.max():.6f}, Mean: {gaps.mean():.6f}")
        print(f"  Median: {np.median(gaps):.6f}")
        
        # Use KMeans clustering on gaps to find letter vs word boundaries
        gaps_2d = gaps.reshape(-1, 1)
        kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
        kmeans_gaps.fit(gaps_2d)
        gap_labels = kmeans_gaps.predict(gaps_2d)
        
        gap_centers = np.sort(kmeans_gaps.cluster_centers_.ravel())
        symbol_gap = gap_centers[0]
        letter_gap = gap_centers[1]
        print(f"  Symbol gap: {symbol_gap:.6f}")
        print(f"  Letter gap: {letter_gap:.6f}")
        
        morse_letters = []
        current_letter = []
        
        for i in range(len(morse)):
            current_letter.append(morse[i])
            
            # Check gap to next peak
            if i < len(gaps):
                if gap_labels[i] == 1:  # Letter gap (larger)
                    morse_letters.append(''.join(current_letter))
                    morse_letters.append('/')
                    current_letter = []
            else:
                morse_letters.append(''.join(current_letter))
        
        print(f"\nMorse letters: {morse_letters}")
        
        # Decode
        MORSE = {
            ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
            "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
            "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
            "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
            "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
            ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
        }
        
        decoded = []
        for code in morse_letters:
            if code == '/':
                decoded.append(' ')
            elif code in MORSE:
                decoded.append(MORSE[code])
            else:
                decoded.append('?')
        
        result = ''.join(decoded)
        print(f"\nDecoded: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

axes[0].scatter(x, y, alpha=0.3, s=3, label='Data', color='blue')
axes[0].plot(x_fine, y_fine, 'r-', linewidth=1, label='Spline fit')
if len(peaks_x) > 0:
    axes[0].scatter(peaks_x, peaks_y, s=50, color='green', marker='^', label='Peaks', zorder=5)
    axes[0].scatter(valleys_x, valleys_y, s=30, color='orange', marker='v', label='Valleys', zorder=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Spline fit with extrema')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Peaks only
if len(peaks_x) > 0:
    for px, py, lab in zip(peaks_x, peaks_y, labels):
        sym = "." if lab == 0 else "-"
        axes[1].scatter(px, py, s=100, color='green', marker='^', zorder=5)
        axes[1].text(px, py + 1, sym, ha='center', fontsize=10, fontweight='bold')
    
    # Mark gap thresholds
    for i, gap in enumerate(gaps):
        color = 'red' if gap_labels[i] == 1 else 'blue'
        axes[1].plot([peaks_x[i], peaks_x[i+1]], [0, 0], color=color, linewidth=3, alpha=0.5)

axes[1].set_xlabel('x')
axes[1].set_ylabel('Peak height')
axes[1].set_title('Peaks with symbols (red line = letter gap)')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([peaks_y.min() - 10, peaks_y.max() + 10])

plt.tight_layout()
plt.savefig('spline_analysis.png', dpi=100)
print("\nPlot saved to spline_analysis.png")
