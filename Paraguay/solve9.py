import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.signal import find_peaks

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()

print(f"Data: {len(x)} points")
print(f"x range: [{x.min():.6f}, {x.max():.6f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# Use polynomial features to fit the signal
# Higher degree = more overfitting (captures detail)
# Lower alpha = more overfitting

degree = 30
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(x)

# Fit with VERY weak regularization to allow overfitting and capture oscillations
model = Ridge(alpha=0.001)
model.fit(X_poly, y)

# Evaluate on original data
y_pred = model.predict(X_poly)
r2 = model.score(X_poly, y)

print(f"\nPolynom ial model (degree={degree}):")
print(f"  R² score: {r2:.6f}")
print(f"  Mean squared error: {np.mean((y - y_pred) ** 2):.4f}")

# Generate smooth prediction on fine grid
x_fine = np.linspace(x.min(), x.max(), 10000).reshape(-1, 1)
X_poly_fine = poly.transform(x_fine)
y_fine = model.predict(X_poly_fine)

print(f"  Predicted y range: [{y_fine.min():.2f}, {y_fine.max():.2f}]")

# Find peaks in the polynomial fit
x_flat = x_fine.ravel()
peaks_idx, peaks_props = find_peaks(y_fine, height=0.0, distance=100)
valleys_idx, valleys_props = find_peaks(-y_fine, distance=100)

print(f"\nFound {len(peaks_idx)} peaks and {len(valleys_idx)} valleys")

if len(peaks_idx) > 10:
    peaks_x = x_flat[peaks_idx]
    peaks_y = y_fine[peaks_idx]
    peaks_x = x_flat[peaks_idx]
    peaks_y = y_fine[peaks_idx]
    
    print(f"Peak heights: min={peaks_y.min():.2f}, max={peaks_y.max():.2f}, mean={peaks_y.mean():.2f}")
    
    # Cluster peak heights into dot/dash
    from sklearn.cluster import KMeans
    peaks_y_2d = peaks_y.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
    kmeans.fit(peaks_y_2d)
    
    centers = np.sort(kmeans.cluster_centers_.ravel())
    print(f"Height clusters: {centers}")
    
    # Map to symbols
    labels = kmeans.predict(peaks_y_2d)
    morse = []
    for label in labels:
        morse.append('.' if label == 0 else '-')
    
    morse_str = ''.join(morse)
    print(f"\nMorse code ({len(morse)} symbols): {morse_str[:100]}...")
    
    # Analyze gaps for letters/words
    gaps = np.diff(peaks_x)
    print(f"\nGaps: min={gaps.min():.6f}, max={gaps.max():.6f}, mean={gaps.mean():.6f}")
    print(f"  Percentiles: 10%={np.percentile(gaps, 10):.6f}, 50%={np.percentile(gaps, 50):.6f}, 90%={np.percentile(gaps, 90):.6f}")
    
    # Use a simple threshold approach
    gap_threshold = np.percentile(gaps, 50)
    
    # Build letters
    morse_letters = []
    current_letter = []
    
    for i, sym in enumerate(morse):
        current_letter.append(sym)
        
        if i < len(gaps):
            if gaps[i] > gap_threshold * 2:  # Large gap = word boundary
                morse_letters.append(''.join(current_letter))
                morse_letters.append('/')
                current_letter = []
            elif gaps[i] > gap_threshold:  # Small gap = letter boundary
                morse_letters.append(''.join(current_letter))
                current_letter = []
    
    if current_letter:
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
    print(f"\nDECODED: {result}")

# Try plotting
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].scatter(x, y, alpha=0.3, s=2, label='Data')
axes[0].plot(x_fine, y_fine, 'r-', linewidth=1, label='Polynomial fit')
if len(peaks_idx) > 10:
    axes[0].scatter(x_flat[peaks_idx], y_fine[peaks_idx], s=30, color='green', marker='^', label='Peaks', zorder=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Polynomial fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if len(peaks_idx) > 10:
    # Plot peaks with labels
    peaks_x_plot = x_flat[peaks_idx]
    peaks_y_plot = y_fine[peaks_idx]
    for px, py, label, sym in zip(peaks_x_plot, peaks_y_plot, labels, morse):
        axes[1].scatter(px, py, s=100, color='green' if label == 0 else 'red', marker='^')
        axes[1].text(px, py + max(peaks_y_plot) * 0.05, sym, ha='center', fontsize=8)
    
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Peak height')
    axes[1].set_title(f'Peaks with Morse symbols')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('poly_decode.png', dpi=100)
print("\nPlot saved to poly_decode.png")
