import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Data loaded: {len(x)} points")

# Create high-degree polynomial features for overfitting
degree = 15
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(x.reshape(-1, 1))

# Fit with regularization (high-degree polynomial to capture details)
model = Ridge(alpha=1.0)  # Regularization to avoid numerical issues
model.fit(X_poly, y)

# Generate smooth prediction
x_fine = np.linspace(x.min(), x.max(), 15000)
X_poly_fine = poly.transform(x_fine.reshape(-1, 1))
y_pred = model.predict(X_poly_fine)

print(f"Model fitted with R^2 score: {model.score(X_poly, y):.6f}")

# Find peaks and valleys by taking derivative
# Approximate derivative using finite differences
dy_dx = np.diff(y_pred) / np.diff(x_fine)

# Find zero crossings (where derivative changes sign)
zero_crossings = np.where(np.diff(np.sign(dy_dx)))[0]

print(f"Found {len(zero_crossings)} extrema (peaks/valleys)")

# Classify as peaks (positive to negative) or valleys (negative to positive)
extrema_x = x_fine[zero_crossings]
extrema_y = y_pred[zero_crossings]

# Determine if each is a peak or valley
is_peak = []
for i in zero_crossings:
    if i > 0 and i < len(dy_dx) - 1:
        if dy_dx[i-1] > 0 and dy_dx[i] < 0:  # Peak (slope changes from + to -)
            is_peak.append(True)
        elif dy_dx[i-1] < 0 and dy_dx[i] > 0:  # Valley (slope changes from - to +)
            is_peak.append(False)
        else:
            is_peak.append(None)
    else:
        is_peak.append(None)

is_peak = np.array(is_peak)

# Filter out None values
valid_mask = is_peak != None
extrema_x_valid = extrema_x[valid_mask]
extrema_y_valid = extrema_y[valid_mask]
is_peak_valid = is_peak[valid_mask].astype(bool)

print(f"Valid extrema: {len(extrema_x_valid)}")
print(f"Peaks: {np.sum(is_peak_valid)}, Valleys: {np.sum(~is_peak_valid)}")

# Separate peaks and valleys
peak_x = extrema_x_valid[is_peak_valid]
peak_y = extrema_y_valid[is_peak_valid]
valley_x = extrema_x_valid[~is_peak_valid]
valley_y = extrema_y_valid[~is_peak_valid]

print(f"\nPeak heights: min={peak_y.min():.2f}, max={peak_y.max():.2f}, mean={peak_y.mean():.2f}")
print(f"Valley depths: min={valley_y.min():.2f}, max={valley_y.max():.2f}, mean={valley_y.mean():.2f}")

# Cluster peak heights into 2 groups (dot vs dash)
from sklearn.cluster import KMeans
peak_y_2d = peak_y.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans.fit(peak_y_2d)

centers = np.sort(kmeans.cluster_centers_.ravel())
print(f"\nHeight clusters: {centers}")

# Create Morse code sequence
morse_sequence = []
labels = kmeans.predict(peak_y_2d)

for peak_label in labels:
    if peak_label == 0:
        morse_sequence.append(".")
    else:
        morse_sequence.append("-")

print(f"Morse sequence ({len(morse_sequence)} symbols): {''.join(morse_sequence)}")

# Analyze gaps between consecutive peaks
gaps = np.diff(peak_x)
print(f"\nGap analysis:")
print(f"  Min gap: {gaps.min():.6f}")
print(f"  Max gap: {gaps.max():.6f}")
print(f"  Mean gap: {gaps.mean():.6f}")

# Cluster gaps
if len(gaps) > 1:
    gaps_2d = gaps.reshape(-1, 1)
    kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
    kmeans_gaps.fit(gaps_2d)
    
    gap_centers = np.sort(kmeans_gaps.cluster_centers_.ravel())
    print(f"  Gap clusters: {gap_centers}")
    
    # Interpret as letters (short gap) vs words (long gap)
    gap_labels = kmeans_gaps.predict(gaps_2d)
    
    morse_letters = []
    current_letter = []
    
    for i, (morse_sym, gap_label) in enumerate(zip(morse_sequence, gap_labels)):
        current_letter.append(morse_sym)
        
        # Check if this is a word boundary
        if i == len(morse_sequence) - 1:  # Last symbol
            morse_letters.append(''.join(current_letter))
        elif gap_label == 1:  # Large gap = word boundary
            morse_letters.append(''.join(current_letter))
            morse_letters.append('/')
            current_letter = []
    
    print(f"\nMorse letters: {morse_letters}")
    
    # Decode Morse to text
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

# Plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Plot 1: Original data and polynomial fit
axes[0].scatter(x, y, alpha=0.3, s=2, label='Data', color='blue')
axes[0].plot(x_fine, y_pred, 'r-', linewidth=1, label='Polynomial fit')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f'Polynomial Fit (degree={degree})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Fit with extrema marked
axes[1].plot(x_fine, y_pred, 'r-', linewidth=1, label='Fit')
axes[1].scatter(peak_x, peak_y, s=50, color='green', marker='^', label='Peaks', zorder=5)
axes[1].scatter(valley_x, valley_y, s=30, color='orange', marker='v', label='Valleys', zorder=5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Detected extrema')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Derivative
axes[2].plot(x_fine[:-1], dy_dx, 'b-', linewidth=0.5, label='dy/dx')
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].scatter(extrema_x_valid, np.zeros(len(extrema_x_valid)), s=50, color='red', marker='x', zorder=5)
axes[2].set_xlabel('x')
axes[2].set_ylabel('dy/dx')
axes[2].set_title('Derivative and zero crossings')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('poly_fit_analysis.png', dpi=100)
print("\nPlot saved to poly_fit_analysis.png")
