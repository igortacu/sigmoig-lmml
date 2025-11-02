import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# Fit a smooth spline
spl = UnivariateSpline(x, y, k=3, s=50.0)

# Sample at high resolution
x_fine = np.linspace(x.min(), x.max(), 10000)
y_fine = spl(x_fine)

# Find peaks
peaks_idx, _ = find_peaks(y_fine, distance=100, prominence=1.0)
valleys_idx, _ = find_peaks(-y_fine, distance=100, prominence=1.0)

peaks_x = x_fine[peaks_idx]
peaks_y = y_fine[peaks_idx]

# Cluster peak heights
peaks_y_2d = peaks_y.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans.fit(peaks_y_2d)

labels = kmeans.predict(peaks_y_2d)

print("Trying a different interpretation:")
print("Maybe peaks between large gaps form a single Morse symbol (dot or dash)")
print()

# Analyze gaps
gaps = np.diff(peaks_x)
gaps_2d = gaps.reshape(-1, 1)
kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans_gaps.fit(gaps_2d)
gap_labels = kmeans_gaps.predict(gaps_2d)

gap_centers = np.sort(kmeans_gaps.cluster_centers_.ravel())
symbol_gap = gap_centers[0]
letter_gap = gap_centers[1]

print(f"Symbol gap threshold: {symbol_gap:.6f}")
print(f"Letter gap threshold: {letter_gap:.6f}")
print()

# Group consecutive peaks between large gaps
# A "symbol" (dot or dash in Morse) = a group of peaks with small gaps
# A "letter boundary" = a large gap

groups = []
current_group = [0]  # Start with first peak

for i in range(len(gaps)):
    if gap_labels[i] == 1:  # Letter boundary (large gap)
        # End current group
        groups.append(current_group)
        current_group = [i + 1]
    else:  # Symbol gap (small gap)
        current_group.append(i + 1)

# Add last group
if current_group:
    groups.append(current_group)

print(f"Found {len(groups)} groups (letters)")
print()

# For each group, determine if it's a dot or dash based on peak heights
morse_letters = []

for group_idx, group in enumerate(groups):
    # Get the labels of all peaks in this group
    group_labels = [labels[i] for i in group]
    
    # Determine the dominant symbol (dot vs dash)
    # If more peaks are dashes (label 1), it's a dash; otherwise it's a dot
    num_dashes = sum(1 for l in group_labels if l == 1)
    num_dots = len(group_labels) - num_dashes
    
    if num_dashes > num_dots:
        symbol = "-"
    else:
        symbol = "."
    
    # For now, treat the whole group as one symbol
    morse_letters.append(symbol)
    
    print(f"Group {group_idx:2d} ({len(group)} peaks): {''.join(['-' if labels[i] == 1 else '.' for i in group])} -> {symbol}")

print()
print(f"Morse letters: {''.join(morse_letters)}")

# Decode
MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
    "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
    "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
    "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}

# Split morse_letters into words based on some pattern
# For now, assume each group is one letter in a word
result = ''
for i, morse_sym in enumerate(morse_letters):
    if morse_sym in MORSE:
        result += MORSE[morse_sym]
    else:
        result += "?"

print(f"\nDecoded (treating each group as one letter): {result}")
