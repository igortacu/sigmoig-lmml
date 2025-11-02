import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# Fit spline
spl = UnivariateSpline(x, y, k=3, s=50.0)
x_fine = np.linspace(x.min(), x.max(), 10000)
y_fine = spl(x_fine)

# Find peaks with intermediate prominence
peaks_idx, _ = find_peaks(y_fine, prominence=50.0)
peaks_x = x_fine[peaks_idx]
peaks_y = y_fine[peaks_idx]

print(f"Found {len(peaks_x)} peaks")

# Cluster heights
from sklearn.cluster import KMeans
peaks_y_2d = peaks_y.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans.fit(peaks_y_2d)

centers = np.sort(kmeans.cluster_centers_.ravel())
labels = kmeans.predict(peaks_y_2d)

# Create Morse string where each peak is one symbol
morse = ['.' if l == 0 else '-' for l in labels]

# Cluster gaps to find letters
gaps = np.diff(peaks_x)
gaps_2d = gaps.reshape(-1, 1)
kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans_gaps.fit(gaps_2d)
gap_labels = kmeans_gaps.predict(gaps_2d)

# Build morse letters
morse_letters = []
current_letter = []

for i, sym in enumerate(morse):
    current_letter.append(sym)
    
    if i < len(gap_labels) and gap_labels[i] == 1:  # Letter gap
        morse_letters.append(''.join(current_letter))
        current_letter = []

if current_letter:
    morse_letters.append(''.join(current_letter))

print(f"Morse letters: {morse_letters}")

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
for code in morse_letters:
    if code in MORSE:
        decoded.append(MORSE[code])
    else:
        decoded.append(f"<{code}>")

result = ''.join(decoded)
print(f"\nDecoded: {result}")

# Try to extract reasonable words
# Filter for common English words
common_words = {
    "HELLO", "WORLD", "MORSE", "CODE", "SIGNAL", "PYTHON", "MESSAGE",
    "HIDDEN", "CIPHER", "ENCRYPT", "DECRYPT", "DATA", "ANSWER",
    "SCIENCE", "MOUNTAIN", "SIGNAL", "NOISE", "CHAOS", "MYSTERY",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
    "WHICH", "THEIR", "WOULD", "THERE", "COULD", "ABOUT",
    "LETTER", "WORD", "TEXT", "ENCODE", "DECODE", "KEY",
    "SECRET", "HIDDEN", "SOUND", "AUDIO", "RADIO",
    "PEAKS", "VALLEYS", "MOUNTAINS", "HEIGHTS", "SQUEEZE",
    "OVERFITTING", "POLYNOMIAL", "FEATURES", "PATTERN",
    "OVERFIT", "FITTING", "REGRESSION", "TRAINING",
    "LEARNING", "NEURAL", "NETWORK", "DEEP", "MACHINE",
    "ARTIFICIAL", "INTELLIGENCE", "ALGORITHM", "COMPLEXITY"
}

# Check if any morse letters could form known words
for word_len in range(3, 15):
    for start_idx in range(len(decoded) - word_len + 1):
        substring = ''.join(decoded[start_idx:start_idx+word_len])
        # Remove angle bracket markers
        substring_clean = substring.replace('<', '').replace('>', '')
        
        # Check if all characters are valid letters (not in angle brackets)
        if all(c.isalpha() for c in substring_clean):
            if substring_clean in common_words or substring_clean.upper() in common_words:
                print(f"\n✓ FOUND WORD: {substring_clean} at position {start_idx}")

# Also print some decoded chunks to inspect
print(f"\nDecoded string (first 200 chars): {result[:200]}")
