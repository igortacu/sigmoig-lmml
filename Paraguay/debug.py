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
morse = ['.' if label == 0 else '-' for label in labels]

# Analyze gaps
gaps = np.diff(peaks_x)
gaps_2d = gaps.reshape(-1, 1)
kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans_gaps.fit(gaps_2d)
gap_labels = kmeans_gaps.predict(gaps_2d)

# Print detailed analysis
print("Peak index | X pos   | Height   | Symbol | Gap to next | Gap type")
print("-" * 70)
for i in range(len(morse)):
    gap_str = f"{gaps[i]:.6f}" if i < len(gaps) else "       END"
    gap_type = "LETTER" if (gap_labels[i] == 1 if i < len(gap_labels) else False) else "SYMBOL"
    print(f"{i:3d}        | {peaks_x[i]:.6f} | {peaks_y[i]:8.2f} | {morse[i]:6s} | {gap_str:11s} | {gap_type}")

# Build letters using gap information
print("\n\nBuilding morse letters:")
morse_letters = []
current_letter = []

for i, symbol in enumerate(morse):
    current_letter.append(symbol)
    
    # Check if there's a gap after this symbol
    if i < len(gap_labels):
        if gap_labels[i] == 1:  # Letter boundary
            letter_code = ''.join(current_letter)
            morse_letters.append(letter_code)
            print(f"Letter #{len(morse_letters)}: {letter_code}")
            morse_letters.append('/')
            current_letter = []

# Don't forget the last letter
if current_letter:
    letter_code = ''.join(current_letter)
    morse_letters.append(letter_code)
    print(f"Letter #{len(morse_letters)}: {letter_code}")

print(f"\nMorse letters list: {morse_letters}")

# Decode
MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
    "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
    "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
    "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}

print("\n\nDecoding:")
decoded = []
for code in morse_letters:
    if code == '/':
        decoded.append(' ')
        print(f"  / -> (space)")
    elif code in MORSE:
        letter = MORSE[code]
        decoded.append(letter)
        print(f"  {code} -> {letter}")
    else:
        decoded.append('?')
        print(f"  {code} -> ? (UNKNOWN)")

result = ''.join(decoded)
print(f"\nFinal result: {result}")
