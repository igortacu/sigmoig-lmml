import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# Fit spline with MINIMAL smoothing to preserve oscillations
spl = UnivariateSpline(x, y, k=3, s=10.0)  # Very small s value = less smoothing

x_fine = np.linspace(x.min(), x.max(), 50000)
y_fine = spl(x_fine)

print(f"Data: {len(x)} points")
print(f"Spline fit y range: [{y_fine.min():.2f}, {y_fine.max():.2f}]")

# Binary encode: is the signal above/below median?
median_y = np.median(y_fine)
binary = (y_fine > median_y).astype(int)

print(f"\nMedian y: {median_y:.2f}")

# Find runs of consecutive 1s and 0s
runs = []
current_val = binary[0]
start_idx = 0

for i in range(1, len(binary)):
    if binary[i] != current_val:
        runs.append((current_val, start_idx, i, x_fine[start_idx:i].mean()))
        current_val = binary[i]
        start_idx = i

runs.append((current_val, start_idx, len(binary), x_fine[start_idx:].mean()))

print(f"Found {len(runs)} runs")

# Separate into HIGH runs (signal/symbols) and LOW runs (gaps)
high_runs = [r for r in runs if r[0] == 1]
low_runs = [r for r in runs if r[0] == 0]

print(f"High runs (signal): {len(high_runs)}")
print(f"Low runs (gaps): {len(low_runs)}")

# Get lengths of high runs (in x-space)
high_lengths = [(r[2] - r[1]) / len(x_fine) * (x.max() - x.min()) for r in high_runs]
low_lengths = [(r[2] - r[1]) / len(x_fine) * (x.max() - x.min()) for r in low_runs]

print(f"\nHigh run lengths (x-space): min={min(high_lengths):.6f}, max={max(high_lengths):.6f}, mean={np.mean(high_lengths):.6f}")
print(f"Low run lengths (x-space): min={min(low_lengths):.6f}, max={max(low_lengths):.6f}, mean={np.mean(low_lengths):.6f}")

# Cluster high run lengths into two groups: dots vs dashes
from sklearn.cluster import KMeans
high_lengths_arr = np.array(high_lengths).reshape(-1, 1)
kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
kmeans.fit(high_lengths_arr)

centers = np.sort(kmeans.cluster_centers_.ravel())
print(f"\nHigh run clusters: {centers}")

# Map to morse
labels = kmeans.predict(high_lengths_arr)
morse = ['.' if l == 0 else '-' for l in labels]
morse_str = ''.join(morse)

print(f"Morse code ({len(morse)} symbols): {morse_str}")

# Now use low run lengths to detect letters/words
low_lengths_arr = np.array(low_lengths).reshape(-1, 1)
if len(low_runs) > 1:
    kmeans_gaps = KMeans(n_clusters=2, n_init=20, random_state=0)
    kmeans_gaps.fit(low_lengths_arr)
    gap_centers = np.sort(kmeans_gaps.cluster_centers_.ravel())
    gap_labels = kmeans_gaps.predict(low_lengths_arr)
    
    print(f"Gap clusters: {gap_centers}")
    
    # Build morse letters
    # Iterate through high runs and check gaps between them
    morse_letters = []
    current_letter = []
    
    for i, sym in enumerate(morse):
        current_letter.append(sym)
        
        # Check if there's a gap after this symbol
        if i < len(gap_labels):
            gap_label = gap_labels[i]
            if gap_label == 1:  # Large gap
                morse_letters.append(''.join(current_letter))
                morse_letters.append('/')
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

# Plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Plot 1: Data and fit
axes[0].scatter(x, y, alpha=0.3, s=2, label='Data')
axes[0].plot(x_fine, y_fine, 'r-', linewidth=0.5, label='Spline fit')
axes[0].axhline(y=median_y, color='g', linestyle='--', label=f'Median={median_y:.2f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Data with spline fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Binary signal
axes[1].fill_between(x_fine, binary, alpha=0.5, color='blue')
axes[1].plot(x_fine, binary, 'b-', linewidth=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Binary')
axes[1].set_title('Binary signal (high/low)')
axes[1].set_ylim([-0.5, 1.5])
axes[1].grid(True, alpha=0.3)

# Plot 3: Runs
for i, (val, start, end, mid_x) in enumerate(runs):
    if i < 50:  # Don't plot too many
        color = 'green' if val == 1 else 'red'
        length = (end - start) / len(x_fine) * (x.max() - x.min())
        axes[2].bar(mid_x, length, width=0.002, color=color, alpha=0.6)

axes[2].set_xlabel('x')
axes[2].set_ylabel('Run length')
axes[2].set_title('Run lengths (green=high, red=low)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binary_morse.png', dpi=100)
print("\nPlot saved to binary_morse.png")
