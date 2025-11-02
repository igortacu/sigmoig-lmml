import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Load and sort data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Data loaded: {len(x)} points")

# Normalize y
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

# Create interpolation to get regular samples
f = interp1d(x, y_norm, kind='cubic', fill_value='extrapolate')

# Sample at high resolution
x_dense = np.linspace(x.min(), x.max(), 10000)
y_dense = f(x_dense)

# Apply savitzky-golay filter for smoothing
y_smooth = savgol_filter(y_dense, window_length=101, polyorder=3)

# Threshold: high values = signal (dash/dot), low values = gap
# The idea: when smoothed y is high, we have a symbol; when low, we have a gap
threshold = np.percentile(y_smooth, 50)  # median

# Create binary signal
binary_signal = (y_smooth > threshold).astype(int)

# Find runs of 1s and 0s (symbols and gaps)
runs = []
current_val = binary_signal[0]
start_idx = 0

for i in range(1, len(binary_signal)):
    if binary_signal[i] != current_val:
        runs.append((current_val, start_idx, i))
        current_val = binary_signal[i]
        start_idx = i
runs.append((current_val, start_idx, len(binary_signal)))

# Analyze run lengths
symbol_runs = [(val, start, end) for val, start, end in runs if val == 1]
gap_runs = [(val, start, end) for val, start, end in runs if val == 0]

print(f"\nSymbol runs: {len(symbol_runs)}")
symbol_lengths = [(end - start) for _, start, end in symbol_runs]
print(f"  Lengths: min={min(symbol_lengths)}, max={max(symbol_lengths)}, mean={np.mean(symbol_lengths):.1f}")

print(f"Gap runs: {len(gap_runs)}")
gap_lengths = [(end - start) for _, start, end in gap_runs]
print(f"  Lengths: min={min(gap_lengths)}, max={max(gap_lengths)}, mean={np.mean(gap_lengths):.1f}")

# Cluster symbol lengths into dots vs dashes
from sklearn.cluster import KMeans
symbol_lengths_arr = np.array(symbol_lengths).reshape(-1, 1)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
kmeans.fit(symbol_lengths_arr)
symbol_labels = kmeans.labels_

centers = np.sort(kmeans.cluster_centers_.ravel())
dot_len = centers[0]
dash_len = centers[1]

print(f"\nSymbol clustering:")
print(f"  Dot length: {dot_len:.1f}")
print(f"  Dash length: {dash_len:.1f}")

# Map symbols
symbols = []
for i, (_, start, end) in enumerate(symbol_runs):
    length = end - start
    if kmeans.predict([[length]])[0] == 0:
        symbols.append(".")
    else:
        symbols.append("-")

print(f"\nSymbols extracted: {''.join(symbols)}")

# Now interpret gap positions to separate letters and words
morse_code = []
letter = []

for i, run_val in enumerate([r[0] for r in runs]):
    if i < len(runs):
        val, start, end = runs[i]
        length = end - start
        
        if val == 1:  # Symbol
            # Add next symbol to current letter
            symbol_idx = len([r for r in runs[:i] if r[0] == 1])
            if symbol_idx < len(symbols):
                letter.append(symbols[symbol_idx])
        else:  # Gap
            # Check gap length
            if length > np.mean(gap_lengths) * 2:  # Long gap = word separator
                if letter:
                    morse_code.append(''.join(letter))
                    letter = []
                morse_code.append('/')  # Word separator
            else:  # Short gap = letter separator
                if letter:
                    morse_code.append(''.join(letter))
                    letter = []

if letter:
    morse_code.append(''.join(letter))

print(f"\nMorse codes: {morse_code}")

# Decode Morse
MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
    "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
    "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
    "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}

decoded_text = []
for code in morse_code:
    if code == '/':
        decoded_text.append(' ')
    elif code in MORSE:
        decoded_text.append(MORSE[code])
    else:
        decoded_text.append('?')

result = ''.join(decoded_text)
print(f"\nDecoded: {result}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Plot 1: Original data
axes[0].scatter(x, y_norm, alpha=0.5, s=2, label='Data')
axes[0].plot(x_dense, y_smooth, 'r-', linewidth=1, label='Smoothed')
axes[0].axhline(y=threshold, color='g', linestyle='--', label=f'Threshold={threshold:.2f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Normalized y')
axes[0].set_title('Data and Smooth Fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Binary signal
axes[1].plot(x_dense, binary_signal, 'b-', linewidth=1)
axes[1].fill_between(x_dense, binary_signal, alpha=0.3)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Binary Signal')
axes[1].set_title('Binary Signal (1=Symbol, 0=Gap)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Run analysis
run_positions = []
run_types = []
run_lens = []
for i, (val, start, end) in enumerate(runs):
    mid = (start + end) / 2
    x_mid = x_dense[int(mid)]
    run_positions.append(x_mid)
    run_types.append(val)
    run_lens.append(end - start)

for i, (pos, typ, length) in enumerate(zip(run_positions, run_types, run_lens)):
    color = 'red' if typ == 1 else 'blue'
    axes[2].bar(pos, length, width=0.002, color=color, alpha=0.6)

axes[2].set_xlabel('x')
axes[2].set_ylabel('Run Length')
axes[2].set_title('Symbol (red) and Gap (blue) Lengths')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binary_analysis.png', dpi=100)
print("\nPlot saved to binary_analysis.png")
