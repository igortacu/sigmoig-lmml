import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("train.csv").sort_values("x")
x = df["x"].to_numpy()
y = df["y"].to_numpy()

print(f"Loaded {len(x)} points")
print(f"x range: [{x.min():.6f}, {x.max():.6f}]")

# Create spline fit
spl = UnivariateSpline(x, y, k=3, s=50.0)

# Sample at regular intervals along x
# Use a step size that makes sense for Morse code
# The x range is ~1.0, so try different step sizes

x_min, x_max = x.min(), x.max()
x_range = x_max - x_min

# Let's try to estimate the step from the data itself
# Use a finer grid first to detect peaks, then group them

x_fine = np.linspace(x_min, x_max, 20000)
y_fine = spl(x_fine)

# Find all significant peaks (potential symbols)
from scipy.signal import find_peaks
peaks_idx, _ = find_peaks(y_fine, distance=50, prominence=0.5)
peaks_x = x_fine[peaks_idx]
peaks_y = y_fine[peaks_idx]

print(f"\nFound {len(peaks_x)} peaks")

if len(peaks_x) < 10:
    print("Too few peaks, try smaller prominence")
    peaks_idx, _ = find_peaks(y_fine, distance=30, prominence=0.1)
    peaks_x = x_fine[peaks_idx]
    peaks_y = y_fine[peaks_idx]
    print(f"Now found {len(peaks_x)} peaks")

# Estimate grid step from peak distances
peak_gaps = np.diff(peaks_x)
mean_gap = peak_gaps.mean()

# Try multiple step sizes and pick the best one
# A good step size should create distinct "symbols" when we bin

best_result = None
best_length = float('inf')

for step_factor in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    step = mean_gap * step_factor
    
    # Create grid
    grid_x = np.arange(x_min, x_max + step, step)
    
    # Sample signal on grid
    grid_vals = []
    grid_x_centers = []
    
    for i in range(len(grid_x) - 1):
        left, right = grid_x[i], grid_x[i+1]
        # Find maximum in this bin
        mask = (peaks_x >= left) & (peaks_x < right)
        if np.any(mask):
            max_idx = np.argmax(peaks_y[mask])
            grid_vals.append(peaks_y[mask][max_idx])
            grid_x_centers.append((left + right) / 2)
    
    if len(grid_vals) < 5:
        continue
    
    grid_vals = np.array(grid_vals)
    
    # Cluster into dot/dash
    grid_vals_2d = grid_vals.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(grid_vals_2d)
    labels = kmeans.predict(grid_vals_2d)
    
    # Convert to morse
    morse = ''.join(['-' if l == 1 else '.' for l in labels])
    
    print(f"  Step factor {step_factor:.1f} (step={step:.6f}): {len(grid_vals)} cells -> {morse}")
    
    if len(grid_vals) < best_length:
        best_length = len(grid_vals)
        best_result = (step_factor, step, grid_vals, labels, morse, grid_x_centers)

if best_result:
    step_factor, step, grid_vals, labels, morse, grid_x_centers = best_result
    print(f"\nBest result (step factor {step_factor}): {morse}")
    print(f"Number of symbols: {len(labels)}")
    
    # Decode Morse
    MORSE = {
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
        "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
        "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
        "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
        "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
        ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    }
    
    # Simple approach: treat consecutive same symbols as part of one letter?
    # Or try to decode runs of symbols
    
    # Let's try to split by long runs of dots or dashes
    decoded = []
    i = 0
    while i < len(morse):
        # Find consecutive same symbols
        j = i
        while j < len(morse) and morse[j] == morse[i]:
            j += 1
        
        # We have morse[i:j] as a run of the same symbol
        # This could be a dash (one or more), or multiple dots/dashes
        
        if morse[i] == '-':
            # Could be a single dash or multiple dashes
            if j - i == 1:
                decoded.append("-")
            else:
                # Multiple dashes -> probably represents one long symbol
                decoded.append("-" * (j - i))
        else:  # dots
            # Could be multiple dots representing a single symbol
            decoded.append("." * (j - i))
        
        i = j
    
    print(f"\nDecoded symbols: {decoded}")
    
    # Map to text
    result = []
    for sym in decoded:
        if sym in MORSE:
            result.append(MORSE[sym])
        else:
            result.append("?")
    
    print(f"Result: {''.join(result)}")
