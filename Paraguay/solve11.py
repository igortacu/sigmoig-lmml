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

print(f"Data y range: [{y_fine.min():.2f}, {y_fine.max():.2f}]")

# Find peaks with HIGH prominence (only significant ones)
for prominence_threshold in [10.0, 20.0, 50.0, 100.0]:
    peaks_idx, props = find_peaks(y_fine, prominence=prominence_threshold)
    print(f"\nProminence={prominence_threshold}: {len(peaks_idx)} peaks")
    
    if 5 < len(peaks_idx) < 200:
        peaks_x = x_fine[peaks_idx]
        peaks_y = y_fine[peaks_idx]
        
        print(f"  Peak heights: {peaks_y}")
        print(f"  Peak x positions: {peaks_x}")
        print(f"  Gaps between peaks: {np.diff(peaks_x)}")
        
        # Cluster heights
        from sklearn.cluster import KMeans
        peaks_y_2d = peaks_y.reshape(-1, 1)
        
        try:
            kmeans = KMeans(n_clusters=2, n_init=20, random_state=0)
            kmeans.fit(peaks_y_2d)
            
            centers = np.sort(kmeans.cluster_centers_.ravel())
            labels = kmeans.predict(peaks_y_2d)
            
            morse = ['.' if l == 0 else '-' for l in labels]
            morse_str = ''.join(morse)
            
            print(f"  Morse: {morse_str}")
            
            # Try to decode
            MORSE = {
                ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
                "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
                "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
                "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
                "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
                ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
            }
            
            # Simple approach: interpret pairs as letters
            result = []
            for i in range(0, len(morse), 2):
                if i + 1 < len(morse):
                    pair = morse[i] + morse[i+1]
                elif i < len(morse):
                    pair = morse[i]
                else:
                    continue
                
                if pair in MORSE:
                    result.append(MORSE[pair])
                else:
                    result.append("?")
            
            print(f"  Decoded (pairs): {''.join(result)}")
            
        except Exception as e:
            print(f"  Error: {e}")

# Also try with very high prominence to get only the most significant peaks
print("\n" + "="*60)
print("Trying ONLY the highest peaks:")
peaks_idx, props = find_peaks(y_fine, prominence=200.0)
if len(peaks_idx) > 0:
    peaks_x = x_fine[peaks_idx]
    peaks_y = y_fine[peaks_idx]
    
    print(f"Found {len(peaks_idx)} very significant peaks")
    for i, (px, py) in enumerate(zip(peaks_x, peaks_y)):
        print(f"  Peak {i}: x={px:.6f}, y={py:.2f}")
