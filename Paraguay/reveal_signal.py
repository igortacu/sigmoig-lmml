import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import find_peaks


def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV."""
    df = pd.read_csv(csv_path)
    X = df["x"].values
    y = df["y"].values
    return X, y


def create_fourier_features(X: np.ndarray, n_freqs: int = 50) -> np.ndarray:
    """Create Fourier (sinusoidal) features for overfitting."""
    features = [np.ones_like(X)]  # bias term
    for k in range(1, n_freqs + 1):
        features.append(np.sin(2 * np.pi * k * X))
        features.append(np.cos(2 * np.pi * k * X))
    return np.column_stack(features)


def create_polynomial_features(X: np.ndarray, degree: int = 20) -> np.ndarray:
    """Create polynomial features for overfitting."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(X.reshape(-1, 1))


def overfit_model(X: np.ndarray, y: np.ndarray, feature_type: str = "fourier", 
                  n_features: int = 100) -> tuple:
    """
    Train a heavily overfitted model to memorize the data.
    
    feature_type: 'fourier' or 'polynomial'
    n_features: number of frequency components or polynomial degree
    """
    if feature_type == "fourier":
        X_features = create_fourier_features(X, n_freqs=n_features)
    else:
        X_features = create_polynomial_features(X, degree=n_features)
    
    # Use Ridge with very small alpha (almost no regularization)
    model = Ridge(alpha=1e-6, max_iter=10000)
    model.fit(X_features, y)
    
    return model, X_features


def predict_smooth(model, X_features_func, x_range: np.ndarray) -> np.ndarray:
    """Generate smooth predictions over a range."""
    X_feat = X_features_func(x_range)
    return model.predict(X_feat)


def extract_peaks_valleys(x_smooth: np.ndarray, y_smooth: np.ndarray, 
                          prominence: float = 1.0) -> tuple:
    """Extract peaks and valleys from the smooth prediction."""
    peaks, peak_props = find_peaks(y_smooth, prominence=prominence)
    valleys, valley_props = find_peaks(-y_smooth, prominence=prominence)
    
    return peaks, valleys, peak_props, valley_props


def decode_message_from_peaks(peak_x_positions: np.ndarray, 
                               peak_y_heights: np.ndarray,
                               gap_symbol: float = 0.1, 
                               gap_letter: float = 0.3) -> tuple[list, str]:
    """
    Decode message from peak positions and heights based on gaps.
    
    According to the clue:
    - Gaps of ~1 unit (normalized: 0.1) separate symbols (dots/dashes in Morse)
    - Gaps of ~3 units (normalized: 0.3) separate letters
    - "Mountains of different heights" - height determines dot vs dash
    
    Returns: (morse_letters, decoded_text)
    """
    if len(peak_x_positions) < 2:
        return [], ""
    
    # Sort positions and heights together
    sorted_indices = np.argsort(peak_x_positions)
    positions = peak_x_positions[sorted_indices]
    heights = peak_y_heights[sorted_indices]
    
    # Calculate gaps between consecutive peaks
    gaps = np.diff(positions)
    
    # Classify gaps: small = within letter, large = between letters
    gap_median = np.median(gaps)
    gap_std = np.std(gaps)
    
    # Threshold: gaps smaller than this are within-letter, larger are between-letters
    letter_threshold = gap_median + gap_std
    
    # Classify peak heights: higher = dash, lower = dot
    height_median = np.median(heights)
    
    print(f"      Gap analysis: median={gap_median:.4f}, std={gap_std:.4f}, threshold={letter_threshold:.4f}")
    print(f"      Height analysis: median={height_median:.4f}, range=[{heights.min():.2f}, {heights.max():.2f}]")
    
    morse_code = []
    current_letter = []
    
    for i, gap in enumerate(gaps):
        # Use peak height to determine dot vs dash
        symbol = '-' if heights[i] > height_median else '.'
        current_letter.append(symbol)
        
        # Check if next gap indicates letter boundary
        if gap >= letter_threshold:
            if current_letter:
                morse_code.append("".join(current_letter))
                current_letter = []
    
    # Add last letter (don't forget the final peak)
    symbol = '-' if heights[-1] > height_median else '.'
    current_letter.append(symbol)
    if current_letter:
        morse_code.append("".join(current_letter))
    
    # Try to decode
    decoded = morse_to_text(morse_code)
    
    return morse_code, decoded


def morse_to_text(morse_code: list[str]) -> str:
    """Convert Morse code to text."""
    morse_dict = {
        '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
        '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
        '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
        '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
        '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
        '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
        '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
        '----.': '9'
    }
    
    text = []
    for code in morse_code:
        letter = morse_dict.get(code, '?')
        text.append(letter)
    
    return "".join(text)


def main():
    base = Path(__file__).parent
    csv_path = base / "train.csv"
    
    print("[1/5] Loading data...")
    X, y = load_data(csv_path)
    print(f"      Loaded {len(X)} points")
    
    print("\n[2/5] Training overfit model with Fourier features...")
    # Try high-frequency Fourier features first
    n_freqs = 200
    model, _ = overfit_model(X, y, feature_type="fourier", n_features=n_freqs)
    
    print("\n[3/5] Generating smooth predictions...")
    x_smooth = np.linspace(0, 1, 10000)
    y_smooth = predict_smooth(
        model, 
        lambda x: create_fourier_features(x, n_freqs=n_freqs),
        x_smooth
    )
    
    print("\n[4/5] Extracting peaks and valleys...")
    peaks, valleys, peak_props, valley_props = extract_peaks_valleys(
        x_smooth, y_smooth, prominence=0.5
    )
    
    peak_x = x_smooth[peaks]
    peak_y = y_smooth[peaks]
    valley_x = x_smooth[valleys]
    
    print(f"      Found {len(peaks)} peaks and {len(valleys)} valleys")
    
    print("\n[5/5] Decoding message from peak positions...")
    if len(peak_x) > 1:
        morse_letters, decoded_text = decode_message_from_peaks(peak_x, peak_y)
        print(f"\n      Morse code: {morse_letters}")
        print(f"      Decoded text: {decoded_text}")
        print(f"\n      FLAG: SIGMOID_{decoded_text}")
    else:
        print("      Not enough peaks found. Try adjusting prominence or n_freqs.")
    
    # Also try with different prominence values
    print("\n[DEBUG] Trying different prominence values...")
    for prom in [0.5, 1.0, 2.0, 3.0, 5.0]:
        peaks_test, _, _, _ = extract_peaks_valleys(x_smooth, y_smooth, prominence=prom)
        print(f"        prominence={prom}: {len(peaks_test)} peaks")
    
    # Plot to visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: scatter + smooth fit
    ax1.scatter(X, y, alpha=0.3, s=1, c='gray', label='Noisy data')
    ax1.plot(x_smooth, y_smooth, 'r-', linewidth=1, label='Overfit model')
    if len(peak_x) > 0:
        ax1.plot(peak_x, peak_y, 'g^', markersize=8, label=f'Peaks ({len(peak_x)})')
    if len(valley_x) > 0:
        ax1.plot(valley_x, y_smooth[valleys], 'bv', markersize=8, label=f'Valleys ({len(valley_x)})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Overfitted Model (Fourier n={n_freqs})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: just the smooth signal
    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2)
    if len(peak_x) > 0:
        ax2.plot(peak_x, peak_y, 'g^', markersize=10, label=f'Peaks (n={len(peak_x)})')
        # Draw vertical lines at peaks to show spacing
        for px in peak_x:
            ax2.axvline(px, color='g', alpha=0.2, linestyle='--', linewidth=0.5)
    if len(valley_x) > 0:
        ax2.plot(valley_x, y_smooth[valleys], 'bv', markersize=10, label=f'Valleys (n={len(valley_x)})')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Fitted y')
    ax2.set_title('Signal Pattern - Peak Spacing Shows Morse Code')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_png = base / "signal_analysis.png"
    plt.savefig(out_png, dpi=150)
    print(f"\n[OK] Saved visualization to {out_png}")
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    if len(peak_x) > 0:
        print(f"  - Number of peaks: {len(peak_x)}")
        print(f"  - Peak x positions: {peak_x[:20]}")
        if len(peak_x) > 1:
            gaps = np.diff(peak_x)
            print(f"  - Gaps between peaks: {gaps[:20]}")
            print(f"  - Gap range: [{gaps.min():.4f}, {gaps.max():.4f}]")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
