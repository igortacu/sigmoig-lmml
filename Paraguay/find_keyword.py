"""
Multiple approaches to reveal the hidden keyword in Paraguay dataset.
Tries different binning, sorting, and visualization strategies.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def method1_grid_binning(df, rows=50, cols=100):
    """Grid binning: divide x,y into grid cells and average y values"""
    print(f"\n=== Method 1: Grid Binning ({rows}x{cols}) ===")
    
    x_bins = np.linspace(0, 1, cols + 1)
    y_bins = np.linspace(df.y.min(), df.y.max(), rows + 1)
    
    matrix = np.full((rows, cols), np.nan)
    
    for i in range(rows):
        for j in range(cols):
            mask = (
                (df.x >= x_bins[j]) & (df.x < x_bins[j+1]) &
                (df.y >= y_bins[i]) & (df.y < y_bins[i+1])
            )
            values = df[mask].y
            if len(values) > 0:
                matrix[rows-1-i, j] = values.mean()
    
    # Binary threshold
    valid_mask = ~np.isnan(matrix)
    if valid_mask.any():
        threshold = np.nanmedian(matrix)
        binary = np.where(valid_mask, matrix > threshold, 0)
        return binary
    return matrix


def method2_sorted_matrix(df, rows=50, cols=300):
    """Sort by x, then arrange into matrix rows"""
    print(f"\n=== Method 2: Sorted Matrix ({rows}x{cols}) ===")
    
    df_sorted = df.sort_values('x').reset_index(drop=True)
    total_points = len(df_sorted)
    
    # Take subset that fits exactly into matrix
    n_cells = rows * cols
    if total_points > n_cells:
        df_sorted = df_sorted.iloc[:n_cells]
    
    # Reshape into matrix
    values = df_sorted.y.values
    n_actual = len(values)
    
    # Pad if needed
    if n_actual < n_cells:
        values = np.concatenate([values, np.full(n_cells - n_actual, np.nan)])
    
    matrix = values.reshape(rows, cols)
    
    # Threshold per row (like Brazil approach)
    binary = np.zeros_like(matrix)
    for i in range(rows):
        row = matrix[i, :]
        valid = ~np.isnan(row)
        if valid.any():
            threshold = np.median(row[valid])
            binary[i, valid] = row[valid] > threshold
    
    return binary


def method3_residual_analysis(df, rows=50, cols=300):
    """Fit polynomial, extract residuals, arrange as matrix"""
    print(f"\n=== Method 3: Residual Analysis ({rows}x{cols}) ===")
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    
    # Fit a high-degree polynomial
    poly = PolynomialFeatures(degree=10)
    X_poly = poly.fit_transform(df.x.values.reshape(-1, 1))
    
    model = Ridge(alpha=0.1)
    model.fit(X_poly, df.y)
    
    y_pred = model.predict(X_poly)
    df['residual'] = df.y - y_pred
    
    # Sort and reshape residuals
    df_sorted = df.sort_values('x').reset_index(drop=True)
    
    n_cells = rows * cols
    if len(df_sorted) > n_cells:
        df_sorted = df_sorted.iloc[:n_cells]
    
    values = df_sorted.residual.values
    n_actual = len(values)
    
    if n_actual < n_cells:
        values = np.concatenate([values, np.full(n_cells - n_actual, np.nan)])
    
    matrix = values.reshape(rows, cols)
    
    # Per-row threshold
    binary = np.zeros_like(matrix)
    for i in range(rows):
        row = matrix[i, :]
        valid = ~np.isnan(row)
        if valid.any():
            threshold = np.median(row[valid])
            binary[i, valid] = row[valid] > threshold
    
    return binary


def method4_x_only_bins(df, rows=50, cols=300):
    """Bin only by x coordinate, use y values directly"""
    print(f"\n=== Method 4: X-only Binning ({rows}x{cols}) ===")
    
    df_sorted = df.sort_values('x').reset_index(drop=True)
    
    n_cells = rows * cols
    if len(df_sorted) > n_cells:
        df_sorted = df_sorted.iloc[:n_cells]
    
    values = df_sorted.y.values
    n_actual = len(values)
    
    if n_actual < n_cells:
        values = np.concatenate([values, np.full(n_cells - n_actual, np.nan)])
    
    matrix = values.reshape(rows, cols)
    
    # Global threshold
    threshold = np.nanmedian(matrix)
    binary = matrix > threshold
    
    return binary


def visualize_and_save(binary, method_name, output_dir):
    """Visualize binary matrix and try to read it"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Show full resolution
    axes[0].imshow(binary, cmap='gray', aspect='auto', interpolation='nearest')
    axes[0].set_title(f'{method_name} - Full')
    axes[0].axis('off')
    
    # Show downsampled version
    h, w = binary.shape
    if h > 20 or w > 100:
        # Downsample
        gh = max(1, h // 20)
        gw = max(1, w // 100)
        downsampled = binary.reshape(h//gh, gh, w//gw, gw).mean(axis=(1, 3))
        downsampled = downsampled > 0.5
    else:
        downsampled = binary
    
    axes[1].imshow(downsampled, cmap='gray', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'{method_name} - Downsampled')
    axes[1].axis('off')
    
    output_file = output_dir / f'{method_name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")
    
    # Try to print as ASCII
    print("\nASCII representation (downsampled):")
    for row in downsampled:
        line = ''.join(['█' if pixel else ' ' for pixel in row])
        print(line)


def main():
    csv_path = Path(__file__).parent / 'train.csv'
    output_dir = Path(__file__).parent / 'keyword_search'
    output_dir.mkdir(exist_ok=True)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} points")
    
    methods = [
        ("Grid_50x100", lambda: method1_grid_binning(df, 50, 100)),
        ("Sorted_50x300", lambda: method2_sorted_matrix(df, 50, 300)),
        ("Sorted_40x375", lambda: method2_sorted_matrix(df, 40, 375)),
        ("Sorted_30x500", lambda: method2_sorted_matrix(df, 30, 500)),
        ("Residual_50x300", lambda: method3_residual_analysis(df, 50, 300)),
        ("XBin_50x300", lambda: method4_x_only_bins(df, 50, 300)),
        ("XBin_60x250", lambda: method4_x_only_bins(df, 60, 250)),
    ]
    
    for method_name, method_func in methods:
        try:
            binary = method_func()
            visualize_and_save(binary, method_name, output_dir)
        except Exception as e:
            print(f"ERROR in {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
