"""
Best approach to reveal Paraguay keyword - based on Brazil method.
Sorts data by x coordinate and arranges into matrix with per-row thresholding.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def reveal_keyword(csv_path, rows=50, cols=300, transpose=False, downsample_h=1, downsample_w=1):
    """
    Reveal hidden keyword from noisy regression data.
    
    Args:
        csv_path: Path to train.csv
        rows: Number of rows in matrix
        cols: Number of columns in matrix
        transpose: Whether to transpose the matrix
        downsample_h: Height downsampling factor
        downsample_w: Width downsampling factor
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} points")
    
    # Sort by x coordinate
    df_sorted = df.sort_values('x').reset_index(drop=True)
    
    # Calculate how many points fit in the matrix
    n_cells = rows * cols
    if len(df_sorted) > n_cells:
        df_sorted = df_sorted.iloc[:n_cells]
        print(f"Using first {n_cells} points")
    
    # Extract y values
    values = df_sorted.y.values
    n_actual = len(values)
    
    # Pad with NaN if needed
    if n_actual < n_cells:
        values = np.concatenate([values, np.full(n_cells - n_actual, np.nan)])
    
    # Reshape into matrix
    matrix = values.reshape(rows, cols)
    print(f"Matrix shape: {matrix.shape}")
    
    # Per-row median thresholding (like Brazil)
    binary = np.zeros_like(matrix, dtype=int)
    for i in range(rows):
        row = matrix[i, :]
        valid = ~np.isnan(row)
        if valid.any():
            threshold = np.median(row[valid])
            binary[i, valid] = (row[valid] > threshold).astype(int)
    
    # Transpose if requested
    if transpose:
        binary = binary.T
        print(f"Transposed to: {binary.shape}")
    
    # Downsample if requested
    if downsample_h > 1 or downsample_w > 1:
        h, w = binary.shape
        new_h = h // downsample_h
        new_w = w // downsample_w
        downsampled = binary[:new_h*downsample_h, :new_w*downsample_w]
        downsampled = downsampled.reshape(new_h, downsample_h, new_w, downsample_w).mean(axis=(1, 3))
        binary = (downsampled > 0.5).astype(int)
        print(f"Downsampled to: {binary.shape}")
    
    return binary


def visualize(binary, title="Hidden Keyword"):
    """Display the binary matrix as an image and ASCII art"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(binary, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'keyword_revealed.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.show()
    
    # ASCII art
    print("\n" + "="*100)
    print("ASCII REPRESENTATION:")
    print("="*100)
    for row in binary:
        line = ''.join(['█' if pixel else ' ' for pixel in row])
        print(line)
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Reveal keyword from Paraguay dataset')
    parser.add_argument('--rows', type=int, default=50, help='Number of rows')
    parser.add_argument('--cols', type=int, default=300, help='Number of columns')
    parser.add_argument('--transpose', action='store_true', help='Transpose the matrix')
    parser.add_argument('--dh', type=int, default=1, help='Height downsampling factor')
    parser.add_argument('--dw', type=int, default=1, help='Width downsampling factor')
    args = parser.parse_args()
    
    csv_path = Path(__file__).parent / 'train.csv'
    
    print(f"\nConfiguration:")
    print(f"  Rows: {args.rows}")
    print(f"  Cols: {args.cols}")
    print(f"  Transpose: {args.transpose}")
    print(f"  Downsample: {args.dh}x{args.dw}")
    print()
    
    binary = reveal_keyword(csv_path, args.rows, args.cols, args.transpose, args.dh, args.dw)
    visualize(binary, f"Paraguay Keyword ({args.rows}x{args.cols})")


if __name__ == "__main__":
    main()
