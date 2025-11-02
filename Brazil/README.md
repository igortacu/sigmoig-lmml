# Brazil — Temporal Vision: The Hidden Message

**Difficulty:** Medium

## 📖 Task Overview

The year is 2042. Attention Analytics has discovered encrypted messages hidden in time series datasets. This folder contains tools to process and visualize forecasting data to reveal a hidden word that appears when properly plotted.

## 🎯 Objective

Process and visualize the provided time series dataset to reveal a hidden word using binary per-row median thresholding and transposed visualization.

## 🚩 FLAG Format

`SIGMOID_{HIDDEN_WORD}`

Where `HIDDEN_WORD` is the word revealed in the visualization.

## 📂 Dataset

- Location: `Brazil/task_25/`
- Files: `dataset_part_1.csv` through `dataset_part_5.csv`
- Columns detected automatically:
  - `unique_id` or similar (series identifier)
  - `ds` or `timestamp` (time column)
  - `value` (primary numeric series)
  - `noise_level` (noise metric)

## 🛠️ Scripts

### 1. `merge_total.py`
Concatenates all dataset parts into a single CSV.

```powershell
python merge_total.py
```

**Output:** `task_25/total.csv` (10,660 rows)

### 2. `main.py`
Original visualization script that creates multiple matplotlib plots:
- Raw normalized grayscale
- Global binary threshold
- Binary per-row median (the key visualization)
- Transposed version

```powershell
python main.py
```

**Note:** Look at the "binary per row – transposed" plot to see the hidden word.

### 3. `reveal.py`
ASCII art terminal renderer with configurable downsampling to read the hidden word directly in console.

```powershell
# Basic usage (per-row median on 'value' column)
python reveal.py

# Try different columns and downsampling
python reveal.py --col value --mode row --gh 2 --gw 2
python reveal.py --col noise_level --mode global --gh 2 --gw 6
```

**Options:**
- `--col`: Column to visualize (`value` or `noise_level`)
- `--mode`: Thresholding mode (`row` or `global`)
- `--gh`: Row downsample factor (height)
- `--gw`: Column downsample factor (width)

### 4. `scatter_all.py`
Creates scatter plots of all numeric columns vs timestamp with optional filtering.

```powershell
python scatter_all.py
```

**Features:**
- Filters to `noise_level < 0.3` (660 rows from 10,660)
- Sorts by timestamp, then value
- Saves to `task_25/scatter_all.png`
- Opens matplotlib window

## 🔍 How to Reveal the Hidden Word

### Method 1: Matplotlib (Visual)
```powershell
python main.py
```
Look at the **"binary per row – transposed"** plot (usually the 4th figure for each column).

### Method 2: ASCII Terminal (Quick)
```powershell
python reveal.py --col value --mode row --gh 2 --gw 2
```
The hidden word appears as ASCII block characters (`█`) in the terminal.

### Method 3: Filtered Scatter
```powershell
python scatter_all.py
```
View the filtered time series scatter; pattern may be visible in the distribution.

## 📊 Data Processing Pipeline

1. **Load** all CSV parts from `task_25/`
2. **Detect** id, timestamp, and numeric columns automatically
3. **Build matrix** where rows = series, columns = timesteps
4. **Threshold** each row by its median (converts to binary 0/1)
5. **Transpose** to time × series layout
6. **Downsample** (optional) for readability
7. **Render** as grayscale or ASCII art

## 💡 Tips

- The hidden word is most visible in the **per-row median binary** visualization
- Try both `value` and `noise_level` columns
- Adjust downsampling (`--gh`, `--gw`) if letters are unclear
- The pattern emerges from the collective behavior of all time series
- Sorting and filtering can help isolate the signal

## 📝 Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib

Install dependencies:
```powershell
pip install pandas numpy matplotlib
```

## 🎁 Outputs

- `task_25/total.csv` — merged dataset
- `task_25/scatter_all.png` — filtered scatter plots
- Terminal ASCII art from `reveal.py`
- Matplotlib figures from `main.py`

## 🚀 Quick Start

```powershell
# 1. Merge all parts (optional)
python merge_total.py

# 2. Reveal the hidden word in terminal
python reveal.py --col value --mode row --gh 2 --gw 2

# 3. Or view in matplotlib
python main.py
```

Look for the word in the transposed binary visualization and submit as `SIGMOID_{WORD}`.
