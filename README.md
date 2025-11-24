# Image Compression using K-Means Color Quantization
CSE 4083 – Analysis of Algorithms Term Project  
Group Members: Woroma Dimkpa, Christian Prieto, Keegan McNear

## Project Overview
This project implements lossy image compression by reducing the number of unique colors in an image using K-Means clustering. The algorithm clusters similar colors together and replaces them with cluster centroids, achieving significant file size reduction while maintaining visual quality.

## Algorithm Complexity
- **Time Complexity**: O(n × k × i) where n = number of pixels, k = number of clusters, i = iterations
- **Space Complexity**: O(n + k)

## Features
- K-Means clustering implemented from scratch (no sklearn)
- Automatic image resizing for memory efficiency
- Support for multiple K values in a single run
- Comprehensive metrics (MSE, PSNR, compression ratio, color reduction)
- Visual comparisons (side-by-side, histograms)
- Test image generator
- JSON metrics export

## Folder Structure
```
src/                    → Python source code
  kmeans_compression.py → Main compression script
  utils.py              → Utility functions
images/                 → Input images for testing
output/                 → Compressed images and visualizations
  tests/                → Auto-generated test images
venv/                   → Virtual environment
requirements.txt        → Required Python libraries
README.md               → This file
```

## Installation

1. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python src\kmeans_compression.py <image_path> <k_values> [options]
```

### Examples

**Generate test images:**
```bash
python src\kmeans_compression.py --generate-tests
```

**Compress with single K value:**
```bash
python src\kmeans_compression.py images\photo.jpg 16
```

**Compress with multiple K values and visualizations:**
```bash
python src\kmeans_compression.py images\photo.jpg 8 16 32 --visualize
```

**Full analysis with metrics export:**
```bash
python src\kmeans_compression.py images\photo.jpg 4 8 16 32 64 --visualize --save-metrics
```

**Adjust convergence parameters:**
```bash
python src\kmeans_compression.py images\photo.jpg 16 --max-iterations 50 --tolerance 0.001
```

### Command Line Options
- `image_path`: Path to input image (required unless using --generate-tests)
- `k_values`: Space-separated list of K values to try (e.g., 8 16 32)
- `--visualize`: Generate side-by-side comparisons and histograms
- `--save-metrics`: Export metrics to JSON file
- `--max-iterations`: Maximum iterations (default: 100)
- `--tolerance`: Convergence tolerance (default: 1e-4)
- `--out-dir`: Output directory (default: output)
- `--generate-tests`: Generate synthetic test images

## Metrics

The algorithm computes and displays:

1. **Color Reduction**: Original colors → Compressed colors (reduction ratio)
2. **MSE (Mean Squared Error)**: Average squared difference between original and compressed pixels
3. **PSNR (Peak Signal-to-Noise Ratio)**: Quality metric in decibels (30+ dB = good quality)
4. **File Size Compression Ratio**: Original file size / Compressed file size
5. **Iterations**: Number of iterations until convergence
6. **Processing Time**: Time taken for clustering

### Example Output
```
Results (K=16):
  Colors: 139,834 → 16
  Reduction: 8739.62x
  MSE: 60.98
  PSNR: 30.28 dB
  Sizes: 2,729,678 bytes -> 248,421 bytes
  Compression ratio (orig/comp): 10.99x
```

## Memory Optimization

The code automatically resizes large images (>1024px) to prevent memory errors. For a 4032×3024 image:
- Original: 12,192,768 pixels (would require 4.36 GB RAM)
- Resized: 1024×768 = 786,432 pixels (manageable memory usage)
- This allows the algorithm to run efficiently on standard computers while still demonstrating K-Means clustering effectively

## Output Files

After running, check the `output/` folder for:
- `imagename_k16.png` - Compressed image
- `imagename_compare_k16_timestamp.png` - Side-by-side comparison
- `imagename_hist_k16.png` - Color distribution histograms
- `imagename_metrics_timestamp.json` - Metrics data

## Implementation Notes

- K-Means implemented from scratch
- Random initialization (future: K-means++ for better convergence)
- Euclidean distance in RGB color space
- Convergence based on centroid movement threshold
- PNG output format (lossless encoding)

## Known Limitations & Future Work

1. **Performance**: O(n×k) distance computation can be slow for very large images
   - Future: Consider minibatch K-Means or approximate nearest neighbor methods
2. **Initialization**: Random initialization may lead to suboptimal clustering
   - Future: Implement K-means++ initialization
3. **File Format**: PNG doesn't always show proportional file size reduction
   - Future: Add JPEG output option for better compression ratios

## Testing

Test images are included in `output/tests/`:
- `solid.png` - Uniform color (minimal compression)
- `gradient.png` - Smooth color transition
- `checkerboard.png` - High contrast patterns
- `noise.png` - Random pixels (challenging case)
- `rings.png` - Concentric color patterns

These demonstrate different compression behaviors based on image characteristics.

## Requirements

- Python 3.7+
- NumPy (array operations)
- Pillow (image I/O)
- Matplotlib (optional, for visualizations)

## Course Information

**CSE 4083: Analysis of Algorithms**  
Term Project - Fall 2025  
Implementation of K-Means clustering for image compression via color quantization.
