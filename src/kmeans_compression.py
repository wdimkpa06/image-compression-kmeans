#!/usr/bin/env python3
"""
K-Means Image Compression
CSE 4083: Analysis of Algorithms Term Project

Authors: Woroma Dimkpa, Christian Prieto, Keegan McNear

WORK IN PROGRESS - See notes at bottom
"""

import numpy as np
from PIL import Image
import sys
import time
import os
import json
import argparse
from datetime import datetime

# optional visualization imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class KMeansImageCompressor:
    """
    K-Means clustering for image compression via color quantization.
    
    Time Complexity: O(n * k * i) where n=pixels, k=clusters, i=iterations right here 
    Space Complexity: O(n + k)
    """
    
    def __init__(self, k=16, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.iterations_taken = 0
        
    def load_image(self, image_path):
        """Load image and convert to RGB array."""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def initialize_centroids(self, pixels):
        """Initialize K centroids randomly."""
        n_pixels = pixels.shape[0]
        # if k > n_pixels, sample with replacement to avoid error
        replace = self.k > n_pixels
        random_indices = np.random.choice(n_pixels, size=self.k, replace=replace)
        return pixels[random_indices].astype(np.float64)
    
    def calculate_distances(self, pixels, centroids):
        """
        Calculate Euclidean distances between pixels and centroids.
        3D color space (RGB)
        """
        # broadcast subtraction and compute distances
        distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
        return distances
    
    def assign_clusters(self, pixels, centroids):
        """pixel assigment """
        distances = self.calculate_distances(pixels, centroids)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, pixels, labels):
        """centroid update """
        new_centroids = np.zeros((self.k, 3))
        for i in range(self.k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                new_centroids[i] = cluster_pixels.mean(axis=0)
            else:
                # If a centroid loses all its pixels, re-use previous (safer than leaving zeros)
                new_centroids[i] = self.centroids[i]
        return new_centroids
    
    def has_converged(self, old_centroids, new_centroids):
        """check for convergence """
        max_shift = np.max(np.abs(new_centroids - old_centroids))
        return max_shift < self.tolerance
    
    def fit(self, image_data):
        """
        main algorithm loop for K-Means clustering.
        1. Initialize centroids
        2. Iterate assignment and update steps
        3. Check for convergence
        4. Stop when converged or max iterations reached
        5. Record iterations taken
        """
        height, width, channels = image_data.shape
        pixels = image_data.reshape(-1, 3).astype(np.float64)
        
        self.centroids = self.initialize_centroids(pixels)
        
        print(f"Starting K-Means with K={self.k} clusters...")
        print(f"Image size: {width}x{height} = {len(pixels):,} pixels")
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Assign pixels to nearest centroid
            self.labels = self.assign_clusters(pixels, self.centroids)
            
            # Update centroids
            old_centroids = self.centroids.copy()
            self.centroids = self.update_centroids(pixels, self.labels)
            
            # Check convergence
            if self.has_converged(old_centroids, self.centroids):
                self.iterations_taken = iteration + 1
                print(f"✓ Converged after {self.iterations_taken} iterations")
                break
        else:
            self.iterations_taken = self.max_iterations
            print(f"! Reached max iterations ({self.max_iterations})")
        
        elapsed = time.time() - start_time
        print(f"Clustering took {elapsed:.2f} seconds")
    
    def compress(self, image_data):
        """Replace each pixel with its cluster centroid color."""
        height, width, channels = image_data.shape
        compressed_pixels = self.centroids[self.labels].astype(np.uint8)
        compressed_image = compressed_pixels.reshape(height, width, channels)
        return compressed_image
    
    def calculate_metrics(self, original, compressed, original_path=None, compressed_path=None):
        """
        Calculate metrics:
         - MSE
         - PSNR (correct formula)
         - number of unique colors before/after
         - file-size based compression ratio (if original_path and compressed_path provided and files exist)
         - iteration count (from self.iterations_taken)
        
        NOTE: original and compressed expected as NumPy arrays (uint8 or similar).
        """
        # convert to float for proper math
        original_f = original.astype(np.float64)
        compressed_f = compressed.astype(np.float64)

        # MSE - Mean Squared Error
        mse = np.mean((original_f - compressed_f) ** 2)

        # PSNR - Peak Signal to Noise Ratio (correct formula)
        # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        # MAX_I = 255 for 8-bit images
        if mse == 0:
            psnr = float('inf')
        else:
            max_i = 255.0
            # use 20*log10(max_i) - 10*log10(mse) to avoid overflow/underflow issues
            psnr = 20.0 * np.log10(max_i) - 10.0 * np.log10(mse)

        # Color counts
        original_colors = len(np.unique(original.reshape(-1, 3), axis=0))
        compressed_colors = len(np.unique(compressed.reshape(-1, 3), axis=0))

        # File size comparison only if paths provided and files exist
        original_size = None
        compressed_size = None
        compression_ratio = None

        if original_path and os.path.exists(original_path):
            try:
                original_size = os.path.getsize(original_path)
            except Exception:
                original_size = None

        if compressed_path and os.path.exists(compressed_path):
            try:
                compressed_size = os.path.getsize(compressed_path)
            except Exception:
                compressed_size = None

        if original_size and compressed_size and compressed_size > 0:
            compression_ratio = original_size / compressed_size

        return {
            'mse': float(mse),
            'psnr': float(psnr) if psnr != float('inf') else float('inf'),
            'original_colors': int(original_colors),
            'compressed_colors': int(compressed_colors),
            'iterations': int(self.iterations_taken),
            'original_size': int(original_size) if original_size is not None else None,
            'compressed_size': int(compressed_size) if compressed_size is not None else None,
            'compression_ratio': float(compression_ratio) if compression_ratio is not None else None
        }


def resize_if_needed(image_array, max_dimension=1024):
    """Resize image if too large to prevent memory errors."""
    height, width = image_array.shape[:2]
    
    if width <= max_dimension and height <= max_dimension:
        return image_array
    
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    print(f"Resizing: {width}x{height} → {new_width}x{new_height}")
    img = Image.fromarray(image_array)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(img_resized)


def save_compressed_image(image_array, k_value, original_path, out_dir="output"):
    """
    Save the compressed image.

    Improvements:
    - Creates an 'output/' folder automatically (or other provided out_dir)
    - Uses clean filenames: <originalname>_k<k>[_n].png
    - Avoids overwrites by adding an incrementing suffix if necessary
    - Returns the full path to the saved file, or None on error
    """
    # Get the original filename without extension
    base_name = os.path.splitext(os.path.basename(original_path))[0]

    # create output directory
    os.makedirs(out_dir, exist_ok=True)

    # build output filename
    filename = f"{base_name}_k{k_value}.png"
    full_path = os.path.join(out_dir, filename)

    # avoid overwriting
    counter = 1
    while os.path.exists(full_path):
        filename = f"{base_name}_k{k_value}_{counter}.png"
        full_path = os.path.join(out_dir, filename)
        counter += 1

    try:
        img = Image.fromarray(image_array)
        img.save(full_path)
        print(f"Saved: {full_path}")
        return full_path
    except Exception as e:
        print(f"ERROR saving file: {e}")
        return None


# Visualization helpers

def save_side_by_side(original_arr, compressed_arr, k_value, original_path, out_dir="output"):
    """
    Save a side-by-side comparison image (original | compressed).
    This is useful for presentation and quick visual checks.
    Returns the path to the saved file or None.
    """
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_compare_k{k_value}_{timestamp}.png"
    full_path = os.path.join(out_dir, filename)

    try:
        orig_img = Image.fromarray(original_arr)
        comp_img = Image.fromarray(compressed_arr)

        # ensure same height
        if orig_img.size[1] != comp_img.size[1]:
            comp_img = comp_img.resize(orig_img.size)

        # create new image wide enough for both
        new_width = orig_img.width + comp_img.width
        new_img = Image.new('RGB', (new_width, orig_img.height))
        new_img.paste(orig_img, (0, 0))
        new_img.paste(comp_img, (orig_img.width, 0))

        new_img.save(full_path)
        print(f"Saved comparison image: {full_path}")
        return full_path
    except Exception as e:
        print(f"ERROR creating side-by-side image: {e}")
        return None


def save_histograms(original_arr, compressed_arr, k_value, original_path, out_dir="output"):
    """
    Save color histograms for original and compressed images.
    Requires matplotlib; if not available, skip gracefully.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping histograms. To enable, pip install matplotlib")
        return None

    base_name = os.path.splitext(os.path.basename(original_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{base_name}_hist_k{k_value}.png"
    full_path = os.path.join(out_dir, filename)

    try:
        orig = original_arr.reshape(-1, 3)
        comp = compressed_arr.reshape(-1, 3)

        plt.figure(figsize=(10, 5))
        # plot each channel as density (simple histogram)
        channels = ('R', 'G', 'B')
        for i, ch in enumerate(channels):
            plt.subplot(1, 3, i+1)
            plt.hist(orig[:, i], bins=32, alpha=0.5, label='orig')
            plt.hist(comp[:, i], bins=32, alpha=0.5, label='comp')
            plt.title(ch)
            plt.tight_layout()
        plt.suptitle(f"Color Histograms (K={k_value})")
        plt.subplots_adjust(top=0.85)
        plt.savefig(full_path)
        plt.close()
        print(f"Saved histogram: {full_path}")
        return full_path
    except Exception as e:
        print(f"ERROR saving histogram: {e}")
        return None


# Test image generator

def generate_test_images(out_dir="output/tests", size=(256, 256)):
    """
    Generate a few test images:
      - solid color
      - gradient
      - checkerboard
      - noise
      - concentric circles (approx)
    Saves them into out_dir and returns list of paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # solid
    solid = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    solid[:, :] = [64, 128, 192]  # arbitrary color
    p = os.path.join(out_dir, "solid.png")
    Image.fromarray(solid).save(p)
    paths.append(p)

    # gradient horizontal
    grad = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        val = int((x / (size[0]-1)) * 255)
        grad[:, x] = [val, 255-val, (val//2)]
    p = os.path.join(out_dir, "gradient.png")
    Image.fromarray(grad).save(p)
    paths.append(p)

    # checkerboard
    cb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    tile = 16
    for y in range(size[1]):
        for x in range(size[0]):
            if ((x // tile) + (y // tile)) % 2 == 0:
                cb[y, x] = [255, 255, 255]
            else:
                cb[y, x] = [0, 0, 0]
    p = os.path.join(out_dir, "checkerboard.png")
    Image.fromarray(cb).save(p)
    paths.append(p)

    # noise
    noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    p = os.path.join(out_dir, "noise.png")
    Image.fromarray(noise).save(p)
    paths.append(p)

    # simple concentric-ish circles approximation
    rings = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cx, cy = size[0] // 2, size[1] // 2
    for y in range(size[1]):
        for x in range(size[0]):
            d = int(np.hypot(x-cx, y-cy))
            val = (d % 32) * 8
            rings[y, x] = [val, 255 - val, (val // 2)]
    p = os.path.join(out_dir, "rings.png")
    Image.fromarray(rings).save(p)
    paths.append(p)

    print(f"Generated {len(paths)} test images into {out_dir}")
    return paths


def save_metrics_json(metrics, base_name, out_dir="output"):
    """
    Save metrics dictionary to JSON file for later reporting.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_metrics_{timestamp}.json"
    full_path = os.path.join(out_dir, filename)
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON: {full_path}")
        return full_path
    except Exception as e:
        print(f"ERROR saving metrics JSON: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="K-Means Image Compression")
    parser.add_argument('image_path', nargs='?', help='Path to input image ')
    parser.add_argument('k_values', nargs='*', type=int, help='List of K values to try (e.g. 8 16 32)')
    parser.add_argument('--max-iterations', type=int, default=100, help='Maximum k-means iterations')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Convergence tolerance')
    parser.add_argument('--visualize', action='store_true', help='Save side-by-side comparison images and histograms (requires matplotlib)')
    parser.add_argument('--generate-tests', action='store_true', help='Generate test images into output/tests/')
    parser.add_argument('--save-metrics', action='store_true', help='Save metrics to output JSON file')
    parser.add_argument('--out-dir', default='output', help='Output directory (default: output)')
    args = parser.parse_args()

    if not args.image_path and not args.generate_tests:
        parser.print_help()
        print("\nEither provide an image path or use --generate-tests to create test images.")
        sys.exit(1)

    # default K values if none provided
    if len(args.k_values) == 0:
        k_values = [8, 16, 32]
    else:
        k_values = args.k_values

    print("=" * 70)
    print("K-MEANS IMAGE COMPRESSION")
    print("=" * 70)
    if args.image_path:
        print(f"Input: {args.image_path}")
    print(f"K values: {k_values}")
    print()

    if args.generate_tests:
        test_paths = generate_test_images(out_dir=os.path.join(args.out_dir, "tests"))
        # If generate-tests requested but no image provided, just exit after creating tests
        if not args.image_path:
            print("Generated test images. Re-run the script with one of the generated images as input.")
            sys.exit(0)

    # If image provided process it
    image_path = args.image_path

    # Validate input file
    if not os.path.exists(image_path):
        print(f"ERROR: Input file not found: {image_path}")
        sys.exit(1)

    # Process each K value
    all_metrics = {}
    for k in k_values:
        print(f"\n{'='*70}")
        print(f"PROCESSING WITH K={k}")
        print(f"{'='*70}")

        # Initialize and run K-Means
        compressor = KMeansImageCompressor(k=k, max_iterations=args.max_iterations, tolerance=args.tolerance)
        original = compressor.load_image(image_path)
        original = resize_if_needed(original, max_dimension=1024)  # ADDED: Resize for memory efficiency

        compressor.fit(original)
        compressed = compressor.compress(original)

        # Save compressed image
        output_file = save_compressed_image(compressed, k, image_path, out_dir=args.out_dir)

        # Generate visualizations if requested
        compare_file = None
        hist_file = None
        if args.visualize:
            compare_file = save_side_by_side(original, compressed, k, image_path, out_dir=args.out_dir)
            hist_file = save_histograms(original, compressed, k, image_path, out_dir=args.out_dir)

        # Calculate metrics (now with correct PSNR and size-based compression ratio)
        metrics = compressor.calculate_metrics(original, compressed, original_path=image_path, compressed_path=output_file)

        # Add file paths for reference
        metrics['compressed_path'] = output_file
        metrics['compare_path'] = compare_file
        metrics['hist_path'] = hist_file

        # Store keyed by k
        all_metrics[f'k_{k}'] = metrics

        # Display metrics
        print(f"\nResults (K={k}):")
        print(f"  Colors: {metrics['original_colors']:,} → {metrics['compressed_colors']}")
        if metrics['compressed_colors'] > 0:
            reduction = metrics['original_colors'] / metrics['compressed_colors']
            print(f"  Reduction: {reduction:.2f}x")
        else:
            print(f"  Reduction: N/A")

        print(f"  MSE: {metrics['mse']:.2f}")
        psnr_val = metrics['psnr']
        if psnr_val == float('inf'):
            print("  PSNR: inf (images identical)")
        else:
            print(f"  PSNR: {psnr_val:.2f} dB")

        if metrics['original_size'] is not None and metrics['compressed_size'] is not None:
            print(f"  Sizes: {metrics['original_size']:,} bytes -> {metrics['compressed_size']:,} bytes")
            if metrics['compression_ratio'] is not None:
                print(f"  Compression ratio (orig/comp): {metrics['compression_ratio']:.2f}x")
        else:
            print("  Sizes: N/A (original or compressed file not found)")

    # Optionally save metrics JSON
    if args.save_metrics:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_metrics_json(all_metrics, base_name, out_dir=args.out_dir)

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


"""
=============================================================================
CURRENT STATUS & ISSUES
=============================================================================

WHAT'S WORKING ✓
- Core K-Means algorithm (initialization, assignment, update, convergence)
- Image loading and conversion
- Basic compression works
- Command line interface
- Multiple K values
- Metrics calculation corrected (MSE + PSNR), and file-size comparison
- File saving improved and output directory handling added
- Visualization helpers (side-by-side, histograms)
- Test image generator
- Optional metrics JSON export
- Automatic image resizing to prevent memory errors on large images

WHAT NEEDS FIXING / FUTURE WORK ✗
1. Speed / performance:
   - Distance computation is still O(n*k) and can be slow for very large images.
   - Consider minibatch K-Means or using vectorized KD-tree / approximate nearest neighbor methods.
2. Initialization:
   - K-means++ initialization would improve convergence and avoid poor random starts.
3. File size vs. perceived compression:
   - PNG is lossless; quantization reduces unique colors but PNG encoding may not always produce proportional file-size reductions depending on image content.
   - Consider saving the compressed image as JPEG with quality parameter to get better file-size reductions (but that introduces lossy compression).
4. Visualization:
   - More polished visual output for reports/presentation could be added (matplotlib styles, labels).
5. Testing:
   - Unit tests could be added for MSE/PSNR formulas and save behavior.

=============================================================================
"""