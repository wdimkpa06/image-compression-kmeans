import numpy as np
from PIL import Image
import sys

def initialize_centroids(pixels, k):
    """Randomly pick k pixels as starting centroids."""
    indices = np.random.choice(len(pixels), size=k, replace=False)
    return pixels[indices]

def assign_pixels_to_centroids(pixels, centroids):
    """Return index of nearest centroid for each pixel."""
    # Distance: squared Euclidean
    distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(pixels, labels, k):
    """Recalculate centroids as the mean of assigned pixels."""
    new_centroids = np.zeros((k, 3))
    for i in range(k):
        cluster = pixels[labels == i]
        if len(cluster) == 0:
            new_centroids[i] = pixels[np.random.randint(0, len(pixels))]
        else:
            new_centroids[i] = cluster.mean(axis=0)
    return new_centroids

def kmeans(pixels, k, max_iters=20):
    centroids = initialize_centroids(pixels, k)

    for _ in range(max_iters):
        labels = assign_pixels_to_centroids(pixels, centroids)
        new_centroids = update_centroids(pixels, labels, k)

        if np.allclose(centroids, new_centroids, atol=1e-3):
            break
        
        centroids = new_centroids

    return centroids, labels

def compress_image(input_path, output_path, k):
    image = Image.open(input_path)
    img_arr = np.array(image)
    
    # Flatten pixels (H×W×3 → (H*W)×3)
    pixels = img_arr.reshape(-1, 3).astype(float)

    # K-Means
    centroids, labels = kmeans(pixels, k)

    # Replace each pixel
    compressed_pixels = centroids[labels].reshape(img_arr.shape).astype(np.uint8)

    compressed_image = Image.fromarray(compressed_pixels)
    compressed_image.save(output_path)

    print(f"Saved compressed image: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compress.py input.png output.png K")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    k = int(sys.argv[3])

    compress_image(input_path, output_path, k)
