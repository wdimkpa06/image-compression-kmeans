# Image Compression using K-Means Color Quantization
CSE 4081 – Term Project  
Group Members: Woroma Dimkpa, 

## Project Overview
This project implements lossy image compression by reducing the number of unique colors in an image using K-Means clustering.

## Folder Structure
src/          → Python source code  
images/       → Input images for testing  
results/      → Output compressed images  
requirements.txt  → Libraries needed  

## Running the Compressor
Activate the virtual environment, then run:

python src/compress.py input.png output.png K

Example:

python src/compress.py images/sample1.png results/sample1_k16.png 16

## Metrics
We compute:
- Compression Ratio
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)

## Notes
This project implements K-Means **from scratch**, following CSE 4081 requirements.
