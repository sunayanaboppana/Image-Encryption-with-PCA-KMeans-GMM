# Image Encryption and Reconstruction using PCA, KMeans, and GMM

This project implements image encryption and decryption using three different machine learning approaches: Principal Component Analysis (PCA), KMeans Clustering, and Gaussian Mixture Models (GMM). It also evaluates the quality of decrypted images using MSE (Mean Squared Error) and SSIM (Structural Similarity Index).

## Overview

The script allows you to:
- Load and preprocess grayscale images.
- Encrypt images using PCA, KMeans, and GMM.
- Decrypt and reconstruct the original images.
- Evaluate each method based on MSE and SSIM.
- Visualize original, encrypted, and decrypted images along with histograms.

## Features

- **PCA-based Encryption**: Reduces image data using dimensionality reduction.
- **KMeans-based Encryption**: Compresses images by clustering pixel intensities.
- **GMM-based Encryption**: Uses probabilistic clustering for compression.
- **Performance Metrics**: Compares decrypted images using MSE and SSIM.
- **Visualization**: Shows all encrypted/decrypted images and histograms using Matplotlib.

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- scikit-learn
- scikit-image
- Pillow
- Matplotlib
- Tkinter (comes pre-installed with most Python distributions)

Install dependencies via pip:

```bash
pip install numpy opencv-python scikit-learn scikit-image Pillow matplotlib
