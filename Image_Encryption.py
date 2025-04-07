import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Function to load image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    image = cv2.resize(image, (100, 100))  # Resize for consistent size
    return image

# Function to flatten the image
def flatten_image(image):
    return image.flatten().astype(float) / 255.0

# Function to perform PCA
def pca_encryption(image, n_components=1):
    # Flatten the image
    image_flattened = flatten_image(image)
    
    # Ensure n_components is not greater than the number of features (10,000 for 100x100 image)
    n_components = min(n_components, len(image_flattened))  # Ensure the number of components doesn't exceed the number of features
    
    pca = PCA(n_components=n_components)
    encrypted_image = pca.fit_transform([image_flattened])  # Perform PCA on the flattened image
    return pca, encrypted_image

# Function to decrypt using PCA
def pca_decryption(pca, encrypted_image):
    # Decrypt the image by transforming it back
    decrypted_image = pca.inverse_transform(encrypted_image)
    decrypted_image = decrypted_image.reshape(100, 100)
    return decrypted_image

# Function to perform KMeans encryption
def kmeans_encryption(image, n_clusters=16):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    image_flattened = flatten_image(image)
    image_flattened = image_flattened.reshape(-1, 1)  # Reshaping for clustering
    kmeans.fit(image_flattened)
    encrypted_image = kmeans.labels_
    encrypted_image = encrypted_image.reshape(100, 100)  # Reshape to image size
    return kmeans, encrypted_image

# Function to decrypt using KMeans
def kmeans_decryption(kmeans, encrypted_image):
    encrypted_image = encrypted_image.flatten().astype(int)  # Ensure labels are integers
    decrypted_image = kmeans.cluster_centers_[encrypted_image].reshape(100, 100)
    return decrypted_image

# Function to perform GMM encryption
def gmm_encryption(image, n_components=16):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    image_flattened = flatten_image(image)
    image_flattened = image_flattened.reshape(-1, 1)  # Reshaping for GMM
    gmm.fit(image_flattened)
    encrypted_image = gmm.predict(image_flattened)
    encrypted_image = encrypted_image.reshape(100, 100)  # Reshape to image size
    return gmm, encrypted_image

# Function to decrypt using GMM
def gmm_decryption(gmm, encrypted_image):
    labels = encrypted_image.flatten()
    decrypted_image = gmm.means_[labels].reshape(100, 100)
    return decrypted_image

# Function to calculate MSE and SSIM for comparison
def compare_images(original, decrypted):
    mse_value = mean_squared_error(original, decrypted)
    data_range = original.max() - original.min()  # Calculate the data range
    ssim_value = ssim(original, decrypted, data_range=data_range)
    return mse_value, ssim_value

# Main function to run the encryption comparison
def run_encryption_comparison(image_path):
    # Load and preprocess the image
    original_image = load_image(image_path)
    
    # PCA Encryption and Decryption
    pca, pca_encrypted = pca_encryption(original_image)
    pca_decrypted = pca_decryption(pca, pca_encrypted)
    
    # KMeans Encryption and Decryption
    kmeans, kmeans_encrypted = kmeans_encryption(original_image)
    kmeans_decrypted = kmeans_decryption(kmeans, kmeans_encrypted)
    
    # GMM Encryption and Decryption
    gmm, gmm_encrypted = gmm_encryption(original_image)
    gmm_decrypted = gmm_decryption(gmm, gmm_encrypted)
    
    # Calculate MSE and SSIM for each model
    pca_mse, pca_ssim = compare_images(original_image, pca_decrypted)
    kmeans_mse, kmeans_ssim = compare_images(original_image, kmeans_decrypted)
    gmm_mse, gmm_ssim = compare_images(original_image, gmm_decrypted)
    
    print(f"PCA MSE: {pca_mse}, SSIM: {pca_ssim}")
    print(f"KMeans MSE: {kmeans_mse}, SSIM: {kmeans_ssim}")
    print(f"GMM MSE: {gmm_mse}, SSIM: {gmm_ssim}")
    
    # Display the results
    plt.figure(figsize=(14, 12))
    
    # Original Image
    plt.subplot(4, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    # PCA Decrypted Image
    plt.subplot(4, 3, 2)
    plt.imshow(pca_decrypted, cmap='gray')
    plt.title(f'PCA Decrypted - SSIM: {pca_ssim:.4f}')

    # PCA Encrypted Histogram
    plt.subplot(4, 3, 3)
    plt.hist(pca_encrypted.flatten(), bins=50, color='b', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('PCA Encrypted Histogram')

    # PCA Decrypted Histogram
    plt.subplot(4, 3, 4)
    plt.hist(pca_decrypted.flatten(), bins=50, color='r', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('PCA Decrypted Histogram')

    # KMeans Encrypted Image
    plt.subplot(4, 3, 5)
    plt.imshow(kmeans_encrypted, cmap='gray')
    plt.title(f'KMeans Encrypted')
    
    # KMeans Decrypted Image
    plt.subplot(4, 3, 6)
    plt.imshow(kmeans_decrypted, cmap='gray')
    plt.title(f'KMeans Decrypted - SSIM: {kmeans_ssim:.4f}')
    
    # KMeans Encrypted Histogram
    plt.subplot(4, 3, 7)
    plt.hist(kmeans_encrypted.flatten(), bins=50, color='g', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('KMeans Encrypted Histogram')

    # KMeans Decrypted Histogram
    plt.subplot(4, 3, 8)
    plt.hist(kmeans_decrypted.flatten(), bins=50, color='y', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('KMeans Decrypted Histogram')

    # GMM Encrypted Image
    plt.subplot(4, 3, 9)
    plt.imshow(gmm_encrypted, cmap='gray')
    plt.title(f'GMM Encrypted')
    
    # GMM Decrypted Image
    plt.subplot(4, 3, 10)
    plt.imshow(gmm_decrypted, cmap='gray')
    plt.title(f'GMM Decrypted - SSIM: {gmm_ssim:.4f}')
    
    # GMM Decrypted Histogram
    plt.subplot(4, 3, 11)
    plt.hist(gmm_decrypted.flatten(), bins=50, color='c', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('GMM Decrypted Histogram')
    
    # GMM Encrypted Histogram
    plt.subplot(4, 3, 12)
    plt.hist(gmm_encrypted.flatten(), bins=50, color='m', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('GMM Encrypted Histogram')

    plt.tight_layout()
    plt.show()

# File Dialog to load image
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

run_encryption_comparison(image_path)
