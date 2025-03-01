import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from handle_raw_audio import read_raw_audio

# Constants
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./pca_analysis"
WINDOW_SIZE = 100  # Size of the sliding window
STEP_SIZE = 50     # Step size for the sliding window

def compute_features_from_sliding_window(data, window_size, step_size):
    """Compute features from a sliding window over the data."""
    features = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        features.append(window)
    return np.array(features)

def compute_pca(features, n_components=2):
    """Compute PCA on the given features."""
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    return principal_components, pca.explained_variance_ratio_

def plot_pca(principal_components, title, output_path):
    """Plot the PCA results."""
    plt.figure(figsize=(12, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_features = compute_features_from_sliding_window(raw_data, WINDOW_SIZE, STEP_SIZE)
    if raw_features.shape[0] < 2:
        print("Not enough data points to perform PCA on raw audio.")
        return

    raw_pca, raw_variance_ratio = compute_pca(raw_features)
    plot_pca(raw_pca, "Raw Audio PCA", os.path.join(OUTPUT_DIR, "raw_pca.png"))
    print(f"Raw Audio Explained Variance Ratio: {raw_variance_ratio}")

    # Process each encrypted RAW file
    encrypted_files = [f for f in os.listdir(ENCRYPTED_DIR) if f.endswith(".raw")]
    if not encrypted_files:
        print("No encrypted files found!")
        return

    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        filepath = os.path.join(ENCRYPTED_DIR, encrypted_file)
        encrypted_data = read_raw_audio(filepath)
        encrypted_features = compute_features_from_sliding_window(encrypted_data, WINDOW_SIZE, STEP_SIZE)
        if encrypted_features.shape[0] < 2:
            print(f"Not enough data points to perform PCA on encrypted audio {basename}.")
            continue

        encrypted_pca, encrypted_variance_ratio = compute_pca(encrypted_features)
        plot_pca(encrypted_pca, f"Encrypted Audio ({basename}) PCA", os.path.join(OUTPUT_DIR, f"{basename}_pca.png"))
        print(f"Encrypted Audio ({basename}) Explained Variance Ratio: {encrypted_variance_ratio}")

        # Compare PCA components between raw and encrypted audio
        plt.figure(figsize=(12, 6))
        plt.scatter(raw_pca[:, 0], raw_pca[:, 1], alpha=0.5, label="Raw Audio")
        plt.scatter(encrypted_pca[:, 0], encrypted_pca[:, 1], alpha=0.5, label=f"Encrypted Audio ({basename})")
        plt.title(f"Comparison of Raw and Encrypted Audio ({basename}) PCA Components")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{basename}_pca_comparison.png"))
        plt.close()

    print(f"PCA analysis results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()