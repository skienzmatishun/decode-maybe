# adaptive_histogram_equalization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import glob
from handle_raw_audio import read_raw_audio # Import the new function

RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")

OUTPUT_DIR = "./adaptive_histogram_results"

def compute_histogram(data):
    """Compute frequency histogram of byte data (0-255)."""
    return np.bincount(data, minlength=256)

def apply_adaptive_histogram_equalization(hist, clip_limit=0.03):
    """Apply adaptive histogram equalization to the given histogram."""
    hist_normalized = hist.astype(np.float32) / hist.sum()
    hist_equalized = exposure.equalize_adapthist(hist_normalized.reshape(16, 16), clip_limit=clip_limit).flatten()
    hist_equalized = (hist_equalized * hist.sum()).astype(np.int32)
    return hist_equalized

def plot_histogram(hist, title, output_path):
    """Plot the given histogram."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(256), hist, color="blue", alpha=0.7)
    plt.title(title)
    plt.xlabel("Byte Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read and process raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    raw_hist_equalized = apply_adaptive_histogram_equalization(raw_hist)

    # Extract features from raw audio
    raw_features = extract_audio_features(RAW_AUDIO_PATH)
    if raw_features:
        print(f"Raw Audio Features: {raw_features}")

    # Process each encrypted RAW file
    encrypted_files = glob.glob(os.path.join(ENCRYPTED_DIR, "*.raw"))
    for encrypted_file in encrypted_files:
        encrypted_data = read_raw_audio(encrypted_file)
        encrypted_hist = compute_histogram(encrypted_data)
        encrypted_hist_equalized = apply_adaptive_histogram_equalization(encrypted_hist)

        # Extract features from encrypted audio
        encrypted_features = extract_audio_features(encrypted_file)
        if encrypted_features:
            print(f"Encrypted Audio Features ({os.path.basename(encrypted_file)}): {encrypted_features}")

        # Generate plot filenames with base name
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        plot_path = os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png")
        equalized_plot_path = os.path.join(OUTPUT_DIR, f"equalized_{basename}_frequency_distribution.png")

        plot_histogram(encrypted_hist, f"Encrypted Audio ({basename}) Frequency Distribution", plot_path)
        plot_histogram(encrypted_hist_equalized, f"Equalized Encrypted Audio ({basename}) Frequency Distribution", equalized_plot_path)

    print(f"Processing results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()