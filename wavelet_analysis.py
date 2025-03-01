import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from handle_raw_audio import read_raw_audio

# Constants
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./wavelet_analysis"

def compute_wavelet_transform(data, wavelet='db4', level=5):
    """Compute wavelet transform on the given data."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def plot_wavelet_coefficients(coeffs, title, output_path):
    """Plot the wavelet coefficients."""
    plt.figure(figsize=(12, 6))
    for i, coeff in enumerate(coeffs):
        plt.subplot(len(coeffs), 1, i + 1)
        plt.plot(coeff)
        plt.title(f"Level {i} Coefficients" if i > 0 else "Approximation Coefficients")
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig(output_path)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_coeffs = compute_wavelet_transform(raw_data)
    plot_wavelet_coefficients(raw_coeffs, "Raw Audio Wavelet Coefficients", os.path.join(OUTPUT_DIR, "raw_wavelet_coefficients.png"))

    # Process each encrypted RAW file
    encrypted_files = [f for f in os.listdir(ENCRYPTED_DIR) if f.endswith(".raw")]
    if not encrypted_files:
        print("No encrypted files found!")
        return

    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        filepath = os.path.join(ENCRYPTED_DIR, encrypted_file)
        encrypted_data = read_raw_audio(filepath)
        encrypted_coeffs = compute_wavelet_transform(encrypted_data)
        plot_wavelet_coefficients(encrypted_coeffs, f"Encrypted Audio ({basename}) Wavelet Coefficients", os.path.join(OUTPUT_DIR, f"{basename}_wavelet_coefficients.png"))

        # Compare wavelet coefficients between raw and encrypted audio
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(raw_coeffs[0])
        plt.title("Raw Audio Approximation Coefficients")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(encrypted_coeffs[0])
        plt.title(f"Encrypted Audio ({basename}) Approximation Coefficients")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{basename}_approximation_coefficients_comparison.png"))
        plt.close()

        # Compare detail coefficients
        for i in range(1, len(raw_coeffs)):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(raw_coeffs[i])
            plt.title(f"Raw Audio Level {i} Detail Coefficients")

            plt.subplot(1, 2, 2)
            plt.plot(encrypted_coeffs[i])
            plt.title(f"Encrypted Audio ({basename}) Level {i} Detail Coefficients")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{basename}_level_{i}_detail_coefficients_comparison.png"))
            plt.close()

    print(f"Wavelet analysis results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()