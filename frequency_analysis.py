import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from handle_raw_audio import read_raw_audio

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./frequency_domain_analysis"

def compute_fft(data, sample_rate):
    """Compute the FFT of the audio data."""
    n = len(data)
    yf = fft(data)
    xf = np.fft.fftfreq(n, 1/sample_rate)
    return xf, np.abs(yf)

def plot_fft(xf, yf, title, output_path, sample_rate):
    """Plot the FFT frequency spectrum."""
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.xlim(0, sample_rate/2)  # Only plot up to Nyquist frequency
    plt.savefig(output_path)
    plt.close()

def find_frequency_peaks(xf, yf, height_factor=0.5):
    """Find peaks in the frequency spectrum."""
    height = height_factor * np.max(yf)
    peaks, _ = find_peaks(yf, height=height)
    return xf[peaks], yf[peaks]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH, sr=16000)
    sample_rate = 16000

    # Compute and plot FFT for raw audio
    xf_raw, yf_raw = compute_fft(raw_data, sample_rate)
    plot_fft(xf_raw, yf_raw, "Raw Audio Frequency Spectrum", os.path.join(OUTPUT_DIR, "raw_fft.png"), sample_rate)

    # Find peaks in raw audio frequency spectrum
    raw_peaks, raw_peak_magnitudes = find_frequency_peaks(xf_raw, yf_raw)
    print(f"Raw Audio Peaks: {raw_peaks}")

    # Process each encrypted RAW file
    encrypted_files = [f for f in os.listdir(ENCRYPTED_RAW_DIR) if f.endswith(".raw")]
    if not encrypted_files:
        print("No encrypted files found!")
        return

    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        filepath = os.path.join(ENCRYPTED_RAW_DIR, encrypted_file)
        encrypted_data = read_raw_audio(filepath, sr=16000)

        # Compute and plot FFT for encrypted audio
        xf_enc, yf_enc = compute_fft(encrypted_data, sample_rate)
        plot_fft(xf_enc, yf_enc, f"Encrypted Audio ({basename}) Frequency Spectrum", os.path.join(OUTPUT_DIR, f"{basename}_fft.png"), sample_rate)

        # Find peaks in encrypted audio frequency spectrum
        enc_peaks, enc_peak_magnitudes = find_frequency_peaks(xf_enc, yf_enc)
        print(f"Encrypted Audio ({basename}) Peaks: {enc_peaks}")

        # Compare peaks between raw and encrypted audio
        plt.figure(figsize=(12, 6))
        plt.plot(xf_raw, yf_raw, label="Raw Audio", alpha=0.7)
        plt.plot(xf_enc, yf_enc, label=f"Encrypted Audio ({basename})", alpha=0.7)
        plt.scatter(raw_peaks, raw_peak_magnitudes, color='red', label="Raw Peaks")
        plt.scatter(enc_peaks, enc_peak_magnitudes, color='green', label="Encrypted Peaks")
        plt.title(f"Comparison of Raw and Encrypted Audio ({basename}) Frequency Spectra")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.xlim(0, sample_rate/2)  # Only plot up to Nyquist frequency
        plt.savefig(os.path.join(OUTPUT_DIR, f"{basename}_fft_comparison.png"))
        plt.close()

    print(f"Frequency domain analysis results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()