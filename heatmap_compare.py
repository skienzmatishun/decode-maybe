import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import wave

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_WAVS_DIR = "./modified_raw"
OUTPUT_DIR = "./adaptive_histogram_results"

def read_raw_audio(file_path):
    """Read raw audio file as a NumPy array."""
    with open(file_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def read_wav_audio(file_path):
    """Read WAV file's audio data as a NumPy array of uint8."""
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(num_frames)
        return np.frombuffer(audio_bytes, dtype=np.uint8)

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
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    raw_hist_equalized = apply_adaptive_histogram_equalization(raw_hist)

    # Plot raw audio histograms
    plot_histogram(raw_hist, "Raw Audio Frequency Distribution", 
                   os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"))
    plot_histogram(raw_hist_equalized, "Equalized Raw Audio Frequency Distribution", 
                   os.path.join(OUTPUT_DIR, "equalized_raw_frequency_distribution.png"))

    # Process each encrypted WAV file
    encrypted_wavs_dir = ENCRYPTED_WAVS_DIR
    for filename in os.listdir(encrypted_wavs_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(encrypted_wavs_dir, filename)
            encrypted_data = read_wav_audio(filepath)
            encrypted_hist = compute_histogram(encrypted_data)
            encrypted_hist_equalized = apply_adaptive_histogram_equalization(encrypted_hist)

            # Generate plot filenames with base name
            basename = os.path.splitext(filename)[0]
            plot_path = os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png")
            equalized_plot_path = os.path.join(OUTPUT_DIR, f"equalized_{basename}_frequency_distribution.png")

            plot_histogram(encrypted_hist, f"Encrypted Audio ({basename}) Frequency Distribution", plot_path)
            plot_histogram(encrypted_hist_equalized, f"Equalized Encrypted Audio ({basename}) Frequency Distribution", equalized_plot_path)

    print(f"Processing results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()