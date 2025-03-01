#  images_modified_raw.py
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import random
import datetime
from dotenv import load_dotenv

#   Purpose: Analyze/modify raw audio via histogram equalization and bit-flipping
#   Input: left.raw (raw audio)
# Process:
# Computes byte frequency histograms
# Applies adaptive histogram equalization
# Generates files with random bit-flips (0.1%–10% of bits)
#  Output:
#  Histogram plots (adaptive_histogram_results/)
#  Bit-flipped files (bit_flip_modified/)


load_dotenv()  # Load environment variables from .env file


RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
REPORT_PATH = os.getenv("REPORT_PATH")
OUTPUT_DIR = "./adaptive_histogram_results"
BIT_FLIP_OUTPUT_DIR = "./bit_flip_modified"
NUM_BIT_FLIP_FILES = 100
MIN_BIT_FLIP_PERCENTAGE = 0.001
MAX_BIT_FLIP_PERCENTAGE = 0.1

def read_raw_audio(file_path):
    """Read raw audio file as a NumPy array."""
    with open(file_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

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

def flip_bits_in_file(input_file, output_file, flip_percentage=0.01):
    """Reads a raw audio file, flips bits randomly, and saves the modified data."""
    with open(input_file, 'rb') as f_in:
        data = bytearray(f_in.read())

    num_bytes_to_flip = int(len(data) * flip_percentage)
    
    for _ in range(num_bytes_to_flip):
        byte_index = random.randint(0, len(data) - 1)
        bit_index = random.randint(0, 7)
        data[byte_index] ^= (1 << bit_index)

    with open(output_file, 'wb') as f_out:
        f_out.write(data)

def generate_random_flipped_files(input_file, output_dir, num_files, min_percentage, max_percentage):
    """Generates multiple versions of the input file with random bit flips."""
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(num_files):
        percentage = random.uniform(min_percentage, max_percentage)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_filename}_flipped_{percentage:.4f}_{i+1:03d}_{timestamp}.raw"
        output_file = os.path.join(output_dir, output_filename)
        flip_bits_in_file(input_file, output_file, percentage)
        print(f"Generated: {output_file}")

def analyze_bit_flipped_files(bit_flip_output_dir, output_dir):
    """Analyze all bit-flipped files in the given directory."""
    for filename in os.listdir(bit_flip_output_dir):
        if filename.endswith(".raw"):
            filepath = os.path.join(bit_flip_output_dir, filename)
            flipped_data = read_raw_audio(filepath)
            flipped_hist = compute_histogram(flipped_data)
            flipped_hist_equalized = apply_adaptive_histogram_equalization(flipped_hist)

            basename = os.path.splitext(filename)[0]
            plot_path = os.path.join(output_dir, f"{basename}_frequency_distribution.png")
            equalized_plot_path = os.path.join(output_dir, f"equalized_{basename}_frequency_distribution.png")

            plot_histogram(flipped_hist, f"Bit-Flipped ({basename}) Frequency Distribution", plot_path)
            plot_histogram(flipped_hist_equalized, f"Equalized Bit-Flipped ({basename}) Frequency Distribution", equalized_plot_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BIT_FLIP_OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    raw_hist_equalized = apply_adaptive_histogram_equalization(raw_hist)

    # Plot raw audio histograms
    plot_histogram(raw_hist, "Raw Audio Frequency Distribution", os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"))
    plot_histogram(raw_hist_equalized, "Equalized Raw Audio Frequency Distribution", os.path.join(OUTPUT_DIR, "equalized_raw_frequency_distribution.png"))

    # Generate and analyze bit-flipped files
    generate_random_flipped_files(RAW_AUDIO_PATH, BIT_FLIP_OUTPUT_DIR, NUM_BIT_FLIP_FILES, MIN_BIT_FLIP_PERCENTAGE, MAX_BIT_FLIP_PERCENTAGE)
    analyze_bit_flipped_files(BIT_FLIP_OUTPUT_DIR, OUTPUT_DIR)

    # Process each encrypted RAW file
    encrypted_raw_dir = ENCRYPTED_DIR
    for filename in os.listdir(encrypted_raw_dir):
        if filename.endswith(".raw"):
            filepath = os.path.join(encrypted_raw_dir, filename)
            encrypted_data = read_raw_audio(filepath)
            encrypted_hist = compute_histogram(encrypted_data)
            encrypted_hist_equalized = apply_adaptive_histogram_equalization(encrypted_hist)

            basename = os.path.splitext(filename)[0]
            plot_path = os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png")
            equalized_plot_path = os.path.join(OUTPUT_DIR, f"equalized_{basename}_frequency_distribution.png")

            plot_histogram(encrypted_hist, f"Encrypted Audio ({basename}) Frequency Distribution", plot_path)
            plot_histogram(encrypted_hist_equalized, f"Equalized Encrypted Audio ({basename}) Frequency Distribution", equalized_plot_path)

    print(f"Processing results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()