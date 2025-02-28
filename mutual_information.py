import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from handle_raw_audio import read_raw_audio

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./mutual_information_analysis"

def compute_histogram(data):
    """Compute frequency histogram of byte data (0-255)."""
    return np.bincount(data, minlength=256)

def plot_histogram(hist, title, output_path):
    """Plot the given histogram with hex values on x-axis."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), hist, color="blue", alpha=0.7)
    plt.title(title)
    
    # Add hex labels at key positions
    hex_positions = list(range(0, 256, 16))
    hex_labels = [f"{i:02X}" for i in hex_positions]
    plt.xticks(hex_positions, hex_labels)
    
    plt.xlabel("Byte Value (Hex)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def calculate_mutual_information(data1, data2):
    """Calculate mutual information between two datasets."""
    hist1 = compute_histogram(data1)
    hist2 = compute_histogram(data2)
    joint_hist = np.histogram2d(data1, data2, bins=256, range=[[0, 256], [0, 256]])[0]
    
    # Normalize histograms
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    joint_hist = joint_hist / joint_hist.sum()
    
    # Calculate marginal entropies
    H1 = entropy(hist1)
    H2 = entropy(hist2)
    
    # Calculate joint entropy
    H12 = entropy(joint_hist.flatten())
    
    # Calculate mutual information
    MI = H1 + H2 - H12
    return MI

def plot_mutual_information(mutual_info, title, output_path):
    """Plot mutual information."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), mutual_info, color="green", alpha=0.7)
    plt.title(title)
    plt.xlabel("Byte Value")
    plt.ylabel("Mutual Information")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    # Convert float32 data to uint8 (scale from -1.0 to 1.0 to 0 to 255)
    raw_data_uint8 = ((raw_data + 1.0) * 127.5).astype(np.uint8)
    raw_hist = compute_histogram(raw_data_uint8)
    plot_histogram(raw_hist, "Raw Audio Frequency Distribution", os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"))

    # Process each encrypted RAW file
    encrypted_files = [f for f in os.listdir(ENCRYPTED_RAW_DIR) if f.endswith(".raw")]
    if not encrypted_files:
        print("No encrypted files found!")
        return

    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        filepath = os.path.join(ENCRYPTED_RAW_DIR, encrypted_file)
        encrypted_data = read_raw_audio(filepath)
        # Convert float32 data to uint8 (scale from -1.0 to 1.0 to 0 to 255)
        encrypted_data_uint8 = ((encrypted_data + 1.0) * 127.5).astype(np.uint8)
        encrypted_hist = compute_histogram(encrypted_data_uint8)
        plot_histogram(encrypted_hist, f"Encrypted Audio ({basename}) Frequency Distribution", os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png"))

        # Calculate mutual information
        mutual_info = calculate_mutual_information(raw_data_uint8, encrypted_data_uint8)
        plot_mutual_information(mutual_info, f"Mutual Information (Raw vs Encrypted {basename})", os.path.join(OUTPUT_DIR, f"{basename}_mutual_information.png"))
        print(f"Mutual Information (Raw vs Encrypted {basename}): {mutual_info:.4f}")

        # Compare mutual information with random data
        random_data = np.random.randint(0, 256, size=len(raw_data_uint8))
        mutual_info_random = calculate_mutual_information(raw_data_uint8, random_data)
        plot_mutual_information(mutual_info_random, f"Mutual Information (Raw vs Random Data)", os.path.join(OUTPUT_DIR, f"{basename}_mutual_information_random.png"))
        print(f"Mutual Information (Raw vs Random Data): {mutual_info_random:.4f}")

    print(f"Mutual information analysis results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()