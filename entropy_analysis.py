# entropy_analysis.py
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from handle_raw_audio import read_raw_audio  # Import from handle_raw_audio

# Constants
ENCRYPTED_RAW_DIR = "./modified_raw"  # Directory with encrypted raw files
OUTPUT_DIR = "./entropy_analysis_results"
WINDOW_SIZES = [100, 500, 1000]  # Different window sizes for entropy calculation
STEP_SIZES = [50, 250, 500]  # Different step sizes for sliding window

def compute_histogram(data):
    """Compute frequency histogram of byte data (0-255)."""
    return np.bincount(data, minlength=256)

def calculate_entropy(data):
    """Calculate entropy of the given byte data."""
    hist = np.bincount(data, minlength=256)
    probs = hist / len(data)
    probs = probs[probs > 0]  # Remove zero probabilities
    return scipy.stats.entropy(probs)

def calculate_sliding_window_entropy(data, window_size, step_size):
    """Calculate entropy with a sliding window."""
    entropies = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        entropy = calculate_entropy(window)
        entropies.append(entropy)
    return entropies

def plot_sliding_window_entropy(entropies, window_size, step_size, filename, output_dir):
    """Plot the sliding window entropy."""
    plt.figure(figsize=(12, 6))
    plt.plot(entropies)
    plt.title(f"Sliding Window Entropy (Window: {window_size}, Step: {step_size}) - {filename}")
    plt.xlabel("Window Position")
    plt.ylabel("Entropy")
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"entropy_{filename}_window_{window_size}_step_{step_size}.png"))
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(ENCRYPTED_RAW_DIR):
        if filename.endswith(".raw"):
            filepath = os.path.join(ENCRYPTED_RAW_DIR, filename)
            data = read_raw_audio(filepath)

            for window_size in WINDOW_SIZES:
                for step_size in STEP_SIZES:
                    entropies = calculate_sliding_window_entropy(data, window_size, step_size)
                    plot_sliding_window_entropy(entropies, window_size, step_size, os.path.splitext(filename)[0], OUTPUT_DIR)

    print(f"Entropy analysis results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()