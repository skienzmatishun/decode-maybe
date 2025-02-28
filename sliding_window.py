import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import wave

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_WAVS_DIR = "./modified_raw"
ENTROPY_ANALYSIS_RESULTS_DIR = "./entropy_analysis_results"

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
    os.makedirs(output_dir, exist_ok=True) #added this line to make sure the directories are created.
    plt.savefig(os.path.join(output_dir, f"entropy_{filename}_window_{window_size}_step_{step_size}.png"))
    plt.close()

def main():
    os.makedirs(ENTROPY_ANALYSIS_RESULTS_DIR, exist_ok=True)

    # Calculate entropy for the raw audio
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_entropy = calculate_entropy(raw_data)
    print(f"Entropy of raw audio: {raw_entropy:.4f}")

    # Calculate entropy for each encrypted WAV file
    for filename in os.listdir(ENCRYPTED_WAVS_DIR):
        if filename.endswith(".wav"):
            filepath = os.path.join(ENCRYPTED_WAVS_DIR, filename)
            encrypted_data = read_wav_audio(filepath)
            encrypted_entropy = calculate_entropy(encrypted_data)
            print(f"Entropy of {filename}: {encrypted_entropy:.4f}")

            # Sliding window entropy analysis
            window_sizes = [100, 500, 1000]
            step_sizes = [50, 250, 500]

            for window_size in window_sizes:
                for step_size in step_sizes:
                    entropies = calculate_sliding_window_entropy(encrypted_data, window_size, step_size)
                    plot_sliding_window_entropy(entropies, window_size, step_size, os.path.splitext(filename)[0], ENTROPY_ANALYSIS_RESULTS_DIR)

    print(f"Entropy analysis results saved to {ENTROPY_ANALYSIS_RESULTS_DIR}")

if __name__ == "__main__":
    main()