import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import random
import datetime
from handle_raw_audio import read_raw_audio

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./adaptive_histogram_results"
BIT_FLIP_OUTPUT_DIR = "./bit_flip_modified"
NUM_BIT_FLIP_FILES = 100
MIN_BIT_FLIP_PERCENTAGE = 0.001
MAX_BIT_FLIP_PERCENTAGE = 0.1


def flip_bits_in_file(input_file, output_file, flip_percentage=0.01):
    """Reads a raw audio file, flips bits randomly, and saves the modified data."""
    with open(input_file, 'rb') as f_in:
        data = bytearray(f_in.read())

    num_bytes_to_flip = int(len(data) * flip_percentage)

    for _ in range(num_bytes_to_flip):
        byte_index = random.randint(0, len(data) - 1)
        bit_index = random.randint(0, 7)  # 8 bits in a byte

        # Flip the bit
        data[byte_index] ^= (1 << bit_index)

    with open(output_file, 'wb') as f_out:
        f_out.write(data)


def generate_random_flipped_files(input_file, output_dir, num_files, min_percentage, max_percentage):
    """Generates multiple versions of the input file with random bit flips, using a random percentage within a given range."""
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(num_files):
        percentage = random.uniform(min_percentage, max_percentage)  # Generate random percentage
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
            equalized_plot_path = os.path.join(
                output_dir, f"equalized_{basename}_frequency_distribution.png"
            )

            plot_histogram(flipped_hist, f"Bit-Flipped ({basename}) Frequency Distribution", plot_path)
            plot_histogram(
                flipped_hist_equalized, f"Equalized Bit-Flipped ({basename}) Frequency Distribution", equalized_plot_path
            )


def compute_histogram(data):
    # Tentative fix: Convert float32 data to integers in 0-255 range (APPROXIMATION - review logic!)
    # Scale float data from -1 to 1 to 0 to 255 range (rough approximation)
    int_data = ((data + 1.0) * 127.5).astype(int)  # Scale and shift to 0-255, then convert to int
    # Clip values to ensure they are within 0-255 (just in case of rounding issues)
    int_data = np.clip(int_data, 0, 255)
    return np.bincount(int_data, minlength=256)


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


def calculate_chi_squared(hist1, hist2):
    """Compute Chi-squared distance between two histograms."""
    smooth = 1e-12
    return np.sum((hist1 - hist2) ** 2 / (hist2 + smooth))


def brute_force_xor_decrypt(raw_data_path, encrypted_data_path, output_dir):
    """Attempt to decrypt using single-byte XOR brute force."""
    try:
        with open(raw_data_path, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        with open(encrypted_data_path, 'rb') as f:
            encrypted_data = np.frombuffer(f.read(), dtype=np.uint8)

        min_length = min(len(raw_data), len(encrypted_data))
        raw_data = raw_data[:min_length]
        encrypted_data = encrypted_data[:min_length]

        raw_hist = compute_histogram(raw_data)

        best_key = None
        best_distance = float('inf')

        for key in range(256):
            decrypted_data = np.bitwise_xor(encrypted_data, key)
            decrypted_hist = compute_histogram(decrypted_data)
            distance = calculate_chi_squared(decrypted_hist, raw_hist)

            if distance < best_distance:
                best_distance = distance
                best_key = key

        print(f"\nBest XOR Key: 0x{best_key:02X} (Chi² Distance: {best_distance:.4f})")

        decrypted = np.bitwise_xor(encrypted_data, best_key)
        output_file = os.path.join(
            output_dir,
            f"decrypted_{os.path.basename(raw_data_path)}_{os.path.basename(encrypted_data_path)}.raw",
        )
        with open(output_file, "wb") as f:
            f.write(decrypted.tobytes())

        print(f"Decrypted data saved to '{output_file}'")
        return output_file

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BIT_FLIP_OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    raw_hist_equalized = apply_adaptive_histogram_equalization(raw_hist)

    # Plot raw audio histograms
    plot_histogram(
        raw_hist,
        "Raw Audio Frequency Distribution",
        os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"),
    )
    plot_histogram(
        raw_hist_equalized,
        "Equalized Raw Audio Frequency Distribution",
        os.path.join(OUTPUT_DIR, "equalized_raw_frequency_distribution.png"),
    )

    # Generate and analyze bit-flipped files
    generate_random_flipped_files(RAW_AUDIO_PATH, BIT_FLIP_OUTPUT_DIR, NUM_BIT_FLIP_FILES, MIN_BIT_FLIP_PERCENTAGE, MAX_BIT_FLIP_PERCENTAGE)
    analyze_bit_flipped_files(BIT_FLIP_OUTPUT_DIR, OUTPUT_DIR)

    # Process each encrypted RAW file
    encrypted_raw_dir = ENCRYPTED_RAW_DIR
    for filename in os.listdir(encrypted_raw_dir):
        if filename.endswith(".raw"):
            filepath = os.path.join(encrypted_raw_dir, filename)
            encrypted_data = read_raw_audio(filepath)
            encrypted_hist = compute_histogram(encrypted_data)
            encrypted_hist_equalized = apply_adaptive_histogram_equalization(encrypted_hist)

            basename = os.path.splitext(filename)[0]
            plot_path = os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png")
            equalized_plot_path = os.path.join(OUTPUT_DIR, f"equalized_{basename}_frequency_distribution.png")

            plot_histogram(
                encrypted_hist,
                f"Encrypted Audio ({basename}) Frequency Distribution",
                plot_path,
            )
            plot_histogram(
                encrypted_hist_equalized,
                f"Equalized Encrypted Audio ({basename}) Frequency Distribution",
                equalized_plot_path,
            )

            # Brute-force XOR decryption and analysis
            decrypted_file = brute_force_xor_decrypt(RAW_AUDIO_PATH, filepath, OUTPUT_DIR)
            if decrypted_file:
                decrypted_data = read_raw_audio(decrypted_file)
                decrypted_hist = compute_histogram(decrypted_data)
                decrypted_hist_equalized = apply_adaptive_histogram_equalization(decrypted_hist)

                basename = os.path.splitext(os.path.basename(decrypted_file))[0]
                plot_path = os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png")
                equalized_plot_path = os.path.join(
                    OUTPUT_DIR, f"equalized_{basename}_frequency_distribution.png"
                )

                plot_histogram(
                    decrypted_hist,
                    f"Decrypted ({basename}) Frequency Distribution",
                    plot_path,
                )
                plot_histogram(
                    decrypted_hist_equalized,
                    f"Equalized Decrypted ({basename}) Frequency Distribution",
                    equalized_plot_path,
                )

    print(f"Processing results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()