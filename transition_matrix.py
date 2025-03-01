import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import glob

# Constants
# Add at the top of your script
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Update your configuration loading
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
REPORT_PATH = os.getenv("REPORT_PATH")
OUTPUT_DIR = "./pattern_validation_results"

def read_raw_audio(file_path):
    """Read raw audio file as a NumPy array."""
    with open(file_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def compute_histogram(data):
    """Compute frequency histogram of byte data (0-255)."""
    return np.bincount(data, minlength=256)

def cosine_similarity(hist1, hist2):
    """Compute cosine similarity between two histograms."""
    return 1 - cosine(hist1, hist2)

def pearson_correlation(hist1, hist2):
    """Compute Pearson correlation coefficient between two histograms."""
    corr, _ = pearsonr(hist1, hist2)
    return corr

def validate_decryption(raw_hist, decrypted_hist):
    """
    Validate decryption by comparing histograms using cosine similarity and Pearson correlation.
    Returns a dictionary with similarity metrics.
    """
    metrics = {
        "cosine_similarity": cosine_similarity(raw_hist, decrypted_hist),
        "pearson_correlation": pearson_correlation(raw_hist, decrypted_hist)
    }
    return metrics

def analyze_byte_transformation_patterns(raw_data, encrypted_data):
    """
    Analyze byte transformation patterns by computing transition probabilities.
    Returns a 256x256 matrix where M[i][j] represents the probability of byte i transforming to byte j.
    """
    min_length = min(len(raw_data), len(encrypted_data))
    transition_matrix = np.zeros((256, 256), dtype=np.float32)

    for i in range(min_length):
        raw_byte = raw_data[i]
        enc_byte = encrypted_data[i]
        transition_matrix[raw_byte][enc_byte] += 1

    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix /= row_sums

    return transition_matrix

def plot_transition_heatmap(matrix, title, output_path):
    """Plot a heatmap of the byte transition matrix."""
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Probability")
    plt.title(title)
    plt.xlabel("Encrypted Byte")
    plt.ylabel("Raw Byte")
    plt.xticks(np.arange(0, 256, 32))
    plt.yticks(np.arange(0, 256, 32))
    plt.savefig(output_path)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)

    # Process each encrypted RAW file
    encrypted_files = glob.glob(os.path.join(ENCRYPTED_DIR, "*.raw"))
    if not encrypted_files:
        print("No encrypted files found!")
        return

    results = []

    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        encrypted_data = read_raw_audio(encrypted_file)
        encrypted_hist = compute_histogram(encrypted_data)

        # Analyze byte transformation patterns
        transition_matrix = analyze_byte_transformation_patterns(raw_data, encrypted_data)
        heatmap_path = os.path.join(OUTPUT_DIR, f"{basename}_transition_heatmap.png")
        plot_transition_heatmap(transition_matrix, f"Byte Transition Heatmap ({basename})", heatmap_path)

        # Validate decrypted files from each directory in DECRYPTED_DIRS
        for decrypted_dir in DECRYPTED_DIRS:
            decrypted_files = glob.glob(os.path.join(decrypted_dir, f"decrypted_{basename}*.raw"))
            for decrypted_file in decrypted_files:
                decrypted_data = read_raw_audio(decrypted_file)
                decrypted_hist = compute_histogram(decrypted_data)

                # Compute validation metrics
                metrics = validate_decryption(raw_hist, decrypted_hist)
                results.append({
                    "encrypted_file": encrypted_file,
                    "decrypted_file": decrypted_file,
                    "cosine_similarity": metrics["cosine_similarity"],
                    "pearson_correlation": metrics["pearson_correlation"]
                })

    # Save results to a report
    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w") as report:
        report.write("Decryption Validation Report\n")
        report.write("=" * 50 + "\n\n")

        for result in results:
            report.write(f"Encrypted File: {result['encrypted_file']}\n")
            report.write(f"Decrypted File: {result['decrypted_file']}\n")
            report.write(f"Cosine Similarity: {result['cosine_similarity']:.4f}\n")
            report.write(f"Pearson Correlation: {result['pearson_correlation']:.4f}\n")
            report.write("-" * 50 + "\n")

    print(f"Validation results saved to {report_path}")

if __name__ == "__main__":
    main()