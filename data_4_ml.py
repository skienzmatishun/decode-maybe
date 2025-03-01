import os
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import glob
from skimage import exposure
import matplotlib.pyplot as plt
from hex_frequency_dist import compute_frequency_distribution
from similarity import compute_mse, compute_ssi, compute_pearson
from bit_flip_analysis import analyze_bit_flips
from aggregate_scores import aggregate_scores
from make_charts import analyze_score_files
from dotenv import load_dotenv

# Machine Learning Feature Preparation for Audio Decryption Analysis
#     Generate a machine learning-ready dataset from decrypted audio files.
#    Extract features to evaluate decryption quality and encryption patterns.
#    Normalize data for ML model compatibility.
# Inputs
#    Raw Audio: .raw file (via RAW_AUDIO_PATH environment variable).
#    Decrypted Files: Directories containing decrypted .wav files (from DECRYPTED_DIRS).
#    Encrypted Files: Optional encrypted .raw files (from ENCRYPTED_DIR) for comparative analysis.


load_dotenv()  # Load environment variables from .env file


RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
REPORT_PATH = os.getenv("REPORT_PATH")
OUTPUT_DIR = "./ml_data"
SCORE_TYPES = ["absdiff", "chiSquared", "euclidean", "pearson", "spearman"]
HISTOGRAM_OUTPUT_DIR = "./histogram_ml_data" #add histogram output directory

def load_audio_data(filepath):
    """Load binary audio file as a NumPy array."""
    with open(filepath, "rb") as f:
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

def extract_features(raw_data, decrypted_data, encrypted_data=None):
    """Extract features from raw, decrypted and optionally encrypted audio data."""
    # Frequency Distribution
    raw_freq = compute_frequency_distribution(raw_data)
    decrypted_freq = compute_frequency_distribution(decrypted_data)
    
    # Similarity Metrics
    mse = compute_mse(raw_data, decrypted_data)
    ssi = compute_ssi(raw_data, decrypted_data)
    pearson_corr = compute_pearson(raw_data, decrypted_data)
    
    # Bit Flip Analysis
    bit_flip_counts = analyze_bit_flips(raw_data, decrypted_data)
    
    # Entropy
    raw_entropy = entropy(raw_freq + 1e-10)  # Add epsilon to avoid log(0)
    decrypted_entropy = entropy(decrypted_freq + 1e-10)

    # Histogram Features
    if encrypted_data is not None:
        encrypted_hist = compute_histogram(encrypted_data)
        encrypted_hist_equalized = apply_adaptive_histogram_equalization(encrypted_hist)
        encrypted_entropy = entropy(encrypted_hist+1e-10)
        encrypted_equalized_entropy = entropy(encrypted_hist_equalized+1e-10)

        features = {
            "mse": mse,
            "ssi": ssi,
            "pearson_corr": pearson_corr,
            "raw_entropy": raw_entropy,
            "decrypted_entropy": decrypted_entropy,
            "encrypted_entropy": encrypted_entropy,
            "encrypted_equalized_entropy": encrypted_equalized_entropy,
            **{f"bit_flip_{i}": count for i, count in enumerate(bit_flip_counts)},
            **{f"encrypted_hist_{i}": val for i, val in enumerate(encrypted_hist)},
            **{f"encrypted_eq_hist_{i}": val for i, val in enumerate(encrypted_hist_equalized)},
        }
    else:
        features = {
            "mse": mse,
            "ssi": ssi,
            "pearson_corr": pearson_corr,
            "raw_entropy": raw_entropy,
            "decrypted_entropy": decrypted_entropy,
            **{f"bit_flip_{i}": count for i, count in enumerate(bit_flip_counts)},
        }

    return features

def process_decryption_results():
    """Process all decrypted audio files and extract features."""
    raw_data = load_audio_data(RAW_AUDIO_PATH)
    results = []
    
    for filename in os.listdir(DECRYPTED_DIRS):
        if filename.startswith("decrypted_audio_") and filename.endswith(".wav"):
            print(f"Processing: {filename}")
            decrypted_data = load_audio_data(os.path.join(DECRYPTED_DIRS, filename))

            encrypted_filename = filename.replace("decrypted_audio_", "").replace(".wav", ".raw")
            encrypted_filepath = os.path.join(ENCRYPTED_DIR, encrypted_filename)

            if os.path.exists(encrypted_filepath):
                encrypted_data = load_audio_data(encrypted_filepath)
                features = extract_features(raw_data, decrypted_data, encrypted_data)
            else:
                features = extract_features(raw_data, decrypted_data)

            features["filename"] = filename
            
            # Add Decryption Scores
            scores = aggregate_scores(DECRYPTED_DIRS)
            if filename in scores:
                features.update(scores[filename])
            
            results.append(features)
    
    return results

def save_to_csv(data, output_path):
    """Save extracted features to a CSV file."""
    df = pd.DataFrame(data)
    
    # Normalize numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HISTOGRAM_OUTPUT_DIR, exist_ok=True)
    
    # Process decryption results and extract features
    print("Processing decryption results...")
    ml_data = process_decryption_results()
    
    # Save processed data to CSV
    output_file = os.path.join(OUTPUT_DIR, "ml_training_data.csv")
    save_to_csv(ml_data, output_file)
    
    print("Machine learning data generation complete.")

if __name__ == "__main__":
    main()