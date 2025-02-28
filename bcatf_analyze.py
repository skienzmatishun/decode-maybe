import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import correlate, correlation_lags
from scipy.fft import fft, ifft
from skimage import exposure
import glob

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./comprehensive_analysis"

def read_raw_audio(file_path):
    """Read raw audio file as a NumPy array."""
    with open(file_path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

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

def compute_bigram_frequencies(data):
    """Computes bigram frequency distribution."""
    bigrams = [tuple(data[i:i+2]) for i in range(len(data)-1)]
    unique, counts = np.unique(bigrams, axis=0, return_counts=True)
    freq_matrix = np.zeros((256, 256), dtype=np.float32)
    
    for (a, b), count in zip(unique, counts):
        freq_matrix[a][b] = count / len(bigrams)  # Normalize to probability
    
    return freq_matrix

def plot_bigram_heatmap(matrix, title, output_path):
    """Plots a bigram frequency heatmap."""
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Probability')
    plt.title(title)
    plt.xlabel('Second Byte')
    plt.ylabel('First Byte')
    
    # Add some hex labels
    hex_positions = list(range(0, 256, 32))
    hex_labels = [f"{i:02X}" for i in hex_positions]
    plt.xticks(hex_positions, hex_labels)
    plt.yticks(hex_positions, hex_labels)
    
    plt.savefig(output_path)
    plt.close()

def compute_cross_correlation(data1, data2):
    """Compute cross-correlation between two datasets."""
    return correlate(data1, data2, mode='full')

def plot_cross_correlation(correlation, lags, title, output_path):
    """Plot cross-correlation."""
    plt.figure(figsize=(12, 6))
    plt.plot(lags, correlation)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def compute_auto_correlation(data):
    """Compute auto-correlation of a dataset."""
    return correlate(data, data, mode='full')

def plot_auto_correlation(auto_corr, lags, title, output_path):
    """Plot auto-correlation."""
    plt.figure(figsize=(12, 6))
    plt.plot(lags, auto_corr)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Auto-Correlation')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def compute_stft(data, n_fft=512, hop_length=256):
    """Compute Short-Time Fourier Transform (STFT)."""
    return librosa.stft(data.astype(np.float32), n_fft=n_fft, hop_length=hop_length)

def plot_stft(stft_matrix, title, output_path):
    """Plot STFT magnitude spectrogram."""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max), sr=22050, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def extract_audio_features(audio_file):
    """Extracts MFCCs, Chroma features, Spectral Centroid, and Spectral Bandwidth from an audio file."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return None

    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)   # Adjust n_mfcc as needed
    features['mfccs'] = np.mean(mfccs.T, axis=0)  # Average MFCCs over time

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma'] = np.mean(chroma.T, axis=0)  # Average chroma over time

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr) 
    features['spectral_centroid'] = np.mean(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth)

    return features

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    plot_histogram(raw_hist, "Raw Audio Frequency Distribution", os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"))

    # Process each encrypted RAW file
    encrypted_files = glob.glob(os.path.join(ENCRYPTED_RAW_DIR, "*.raw"))
    if not encrypted_files:
        print("No encrypted files found!")
        return

    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        encrypted_data = read_raw_audio(encrypted_file)
        encrypted_hist = compute_histogram(encrypted_data)
        plot_histogram(encrypted_hist, f"Encrypted Audio ({basename}) Frequency Distribution", os.path.join(OUTPUT_DIR, f"{basename}_frequency_distribution.png"))

        # Bigram Analysis
        raw_bigram_freq = compute_bigram_frequencies(raw_data)
        encrypted_bigram_freq = compute_bigram_frequencies(encrypted_data)
        
        plot_bigram_heatmap(raw_bigram_freq, "Raw Audio Bigram Heatmap", os.path.join(OUTPUT_DIR, f"{basename}_raw_bigram_heatmap.png"))
        plot_bigram_heatmap(encrypted_bigram_freq, "Encrypted Audio Bigram Heatmap", os.path.join(OUTPUT_DIR, f"{basename}_encrypted_bigram_heatmap.png"))
        
        # Cross-Correlation
        cross_corr = compute_cross_correlation(raw_data, encrypted_data)
        lags = correlation_lags(len(raw_data), len(encrypted_data), mode='full')
        plot_cross_correlation(cross_corr, lags, f"Cross-Correlation (Raw vs Encrypted {basename})", os.path.join(OUTPUT_DIR, f"{basename}_cross_correlation.png"))
        
        # Auto-Correlation
        raw_auto_corr = compute_auto_correlation(raw_data)
        encrypted_auto_corr = compute_auto_correlation(encrypted_data)
        raw_lags = correlation_lags(len(raw_data), len(raw_data), mode='full')
        encrypted_lags = correlation_lags(len(encrypted_data), len(encrypted_data), mode='full')
        
        plot_auto_correlation(raw_auto_corr, raw_lags, f"Auto-Correlation (Raw {basename})", os.path.join(OUTPUT_DIR, f"{basename}_raw_auto_correlation.png"))
        plot_auto_correlation(encrypted_auto_corr, encrypted_lags, f"Auto-Correlation (Encrypted {basename})", os.path.join(OUTPUT_DIR, f"{basename}_encrypted_auto_correlation.png"))
        
        # Time-Frequency Analysis
        raw_stft = compute_stft(raw_data)
        encrypted_stft = compute_stft(encrypted_data)
        
        plot_stft(raw_stft, f"STFT Magnitude Spectrogram (Raw {basename})", os.path.join(OUTPUT_DIR, f"{basename}_raw_stft.png"))
        plot_stft(encrypted_stft, f"STFT Magnitude Spectrogram (Encrypted {basename})", os.path.join(OUTPUT_DIR, f"{basename}_encrypted_stft.png"))
        
        # Feature Extraction
        raw_features = extract_audio_features(RAW_AUDIO_PATH)
        encrypted_features = extract_audio_features(encrypted_file)
        
        if raw_features:
            print(f"Raw Audio Features: {raw_features}")
        if encrypted_features:
            print(f"Encrypted Audio Features ({basename}): {encrypted_features}")

    print(f"All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()