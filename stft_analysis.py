import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from handle_raw_audio import read_raw_audio
from dotenv import load_dotenv
import glob

load_dotenv()  # Load environment variables from .env file

# Constants
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./stft_analysis"

def compute_stft(data, sr, n_fft=512, hop_length=256):
    """Compute Short-Time Fourier Transform (STFT) of the audio data."""
    return librosa.stft(data, n_fft=n_fft, hop_length=hop_length)

def plot_stft(stft_matrix, sr, n_fft, hop_length, title, output_path):
    """Plot the magnitude spectrogram of the STFT."""
    plt.figure(figsize=(12, 6))
    D = np.abs(stft_matrix)
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_audio(raw_data, raw_sr, audio_data, audio_sr, basename, output_dir):
    """Analyze a single audio file and compare with raw audio."""
    audio_stft = compute_stft(audio_data, audio_sr)
    plot_stft(audio_stft, audio_sr, 512, 256, f"{basename} STFT Magnitude Spectrogram", os.path.join(output_dir, f"{basename}_stft.png"))

    # Compare STFT magnitude spectrograms between raw and audio
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    D_raw = np.abs(compute_stft(raw_data, raw_sr))
    S_db_raw = librosa.amplitude_to_db(D_raw, ref=np.max)
    librosa.display.specshow(S_db_raw, sr=raw_sr, x_axis='time', y_axis='log', hop_length=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Raw Audio STFT Magnitude Spectrogram")

    plt.subplot(2, 1, 2)
    D_audio = np.abs(audio_stft)
    S_db_audio = librosa.amplitude_to_db(D_audio, ref=np.max)
    librosa.display.specshow(S_db_audio, sr=audio_sr, x_axis='time', y_axis='log', hop_length=256)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{basename} STFT Magnitude Spectrogram")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basename}_stft_comparison.png"))
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    if isinstance(raw_data, tuple):
        raw_data, sr = raw_data
    else:
        sr = 16000  # Default sample rate if not returned as a tuple

    raw_stft = compute_stft(raw_data, sr)
    plot_stft(raw_stft, sr, 512, 256, "Raw Audio STFT Magnitude Spectrogram", os.path.join(OUTPUT_DIR, "raw_stft.png"))

    # Process each encrypted RAW file
    encrypted_files = glob.glob(os.path.join(ENCRYPTED_DIR, "*.raw"))
    if not encrypted_files:
        print("No encrypted files found!")
    else:
        for encrypted_file in encrypted_files:
            basename = os.path.splitext(os.path.basename(encrypted_file))[0]
            filepath = os.path.join(ENCRYPTED_DIR, encrypted_file)
            encrypted_data = read_raw_audio(filepath)
            if isinstance(encrypted_data, tuple):
                encrypted_data, sr = encrypted_data
            else:
                sr = 16000  # Default sample rate if not returned as a tuple

            analyze_audio(raw_data, sr, encrypted_data, sr, basename, OUTPUT_DIR)

    # Process each decrypted RAW file from each directory in DECRYPTED_DIRS
    for decrypted_dir in DECRYPTED_DIRS:
        print(f"Checking directory: {decrypted_dir}")
        decrypted_files = glob.glob(os.path.join(decrypted_dir, "*.raw"))

        if not decrypted_files:
            print(f"No decrypted files found in {decrypted_dir}")
            continue

        for decrypted_file in decrypted_files:
            basename = os.path.splitext(os.path.basename(decrypted_file))[0]
            filepath = os.path.join(decrypted_dir, decrypted_file)
            decrypted_data = read_raw_audio(filepath)
            if isinstance(decrypted_data, tuple):
                decrypted_data, sr = decrypted_data
            else:
                sr = 16000  # Default sample rate if not returned as a tuple

            analyze_audio(raw_data, sr, decrypted_data, sr, basename, OUTPUT_DIR)

    print(f"STFT analysis results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()