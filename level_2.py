import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")

def run_script(script_name):
    """Runs a Python script using subprocess."""
    command = ['python3', script_name] # Modified line
    try:
        subprocess.run(command, check=True)
        print(f"Script {script_name} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_name}: {e}")

def main():
    print("Generating modified raw audio files with random bit flips...")
    run_script("create_variable_raw.py")

    print("Reading and processing raw audio data...")
    run_script("handle_raw_audio.py")

    print("Generating top 20 decrypted files report...")
    run_script("generate_top_20.py")

    print("Analyzing spectrograms for encryption patterns...")
    run_script("gimp_encryption_pattern_detection.py")

    print("Cropping images to specified coordinates...")
    run_script("batch_crop_images.py")

    # Step 2: Analyze the data produced by Level 1 scripts
    print("Running histogram analysis on raw and modified audio files...")
    run_script("images_modified_raw.py")

    print("Calculating mutual information between raw and encrypted audio files...")
    run_script("mutual_information.py")

    print("Analyzing byte transformation patterns and validating decryption...")
    run_script("transition_matrix.py")

    print("Performing time-frequency analysis using sliding windows...")
    run_script("sliding_window.py")

    print("Performing wavelet analysis on audio data...")
    run_script("wavelet_analysis.py")

    print("Analyzing the frequency domain of audio files...")
    run_script("frequency_analysis.py")

    print("Calculating entropy of audio files using sliding windows...")
    run_script("entropy_analysis.py")

    print("Performing PCA on audio features...")
    run_script("pca_analysis.py")

    print("Analyzing short-time Fourier transform of audio data...")
    run_script("sft_analysis.py")

    print("Performing differential and Kasiski examination analysis...")
    run_script("diff_analysis.py")

    print("Validating decrypted audio files using various metrics...")
    run_script("audio_validation.py")

    print("Applying adaptive histogram equalization to audio data...")
    run_script("adaptive_histogram_equalization.py")

    print("Analyzing spectrograms for frequency-band encryption patterns...")
    run_script("gimp_cipher_analysis.py")

    print("Analyzing spectrograms for block-based encryption patterns...")
    run_script("gimp_block_analysis.py")

    print("Analyzing wavelet-like features from spectrograms...")
    run_script("audio_encryption_wavelet_analysis.py")

    print("All data generation and modification tasks complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
