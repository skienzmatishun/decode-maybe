import os
import subprocess



# "Intermediate Audio Data Analysis and Preparation Orchestrator"
#
#  This script orchestrates the execution of various data generation and modification tasks, ensuring t
# hat raw audio data is processed, modified, and prepared for further analysis. It manages the creation 
# of modified audio files, reads and processes raw audio data, and runs initial analyses to extract features 
# and validate decryption. This script serves as a bridge between raw data and comprehensive analysis, 
# preparing the data for more advanced analytical tasks.

# Define directories and paths
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./level_2_results"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_script(script_name, *args):
    """Run a Python script with the given arguments."""
    command = ["python", script_name] + list(args)
    subprocess.run(command, check=True)

def main():
    # Step 1: Generate and modify data using Level 1 scripts
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
