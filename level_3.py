import os
import subprocess


# "Comprehensive Audio Analysis and Validation Orchestrator"
# This script orchestrates the execution of various analysis tasks on 
# preprocessed audio data. It runs multiple analysis scripts to extract features, validate 
# decryption, and perform advanced cryptanalysis. It also manages the final validation and 
# clustering of the results, ensuring a comprehensive analysis workflow.


# Define directories and paths
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./final_analysis_results"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_script(script_name, *args):
    """Run a Python script with the given arguments."""
    command = ["python", script_name] + list(args)
    subprocess.run(command, check=True)

def main():
    # Step 1: Run Level 2 scripts to analyze data
    print("Running mutual information analysis...")
    run_script("mutual_information.py")

    print("Running transition matrix analysis...")
    run_script("transition_matrix.py")

    print("Running sliding window analysis...")
    run_script("sliding_window.py")

    print("Running wavelet analysis...")
    run_script("wavelet_analysis.py")

    print("Running frequency domain analysis...")
    run_script("frequency_analysis.py")

    print("Running entropy analysis...")
    run_script("entropy_analysis.py")

    print("Running PCA analysis...")
    run_script("pca_analysis.py")

    print("Running STFT analysis...")
    run_script("sft_analysis.py")

    print("Running differential and Kasiski analysis...")
    run_script("diff_analysis.py")

    print("Running audio validation...")
    run_script("audio_validation.py")

    print("Running adaptive histogram equalization...")
    run_script("adaptive_histogram_equalization.py")

    print("Running GIMP cipher analysis...")
    run_script("gimp_cipher_analysis.py")

    print("Running GIMP block analysis...")
    run_script("gimp_block_analysis.py")

    print("Running wavelet encryption analysis...")
    run_script("audio_encryption_wavelet_analysis.py")

    # Step 2: Run Level 3 analysis scripts
    print("Running NPU clustering and classification analysis...")
    run_script("NPU_clustering.py")

    print("Running advanced cryptanalysis...")
    run_script("advanced_cryptanalysis.py")

    print("Running time-frequency pattern analysis...")
    run_script("time_frequency.py")

    print("All analyses complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
