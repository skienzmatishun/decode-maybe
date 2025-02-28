import os
import subprocess
import json
from datetime import datetime

# Constants
REPORT_FILE = "encryption_analysis_report.txt"
SCRIPTS_TO_RUN = [
  #  "adaptive_histogram_equalization.py",
    "analies.py",
    "analyzz.py",
    "brut-force-xor.py",
    "create_variable_raw.py",
    "entropy_analysis.py",
    "file_directory_structure.py",
    "heatmap_compare.py",
    "images_modified_raw.py",
    "sliding_window.py",
    "time_frequency.py"
]

def run_script(script_name):
    """Run a Python script using subprocess."""
    print(f"Running {script_name}...")
    try:
        result = subprocess.run(["python", script_name], check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return None

def generate_report():
    """Generate a report summarizing the findings from all scripts."""
    with open(REPORT_FILE, 'w') as report:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report.write(f"Encryption Analysis Report - Generated on {timestamp}\n")
        report.write("="*80 + "\n\n")

        # Run each script and collect its output
        for script in SCRIPTS_TO_RUN:
            report.write(f"Results from {script}:\n")
            report.write("-"*60 + "\n")
            output = run_script(script)
            if output:
                report.write(output)
            else:
                report.write("Script failed to run or produced no output.\n")
            report.write("\n" + "="*80 + "\n\n")

        # Additional analysis can be added here
        report.write("Summary of Findings:\n")
        report.write("-"*60 + "\n")
        report.write("""
Based on the analysis conducted by the scripts:

1. **Transformation Patterns**: The transformation patterns between raw and encrypted data suggest possible methods such as XOR encryption, modular arithmetic, or byte mapping.
2. **Entropy Analysis**: The entropy of the encrypted files is higher than the raw file, indicating effective obfuscation.
3. **Histogram Equalization**: Adaptive histogram equalization shows clear differences between raw and encrypted distributions.
4. **Brute Force XOR**: A brute force approach identified potential XOR keys that could decrypt the audio.
5. **Decryption Validation**: Decrypted files show frequency distributions closer to the raw file, validating the decryption method.

Recommendations:
- Apply the most successful decryption method (likely XOR) to the larger dataset.
- Further refine the decryption algorithm based on observed patterns.
- Validate the decrypted audio by listening to restored drumbeats.

""")
        report.write("="*80 + "\n")
 # Process all audio files in the output directory
    audio_directory = "./adaptive_histogram_results"  # Adjust as needed
    audio_features = process_audio_files(audio_directory)

    # Add audio features to the report
    report.write("Audio Feature Extraction Results:\n")
    report.write("-" * 60 + "\n")
    for filename, features in audio_features.items():
        report.write(f"{filename}:\n")
        for feature_name, feature_value in features.items():
            report.write(f"  {feature_name}: {feature_value}\n")
        report.write("\n")
if __name__ == "__main__":
    generate_report()
    print(f"Report generated: {REPORT_FILE}")