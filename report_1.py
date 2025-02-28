import subprocess
import os

# Configuration
PROJECT_DIR = os.getcwd()
RAW_AUDIO_PATH = os.path.join(PROJECT_DIR, "left.raw")
ENCRYPTED_DIR = os.path.join(PROJECT_DIR, "modified_raw")
REPORT_DIR = os.path.join(PROJECT_DIR, "analysis_reports")
SCRIPTS_DIR = PROJECT_DIR

# Ensure directories exist
os.makedirs(REPORT_DIR, exist_ok=True)

# Script execution order
analysis_sequence = [
    "images_modified_raw.py",       # Initial histogram analysis
    "analyzz.py",                   # XOR analysis and decryption methods
    "stft_analysis.py",             # Spectral analysis
    "entropy_analysis.py",          # Entropy analysis
    "pca_analysis.py",              # Principal component analysis
    "mutual_information.py",        # Mutual information analysis
    "do_stuff.py",                  # Bit-flip analysis and validation
    "transition_matrix.py",         # Transformation pattern analysis
    "sliding_window.py"             # Time-series entropy analysis
]

# Run all analysis scripts
for script in analysis_sequence:
    print(f"\nRunning {script}...")
    try:
        subprocess.run(
            ["python", os.path.join(SCRIPTS_DIR, script)],
            check=True,
            cwd=SCRIPTS_DIR
        )
        print(f"✅ {script} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script}: {e}")
        print("Continuing with remaining scripts...")

# Generate final report
print("\nGenerating final report...")
try:
    subprocess.run(
        ["python", os.path.join(SCRIPTS_DIR, "make_report.py")],
        check=True,
        cwd=SCRIPTS_DIR
    )
    print(f"✅ Report saved to {REPORT_DIR}/encryption_analysis_report.txt")
except subprocess.CalledProcessError as e:
    print(f"❌ Error generating report: {e}")

print("\nAnalysis workflow complete. Key deliverables:")
print(f"- Frequency distribution comparisons in {REPORT_DIR}/decryption_analysis")
print(f"- STFT spectrograms in {REPORT_DIR}/stft_analysis")
print(f"- Final report in {REPORT_DIR}/encryption_analysis_report.txt")
