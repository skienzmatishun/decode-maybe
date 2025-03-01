import subprocess
import os

# Configuration
PROJECT_DIR = os.getcwd()
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
REPORT_DIR = os.path.join(PROJECT_DIR, "analysis_reports")
SCRIPTS_DIR = PROJECT_DIR

# Ensure directories exist
os.makedirs(ENCRYPTED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Script execution order
analysis_sequence = [
    "create_variable_raw.py",   # Generate modified encrypted samples
    "images_modified_raw.py",   # Initial histogram analysis
    "brut-force-xor.py",        # Brute-force XOR decryption attempts
    "frequency_analysis.py",    # Frequency domain analysis
    "heatmap_compare.py",       # Compare raw/encrypted heatmaps
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
    # Create simplified report from analysis outputs
    report_path = os.path.join(REPORT_DIR, "decryption_summary.txt")
    with open(report_path, "w") as report:
        report.write("Decryption Analysis Summary:\n")
        report.write("============================\n\n")
        
        # List key findings from each analysis
        report.write("1. Brute-force XOR analysis:\n")
        report.write("   - Top candidates: [Manual analysis required]\n\n")
        
        report.write("2. Histogram comparisons:\n")
        report.write("   - Closest match: [Manual analysis required]\n\n")
        
        report.write("3. Frequency domain analysis:\n")
        report.write("   - Peak frequencies: [Manual analysis required]\n\n")
        
        report.write("Recommendations:\n")
        report.write(" - Validate top XOR keys manually\n")
        report.write(" - Compare decrypted histograms with original\n")
        report.write(" - Listen to decrypted audio samples\n")

    print(f"✅ Report saved to {report_path}")
except Exception as e:
    print(f"❌ Error generating report: {e}")

print("\nAnalysis workflow complete. Key deliverables:")
print(f"- Modified encrypted samples: {ENCRYPTED_DIR}")
print(f"- Histogram comparisons: {REPORT_DIR}/images_modified_raw")
print(f"- Frequency domain plots: {REPORT_DIR}/frequency_domain_analysis")
print(f"- Heatmap comparisons: {REPORT_DIR}/heatmap_compare")