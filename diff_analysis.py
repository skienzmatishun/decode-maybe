import os
import numpy as np
from math import gcd
from dotenv import load_dotenv
import glob

load_dotenv()  # Load environment variables from .env file

# Constants
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
ENCRYPTED_FILE = os.getenv("ENCRYPTED_FILE")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./analysis_results"
WINDOW_SIZE = 10000  # Size of the window for brute-force XOR

def read_file(filename):
    """Reads a binary file and returns its content as a NumPy array of bytes"""
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Invalid filename: {filename}")
    try:
        with open(filename, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None

def create_transition_matrix(raw, encrypted):
    """Creates a byte transition matrix from raw to encrypted data"""
    min_len = min(len(raw), len(encrypted))
    transition = np.zeros((256, 256), dtype=int)
    for r, e in zip(raw[:min_len], encrypted[:min_len]):
        transition[r, e] += 1
    return transition

def report_common_transitions(transition, top=10):
    """Reports the most common byte transitions in the transition matrix"""
    transitions = []
    for r in range(256):
        for e in range(256):
            count = transition[r, e]
            if count > 0:
                transitions.append((r, e, count))
    transitions.sort(key=lambda x: (-x[2], x[0], x[1]))
    return transitions[:top]

def calculate_similarity(a, b):
    """Calculates similarity between two byte arrays using Chi-squared statistic"""
    hist_a = np.bincount(a, minlength=256)
    hist_b = np.bincount(b, minlength=256)
    smooth = 1e-10
    chi2 = np.sum(((hist_a - hist_b) ** 2) / (hist_b + smooth))
    return 1 / (1 + chi2)

def brute_force_xor_key(raw, encrypted, window_size=10000):
    """Brute-forces XOR keys and scores decrypted data using histogram similarity"""
    best_key = None
    best_similarity = -1
    min_len = min(len(raw), len(encrypted))
    window_size = min(window_size, min_len)

    raw_window = raw[:window_size]
    encrypted_window = encrypted[:window_size]

    for key in range(256):
        decrypted_window = encrypted_window ^ key
        similarity = calculate_similarity(decrypted_window, raw_window)
        if similarity > best_similarity:
            best_similarity = similarity
            best_key = key
    return best_key, best_similarity

def kasiski(encrypted_data, sequence_length=3):
    """Estimates key length using Kasiski examination"""
    sequences = {}
    for i in range(len(encrypted_data) - sequence_length):
        seq = tuple(encrypted_data[i:i+sequence_length])
        if seq in sequences:
            sequences[seq].append(i)
        else:
            sequences[seq] = [i]
    distances = []
    for indices in sequences.values():
        if len(indices) >= 2:
            for j in range(1, len(indices)):
                distances.append(indices[j] - indices[0])
    if not distances:
        return None
    current_gcd = distances[0]
    for d in distances[1:]:
        current_gcd = gcd(current_gcd, d)
        if current_gcd == 1:
            break
    return current_gcd

def analyze_file(raw_data, encrypted_data, basename, output_dir):
    """Analyze a single encrypted or decrypted file"""
    if encrypted_data is None:
        print(f"Skipping analysis for {basename} due to missing data.")
        return

    # Differential Analysis
    transition = create_transition_matrix(raw_data, encrypted_data)
    common_transitions = report_common_transitions(transition)
    print(f"\nTop 10 byte transitions (raw -> {basename}):")
    for r, e, count in common_transitions:
        print(f"0x{r:02X} -> 0x{e:02X}: {count} occurrences")

    # Key Space Exhaustion (Brute-force XOR)
    key, similarity = brute_force_xor_key(raw_data, encrypted_data, window_size=WINDOW_SIZE)
    print(f"\nBest XOR Key for {basename}: 0x{key:02X} (Similarity Score: {similarity:.4f})")

    # Kasiski Examination
    key_length = kasiski(encrypted_data)
    print(f"\nEstimated Key Length (Kasiski) for {basename}: {key_length}")

def main():
    """Main entry point for the script"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if required environment variables are set
    if not RAW_AUDIO_PATH:
        print("Error: RAW_AUDIO_PATH is not set in the .env file.")
        return
    if not ENCRYPTED_DIR:
        print("Error: ENCRYPTED_DIR is not set in the .env file.")
        return
    if not ENCRYPTED_FILE:
        print("Error: ENCRYPTED_FILE is not set in the .env file.")
        return
    if not DECRYPTED_DIRS:
        print("Error: DECRYPTED_DIRS is not set in the .env file.")
        return

    # Construct the full path for the encrypted file
    encrypted_filepath = os.path.join(ENCRYPTED_DIR, ENCRYPTED_FILE)
    print(f"Processing encrypted file: {encrypted_filepath}")

    # Read raw audio data
    raw_data = read_file(RAW_AUDIO_PATH)
    if raw_data is None:
        print(f"Failed to read raw audio file {RAW_AUDIO_PATH}. Exiting.")
        return

    # Process the encrypted file
    encrypted_data = read_file(encrypted_filepath)
    basename = os.path.splitext(os.path.basename(ENCRYPTED_FILE))[0]
    analyze_file(raw_data, encrypted_data, basename, OUTPUT_DIR)

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
            print(f"Processing file: {filepath}")
            decrypted_data = read_file(filepath)
            analyze_file(raw_data, decrypted_data, basename, OUTPUT_DIR)

if __name__ == "__main__":
    main()