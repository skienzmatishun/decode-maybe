import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import glob
from handle_raw_audio import read_raw_audio
from extract_audio_features import extract_audio_features
import random
import datetime
from adaptive_histogram_equalization import apply_adaptive_histogram_equalization

# Constants
RAW_AUDIO_PATH = "./left.raw" 
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./decryption_analysis_2"
BIT_FLIP_OUTPUT_DIR = "./bit_flip_modified"
NUM_BIT_FLIP_FILES = 100
MIN_BIT_FLIP_PERCENTAGE = 0.001
MAX_BIT_FLIP_PERCENTAGE = 0.1

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

def analyze_transformation_patterns(raw_data, encrypted_data):
    """Analyze possible transformation patterns between raw and encrypted data."""
    # Create a transformation mapping
    transform_mapping = {}
    inverse_mapping = {}
    
    # Find byte pairs that occur most frequently
    min_length = min(len(raw_data), len(encrypted_data))
    for i in range(min_length):
        raw_byte = raw_data[i]
        enc_byte = encrypted_data[i]
        
        if raw_byte not in transform_mapping:
            transform_mapping[raw_byte] = {}
        
        if enc_byte not in transform_mapping[raw_byte]:
            transform_mapping[raw_byte][enc_byte] = 0
        
        transform_mapping[raw_byte][enc_byte] += 1
        
        # Also track inverse mapping
        if enc_byte not in inverse_mapping:
            inverse_mapping[enc_byte] = {}
        
        if raw_byte not in inverse_mapping[enc_byte]:
            inverse_mapping[enc_byte][raw_byte] = 0
        
        inverse_mapping[enc_byte][raw_byte] += 1
    
    # Determine most likely transformation rule
    transformation_rule = {}
    inverse_rule = {}
    
    for raw_byte, mappings in transform_mapping.items():
        most_common_enc = max(mappings.items(), key=lambda x: x[1])[0]
        transformation_rule[raw_byte] = most_common_enc
    
    for enc_byte, mappings in inverse_mapping.items():
        most_common_raw = max(mappings.items(), key=lambda x: x[1])[0]
        inverse_rule[enc_byte] = most_common_raw
    
    return transformation_rule, inverse_rule

def analyze_xor_key(raw_data, encrypted_data):
    """Check if the transformation could be a simple XOR with a key."""
    min_length = min(len(raw_data), len(encrypted_data))
    
    possible_keys = {}
    for i in range(min_length):
        key = raw_data[i] ^ encrypted_data[i]
        if key not in possible_keys:
            possible_keys[key] = 0
        possible_keys[key] += 1
    
    # Sort by frequency
    sorted_keys = sorted(possible_keys.items(), key=lambda x: x[1], reverse=True)
    
    # Test the top 5 keys
    top_keys = [k for k, _ in sorted_keys[:5]]
    results = []
    
    for key in top_keys:
        decrypted = np.bitwise_xor(encrypted_data[:min_length], key)
        raw_hist = compute_histogram(raw_data[:min_length])
        decrypted_hist = compute_histogram(decrypted)
        
        # Use Jensen-Shannon divergence to compare histograms
        similarity = 1.0 - entropy(raw_hist + 1, decrypted_hist + 1) / np.log(2)
        results.append((key, similarity))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

def test_byte_mapping(raw_data, encrypted_data, output_dir):
    """Test if there's a consistent byte-to-byte mapping."""
    min_length = min(len(raw_data), len(encrypted_data))
    mapping = np.zeros((256, 256), dtype=np.int32)
    
    for i in range(min_length):
        mapping[raw_data[i], encrypted_data[i]] += 1
    
    # Plot the mapping as a heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log1p(mapping), cmap='viridis')
    plt.colorbar(label='log(count + 1)')
    plt.title('Byte Mapping Heatmap (Raw → Encrypted)')
    plt.xlabel('Encrypted Byte')
    plt.ylabel('Raw Byte')
    
    # Add some hex labels
    hex_positions = list(range(0, 256, 32))
    hex_labels = [f"{i:02X}" for i in hex_positions]
    plt.xticks(hex_positions, hex_labels)
    plt.yticks(hex_positions, hex_labels)
    
    plt.savefig(os.path.join(output_dir, "byte_mapping_heatmap.png"))
    plt.close()
    
    # Find the most common mapping for each raw byte
    forward_mapping = {}
    for i in range(256):
        if np.sum(mapping[i, :]) > 0:
            forward_mapping[i] = np.argmax(mapping[i, :])
    
    return forward_mapping

def decrypt_with_mapping(encrypted_data, mapping):
    """Decrypt data using a byte-to-byte mapping."""
    decrypted = np.zeros_like(encrypted_data)
    for i in range(len(encrypted_data)):
        if encrypted_data[i] in mapping:
            decrypted[i] = mapping[encrypted_data[i]]
        else:
            decrypted[i] = encrypted_data[i]  # Keep unchanged if no mapping
    return decrypted

def analyze_modular_arithmetic(raw_data, encrypted_data):
    """Test if the transformation follows the pattern: enc = (raw + key) % 256 or enc = (raw - key) % 256."""
    min_length = min(len(raw_data), len(encrypted_data))
    
    # Cast to int32 to avoid overflow during subtraction
    raw_data_int32 = raw_data.astype(np.int32)
    encrypted_data_int32 = encrypted_data.astype(np.int32)
    
    # Test addition
    add_keys = {}
    for i in range(min_length): 
        key = (encrypted_data_int32[i] - raw_data_int32[i]) % 256
        if key not in add_keys:
            add_keys[key] = 0
        add_keys[key] += 1
    
    # Test subtraction
    sub_keys = {}
    for i in range(min_length):
        key = (raw_data_int32[i] - encrypted_data_int32[i]) % 256
        if key not in sub_keys:
            sub_keys[key] = 0
        sub_keys[key] += 1
    
    # Get top candidates
    top_add_keys = sorted(add_keys.items(), key=lambda x: x[1], reverse=True)[:3]
    top_sub_keys = sorted(sub_keys.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "addition": top_add_keys,
        "subtraction": top_sub_keys
    }

def test_decryption_methods(raw_data, encrypted_data, output_dir):
    """Test different decryption methods and evaluate results."""
    results = []
    
    # 1. Test XOR keys
    xor_results = analyze_xor_key(raw_data, encrypted_data)
    best_xor_key, best_xor_similarity = xor_results[0]
    
    decrypted_xor = np.bitwise_xor(encrypted_data, best_xor_key)
    results.append(("XOR", best_xor_key, best_xor_similarity, decrypted_xor))
    
    # 2. Test modular arithmetic
    mod_results = analyze_modular_arithmetic(raw_data, encrypted_data)
    
    # Addition (decryption is subtraction)
    best_add_key, best_add_count = mod_results["addition"][0]
    decrypted_add = (encrypted_data - best_add_key) % 256
    add_similarity = 1.0 - entropy(compute_histogram(raw_data), compute_histogram(decrypted_add) + 1) / np.log(2)
    results.append(("Addition", best_add_key, add_similarity, decrypted_add))
    
    # Subtraction (decryption is addition)
    best_sub_key, best_sub_count = mod_results["subtraction"][0]
    decrypted_sub = (encrypted_data + best_sub_key) % 256
    sub_similarity = 1.0 - entropy(compute_histogram(raw_data), compute_histogram(decrypted_sub) + 1) / np.log(2)
    results.append(("Subtraction", best_sub_key, sub_similarity, decrypted_sub))
    
    # 3. Test byte mapping
    forward_mapping = test_byte_mapping(raw_data, encrypted_data, output_dir)
    
    # Invert the mapping
    inverse_mapping = {}
    for raw_byte, enc_byte in forward_mapping.items():
        inverse_mapping[enc_byte] = raw_byte
    
    decrypted_mapping = decrypt_with_mapping(encrypted_data, inverse_mapping)
    mapping_similarity = 1.0 - entropy(compute_histogram(raw_data), compute_histogram(decrypted_mapping) + 1) / np.log(2)
    results.append(("Mapping", len(inverse_mapping), mapping_similarity, decrypted_mapping))
    
    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Plot histograms for comparison
    raw_hist = compute_histogram(raw_data)
    
    for method, key, similarity, decrypted in results:
        decrypted_hist = compute_histogram(decrypted)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(256), raw_hist, color="blue", alpha=0.5, label="Raw")
        plt.bar(range(256), decrypted_hist, color="red", alpha=0.5, label="Decrypted")
        plt.legend()
        plt.title(f"{method} Decryption (Key: {key}, Similarity: {similarity:.4f})")
        
        plt.subplot(1, 2, 2)
        plt.bar(range(256), np.abs(raw_hist - decrypted_hist), color="green", alpha=0.7)
        plt.title("Absolute Difference")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"decryption_{method}_key_{key}.png"))
        plt.close()
        
        # Save the decrypted data
        with open(os.path.join(output_dir, f"decrypted_{method}_key_{key}.raw"), "wb") as f:
            f.write(decrypted.tobytes())
    
    return results

def flip_bits_in_file(input_file, output_file, flip_percentage=0.01):
    """Reads a raw audio file, flips bits randomly, and saves the modified data."""
    with open(input_file, 'rb') as f_in:
        data = bytearray(f_in.read())

    num_bytes_to_flip = int(len(data) * flip_percentage)
    
    for _ in range(num_bytes_to_flip):
        byte_index = random.randint(0, len(data) - 1)
        bit_index = random.randint(0, 7)  # 8 bits in a byte

        # Flip the bit
        data[byte_index] ^= (1 << bit_index)

    with open(output_file, 'wb') as f_out:
        f_out.write(data)

def generate_random_flipped_files(input_file, output_dir, num_files, min_percentage, max_percentage):
    """Generates multiple versions of the input file with random bit flips, using a random percentage within a given range."""
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(num_files):
        percentage = random.uniform(min_percentage, max_percentage)  # Generate random percentage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_filename}_flipped_{percentage:.4f}_{i+1:03d}_{timestamp}.raw"
        output_file = os.path.join(output_dir, output_filename)
        flip_bits_in_file(input_file, output_file, percentage)
        print(f"Generated: {output_file}")

def analyze_bit_flipped_files(bit_flip_output_dir, output_dir):
    """Analyze all bit-flipped files in the given directory."""
    for filename in os.listdir(bit_flip_output_dir):
        if filename.endswith(".raw"):
            filepath = os.path.join(bit_flip_output_dir, filename)
            flipped_data = read_raw_audio(filepath)
            flipped_hist = compute_histogram(flipped_data)
            flipped_hist_equalized = apply_adaptive_histogram_equalization(flipped_hist)
            
            basename = os.path.splitext(filename)[0]
            plot_path = os.path.join(output_dir, f"{basename}_frequency_distribution.png")
            equalized_plot_path = os.path.join(output_dir, f"equalized_{basename}_frequency_distribution.png")
            
            plot_histogram(flipped_hist, f"Bit-Flipped ({basename}) Frequency Distribution", plot_path)
            plot_histogram(flipped_hist_equalized, f"Equalized Bit-Flipped ({basename}) Frequency Distribution", equalized_plot_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BIT_FLIP_OUTPUT_DIR, exist_ok=True)

    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    plot_histogram(raw_hist, "Raw Audio Frequency Distribution", os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"))

    # Generate and analyze bit-flipped files
    generate_random_flipped_files(RAW_AUDIO_PATH, BIT_FLIP_OUTPUT_DIR, NUM_BIT_FLIP_FILES, MIN_BIT_FLIP_PERCENTAGE, MAX_BIT_FLIP_PERCENTAGE)
    analyze_bit_flipped_files(BIT_FLIP_OUTPUT_DIR, OUTPUT_DIR)

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

        # Analyze transformation patterns
        print("Testing possible decryption methods...")
        decryption_results = test_decryption_methods(raw_data, encrypted_data, OUTPUT_DIR)
        
        # Report results
        print("\nDecryption Method Results (Sorted by Similarity):")
        for method, key, similarity, _ in decryption_results:
            print(f"Method: {method}, Key: {key}, Similarity: {similarity:.4f}")
        
        best_method, best_key, best_similarity, best_decrypted = decryption_results[0]
        print(f"\nBest method: {best_method} with key {best_key} (Similarity: {best_similarity:.4f})")

        # Apply the best method to all encrypted files
        print("\nApplying best decryption method to all encrypted files...")
        for encrypted_file in encrypted_files:
            basename = os.path.splitext(os.path.basename(encrypted_file))[0]
            encrypted_data = read_raw_audio(encrypted_file)
            
            # Apply the best decryption method
            if best_method == "XOR":
                decrypted = np.bitwise_xor(encrypted_data, best_key)
            elif best_method == "Addition":
                decrypted = (encrypted_data - best_key) % 256
            elif best_method == "Subtraction":
                decrypted = (encrypted_data + best_key) % 256
            elif best_method == "Mapping":
                # Recreate the inverse mapping for the best result
                forward_mapping = test_byte_mapping(raw_data, read_raw_audio(encrypted_file), OUTPUT_DIR)
                inverse_mapping = {v: k for k, v in forward_mapping.items()}
                decrypted = decrypt_with_mapping(encrypted_data, inverse_mapping)
            
            # Save the decrypted file
            output_file = os.path.join(OUTPUT_DIR, f"decrypted_{basename}.raw")
            with open(output_file, "wb") as f:
                f.write(decrypted.tobytes())
            
            print(f"Decrypted {basename} saved to {output_file}")

            # Extract features from decrypted audio
            decrypted_features = extract_audio_features_librosa(output_file)
            if decrypted_features:
                print(f"Decrypted Audio Features ({basename}): {decrypted_features}")

    print(f"\nAll results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()