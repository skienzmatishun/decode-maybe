# analies.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import glob
from handle_raw_audio import read_raw_audio
import soundfile as sf
from extract_audio_features import extract_audio_features
# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./decryption_analysis"

import numpy as np

def compute_histogram(data):
    # Tentative fix: Convert float32 data to integers in 0-255 range (APPROXIMATION - review logic!)
    # Scale float data from -1 to 1 to 0 to 255 range (rough approximation)
    int_data = ((data + 1.0) * 127.5).astype(int)  # Scale and shift to 0-255, then convert to int
    # Clip values to ensure they are within 0-255 (just in case of rounding issues)
    int_data = np.clip(int_data, 0, 255)
    return np.bincount(int_data, minlength=256)

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
    print("Analyzing XOR key...")
    possible_keys = {}

    # Convert to integers
    raw_data_int = ((raw_data + 1.0) * 127.5).astype(int)
    encrypted_data_int = ((encrypted_data + 1.0) * 127.5).astype(int)

    for i in range(min(len(raw_data_int), len(encrypted_data_int))):
        key = raw_data_int[i] ^ encrypted_data_int[i]
        key = np.clip(key, 0, 255)
        if key not in possible_keys:
            possible_keys[key] = 0
        possible_keys[key] += 1

    # Convert dictionary to list of (key, frequency) tuples and sort by frequency (descending)
    sorted_keys = sorted(possible_keys.items(), key=lambda item: item[1], reverse=True) # Sort by value (frequency)

    return sorted_keys # Return the sorted list

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

def test_decryption_methods(raw_data, encrypted_data, raw_hist, OUTPUT_DIR):
    """Test different decryption methods and evaluate results."""
    print("\nTesting possible decryption methods...")

    xor_results = analyze_xor_key(raw_data, encrypted_data)
    if not xor_results:  # Check if xor_results is empty
        print("No XOR keys found to analyze.")
        return {}  # Return empty dictionary if no keys

    best_xor_key_tuple = xor_results[0]  # Get the tuple (key, frequency) for the most frequent key
    best_xor_key = best_xor_key_tuple[0]  # Extract just the key (the first element of the tuple)
    best_xor_similarity = xor_results[0][1]  # Extract the frequency (second element of the tuple)

    print(f"Best XOR key: {best_xor_key}, Frequency: {best_xor_similarity}")

    # Convert encrypted_data to integer format BEFORE bitwise_xor
    encrypted_data_int = ((encrypted_data + 1.0) * 127.5).astype(int)

    # Apply XOR decryption with the best key
    decrypted_xor = np.bitwise_xor(encrypted_data_int, best_xor_key)  # Use the INT version of encrypted_data

    # Clip and convert decrypted data back to float for saving (if needed for audio output)
    decrypted_xor_float = np.clip(decrypted_xor / 127.5 - 1.0, -1.0, 1.0).astype(np.float32)  # Reverse scaling and clip

    output_file_xor = os.path.join(OUTPUT_DIR, "decrypted_xor.raw")
    sf.write(output_file_xor, decrypted_xor_float, 16000, format='RAW', subtype='PCM_16')
    print(f"Decrypted XOR audio saved to: {output_file_xor}")

    # Ensure raw_hist is passed correctly to analyze_frequency
    frequency_analysis_results = analyze_frequency(decrypted_xor_float, raw_hist)  

    return {"XOR": (best_xor_key, best_xor_similarity, output_file_xor, frequency_analysis_results)}

def analyze_frequency(data, raw_hist):
    print("Placeholder: analyze_frequency function called, but not yet implemented.")
    return None  # Or return some default value

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read raw audio data
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    plot_histogram(raw_hist, "Raw Audio Frequency Distribution", 
                 os.path.join(OUTPUT_DIR, "raw_frequency_distribution.png"))
    
    # Process the first encrypted file for initial analysis
    encrypted_files = glob.glob(os.path.join(ENCRYPTED_RAW_DIR, "*.raw"))
    if not encrypted_files:
        print("No encrypted files found!")
        return
    
    first_file = encrypted_files[0]
    print(f"Analyzing primary file: {first_file}")
    encrypted_data = read_raw_audio(first_file)
    encrypted_hist = compute_histogram(encrypted_data)
    plot_histogram(encrypted_hist, f"Encrypted Audio Frequency Distribution", 
                 os.path.join(OUTPUT_DIR, "encrypted_frequency_distribution.png"))
    
    # Analyze transformation patterns
    print("Testing possible decryption methods...")
    decryption_results = test_decryption_methods(raw_data, encrypted_data, raw_hist, OUTPUT_DIR)
    
    # Report results
    print("\nDecryption Method Results (Sorted by Similarity):")
    for method, (key, similarity, output_file, frequency_analysis_results) in decryption_results.items():
        print(f"Method: {method}, Key: {key}, Similarity: {similarity:.4f}")

    
    best_method, (best_key, best_similarity, best_decrypted, frequency_analysis_results) = max(
    decryption_results.items(), key=lambda x: x[1][1]
)
    
    # Apply the best method to all encrypted files
    # Apply the best decryption method to all encrypted files
    print("\nApplying best decryption method to all encrypted files...")
    for encrypted_file in encrypted_files:
        basename = os.path.splitext(os.path.basename(encrypted_file))[0]
        encrypted_data = read_raw_audio(encrypted_file)

        # Apply the best decryption method
        decrypted = None  # Ensure decrypted is always defined

        if best_method == "XOR":
            encrypted_data_int = ((encrypted_data + 1.0) * 127.5).astype(int)  # Convert to int
            decrypted = np.bitwise_xor(encrypted_data_int, best_key)
            decrypted = np.clip(decrypted / 127.5 - 1.0, -1.0, 1.0).astype(np.float32)  # Convert back to float
        elif best_method == "Addition":
            decrypted = (encrypted_data - best_key) % 256
        elif best_method == "Subtraction":
            decrypted = (encrypted_data + best_key) % 256
        elif best_method == "Mapping":
            # Recreate the inverse mapping for the best result
            forward_mapping = test_byte_mapping(raw_data, read_raw_audio(first_file), OUTPUT_DIR)
            inverse_mapping = {v: k for k, v in forward_mapping.items()}
            decrypted = decrypt_with_mapping(encrypted_data, inverse_mapping)

        # Ensure decryption was successful before saving
        if decrypted is None:
            print(f"Best decryption method: {best_method}")
            print(f"Warning: No decryption applied for {basename}. Skipping...")
            continue  # Skip to the next file

        # Save the decrypted file
        output_file = os.path.join(OUTPUT_DIR, f"decrypted_{basename}.raw")
        with open(output_file, "wb") as f:
            f.write(decrypted.tobytes())

    print(f"Decrypted {basename} saved to {output_file}")

        
        # Extract features from decrypted audio
    decrypted_features = extract_audio_features(output_file)
    if decrypted_features:
        print(f"Decrypted Audio Features ({basename}): {decrypted_features}")
    
    print(f"\nAll results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()