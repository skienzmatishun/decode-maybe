import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import glob
from skimage import exposure
import soundfile as sf
from extract_audio_features import extract_audio_features

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
OUTPUT_DIR = "./decryption_analysis"
BIT_FLIP_OUTPUT_DIR = "./bit_flip_modified"
NUM_BIT_FLIP_FILES = 100
MIN_BIT_FLIP_PERCENTAGE = 0.001
MAX_BIT_FLIP_PERCENTAGE = 0.1

def read_raw_audio(file_path):
    """Read raw audio file as a NumPy array."""
    try:
        data, samplerate = sf.read(file_path, samplerate=16000, channels=1, format='RAW', subtype='PCM_16')
    except Exception as e:
        raise Exception(f"Error reading audio file {file_path}: {e}")
    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    data = data.astype(np.float32) / 32768.0
    
    return data

def compute_histogram(data):
    """Compute frequency histogram of byte data (0-255)."""
    int_data = ((data + 1.0) * 127.5).astype(int)
    int_data = np.clip(int_data, 0, 255)
    return np.bincount(int_data, minlength=256)

def apply_adaptive_histogram_equalization(hist, clip_limit=0.03):
    """Apply adaptive histogram equalization to the given histogram."""
    hist_normalized = hist.astype(np.float32) / hist.sum()
    hist_equalized = exposure.equalize_adapthist(hist_normalized.reshape(16, 16), clip_limit=clip_limit).flatten()
    hist_equalized = (hist_equalized * hist.sum()).astype(np.int32)
    return hist_equalized

def plot_histogram(hist, title, output_path):
    """Plot the given histogram with hex values on x-axis."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), hist, color="blue", alpha=0.7)
    plt.title(title)
    
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
    transform_mapping = {}
    inverse_mapping = {}
    
    min_length = min(len(raw_data), len(encrypted_data))
    for i in range(min_length):
        raw_byte = raw_data[i]
        enc_byte = encrypted_data[i]
        
        if raw_byte not in transform_mapping:
            transform_mapping[raw_byte] = {}
        
        if enc_byte not in transform_mapping[raw_byte]:
            transform_mapping[raw_byte][enc_byte] = 0
        
        transform_mapping[raw_byte][enc_byte] += 1
        
        if enc_byte not in inverse_mapping:
            inverse_mapping[enc_byte] = {}
        
        if raw_byte not in inverse_mapping[enc_byte]:
            inverse_mapping[enc_byte][raw_byte] = 0
        
        inverse_mapping[enc_byte][raw_byte] += 1
    
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
    possible_keys = {}
    
    for i in range(min(len(raw_data), len(encrypted_data))):
        key = raw_data[i] ^ encrypted_data[i]
        key = np.clip(key, 0, 255)
        if key not in possible_keys:
            possible_keys[key] = 0
        possible_keys[key] += 1
    
    sorted_keys = sorted(possible_keys.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_keys

def test_byte_mapping(raw_data, encrypted_data, output_dir):
    """Test if there's a consistent byte-to-byte mapping."""
    min_length = min(len(raw_data), len(encrypted_data))
    mapping = np.zeros((256, 256), dtype=np.int32)
    
    for i in range(min_length):
        mapping[raw_data[i], encrypted_data[i]] += 1
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log1p(mapping), cmap='viridis')
    plt.colorbar(label='log(count + 1)')
    plt.title('Byte Mapping Heatmap (Raw → Encrypted)')
    plt.xlabel('Encrypted Byte')
    plt.ylabel('Raw Byte')
    
    hex_positions = list(range(0, 256, 32))
    hex_labels = [f"{i:02X}" for i in hex_positions]
    plt.xticks(hex_positions, hex_labels)
    plt.yticks(hex_positions, hex_labels)
    
    plt.savefig(os.path.join(output_dir, "byte_mapping_heatmap.png"))
    plt.close()
    
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
            decrypted[i] = encrypted_data[i]
    return decrypted

def analyze_modular_arithmetic(raw_data, encrypted_data):
    """Test if the transformation follows the pattern: enc = (raw + key) % 256 or enc = (raw - key) % 256."""
    min_length = min(len(raw_data), len(encrypted_data))
    
    raw_data_int32 = raw_data.astype(np.int32)
    encrypted_data_int32 = encrypted_data.astype(np.int32)
    
    add_keys = {}
    for i in range(min_length):
        key = (encrypted_data_int32[i] - raw_data_int32[i]) % 256
        if key not in add_keys:
            add_keys[key] = 0
        add_keys[key] += 1
    
    sub_keys = {}
    for i in range(min_length):
        key = (raw_data_int32[i] - encrypted_data_int32[i]) % 256
        if key not in sub_keys:
            sub_keys[key] = 0
        sub_keys[key] += 1
    
    top_add_keys = sorted(add_keys.items(), key=lambda x: x[1], reverse=True)[:3]
    top_sub_keys = sorted(sub_keys.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "addition": top_add_keys,
        "subtraction": top_sub_keys
    }

def test_decryption_methods(raw_data, encrypted_data, raw_hist, output_dir):
    """Test different decryption methods and evaluate results."""
    results = []
    
    xor_results = analyze_xor_key(raw_data, encrypted_data)
    if not xor_results:
        print("No XOR keys found to analyze.")
        return {}
    
    best_xor_key_tuple = xor_results[0]
    best_xor_key = best_xor_key_tuple[0]
    best_xor_similarity = xor_results[0][1]
    
    print(f"Best XOR key: {best_xor_key}, Frequency: {best_xor_similarity}")
    
    encrypted_data_int = ((encrypted_data + 1.0) * 127.5).astype(int)
    decrypted_xor = np.bitwise_xor(encrypted_data_int, best_xor_key)
    decrypted_xor_float = np.clip(decrypted_xor / 127.5 - 1.0, -1.0, 1.0).astype(np.float32)
    
    output_file_xor = os.path.join(output_dir, "decrypted_xor.raw")
    sf.write(output_file_xor, decrypted_xor_float, 16000, format='RAW', subtype='PCM_16')
    print(f"Decrypted XOR audio saved to: {output_file_xor}")
    
    frequency_analysis_results = analyze_frequency(decrypted_xor_float, raw_hist)
    
    results.append(("XOR", best_xor_key, best_xor_similarity, output_file_xor, frequency_analysis_results))
    
    mod_results = analyze_modular_arithmetic(raw_data, encrypted_data)
    
    best_add_key, best_add_count = mod_results["addition"][0]
    decrypted_add = (encrypted_data - best_add_key) % 256
    add_similarity = 1.0 - entropy(compute_histogram(raw_data), compute_histogram(decrypted_add) + 1) / np.log(2)
    results.append(("Addition", best_add_key, add_similarity, None, None))
    
    best_sub_key, best_sub_count = mod_results["subtraction"][0]
    decrypted_sub = (encrypted_data + best_sub_key) % 256
    sub_similarity = 1.0 - entropy(compute_histogram(raw_data), compute_histogram(decrypted_sub) + 1) / np.log(2)
    results.append(("Subtraction", best_sub_key, sub_similarity, None, None))
    
    forward_mapping = test_byte_mapping(raw_data, encrypted_data, output_dir)
    inverse_mapping = {v: k for k, v in forward_mapping.items()}
    decrypted_mapping = decrypt_with_mapping(encrypted_data, inverse_mapping)
    mapping_similarity = 1.0 - entropy(compute_histogram(raw_data), compute_histogram(decrypted_mapping) + 1) / np.log(2)
    results.append(("Mapping", len(inverse_mapping), mapping_similarity, None, None))
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    raw_hist = compute_histogram(raw_data)
    
    for method, key, similarity, decrypted_file, _ in results:
        decrypted_data = None
        if decrypted_file:
            decrypted_data = read_raw_audio(decrypted_file)
        elif method == "Addition":
            decrypted_data = (encrypted_data - key) % 256
        elif method == "Subtraction":
            decrypted_data = (encrypted_data + key) % 256
        elif method == "Mapping":
            decrypted_data = decrypt_with_mapping