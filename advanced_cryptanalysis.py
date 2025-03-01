# advanced_cryptanalysis.py (revised)
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
from tqdm import tqdm
from handle_raw_audio import read_raw_audio

# Add at the top of your script
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Update your configuration loading
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
REPORT_PATH = os.getenv("REPORT_PATH")
# Constants
RESULTS_DIR = "./cryptanalysis_results"
NUM_FLIP_TRIALS = 1000

def convert_to_uint8(audio_data):
    """Convert float audio data to uint8 format"""
    return np.clip((audio_data * 127.5 + 127.5), 0, 255).astype(np.uint8)

def differential_cryptanalysis():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load and convert data to uint8
    raw_data = convert_to_uint8(read_raw_audio(RAW_AUDIO_PATH))
    encrypted_data = convert_to_uint8(read_raw_audio(ENCRYPTED_DIR))
    
    delta_inputs = []
    delta_outputs = []
    
    modified_files = [f for f in os.listdir(DECRYPTED_DIRS) if f.endswith(".raw")]
    
    for filename in tqdm(modified_files, desc="Differential Analysis"):
        mod_data = convert_to_uint8(read_raw_audio(os.path.join(DECRYPTED_DIRS, filename)))
        
        # Align array lengths
        min_len = min(len(encrypted_data), len(mod_data))
        enc_subset = encrypted_data[:min_len]
        mod_subset = mod_data[:min_len]
        
        diff = np.bitwise_xor(enc_subset, mod_subset)
        changed_indices = np.where(diff != 0)[0]
        
        for idx in changed_indices:
            if idx < len(raw_data):
                delta_input = raw_data[idx] ^ mod_subset[idx]
                delta_output = enc_subset[idx] ^ mod_subset[idx]
                
                delta_inputs.append(delta_input)
                delta_outputs.append(delta_output)

    # Plotting code remains the same
    plt.figure(figsize=(10, 6))
    plt.scatter(delta_inputs, delta_outputs, alpha=0.5)
    plt.title("Differential Cryptanalysis: Input vs Output Changes")
    plt.xlabel("Input Δ (XOR)")
    plt.ylabel("Output Δ (XOR)")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "differential_analysis.png"))
    plt.close()

    return delta_inputs, delta_outputs

def linear_cryptanalysis():
    # Convert to uint8 for bitwise operations
    raw_data = convert_to_uint8(read_raw_audio(RAW_AUDIO_PATH))
    encrypted_data = convert_to_uint8(read_raw_audio(ENCRYPTED_DIR))
    
    min_len = min(len(raw_data), len(encrypted_data))
    raw_data = raw_data[:min_len]
    encrypted_data = encrypted_data[:min_len]
    
    correlation_matrix = np.zeros((8, 8))
    
    for input_bit in range(8):
        for output_bit in range(8):
            input_bits = (raw_data >> input_bit) & 1
            output_bits = (encrypted_data >> output_bit) & 1
            
            corr, _ = pearsonr(input_bits, output_bits)
            correlation_matrix[input_bit, output_bit] = corr

    # Plotting code remains the same
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label="Correlation Coefficient")
    plt.title("Linear Cryptanalysis: Bit-wise Correlations")
    plt.xlabel("Output Bit Position")
    plt.ylabel("Input Bit Position")
    plt.xticks(range(8))
    plt.yticks(range(8))
    plt.savefig(os.path.join(RESULTS_DIR, "linear_analysis.png"))
    plt.close()

    return correlation_matrix

def stream_cipher_detection():
    # Convert to uint8 for analysis
    encrypted_data = convert_to_uint8(read_raw_audio(ENCRYPTED_DIR))
    
    # Analysis code remains the same
    hist = np.bincount(encrypted_data, minlength=256)
    prob = hist / hist.sum()
    enc_entropy = entropy(prob, base=2)
    
    random_data = np.random.randint(0, 256, len(encrypted_data))
    random_hist = np.bincount(random_data, minlength=256)
    random_prob = random_hist / random_hist.sum()
    random_entropy = entropy(random_prob, base=2)

    autocorr = np.correlate(encrypted_data, encrypted_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Plotting code remains the same
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].bar(['Encrypted', 'Random'], [enc_entropy, random_entropy])
    axs[0].set_title("Entropy Comparison")
    axs[0].set_ylabel("Shannon Entropy (bits)")
    axs[1].plot(autocorr[:500])
    axs[1].set_title("Autocorrelation Analysis")
    axs[1].set_xlabel("Lag")
    axs[1].set_ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "stream_cipher_analysis.png"))
    plt.close()

    return {
        'encrypted_entropy': enc_entropy,
        'random_entropy': random_entropy,
        'autocorrelation': autocorr
    }

def main():
    print("Performing differential cryptanalysis...")
    delta_in, delta_out = differential_cryptanalysis()
    
    print("\nPerforming linear cryptanalysis...")
    lin_corr = linear_cryptanalysis()
    
    print("\nAnalyzing stream cipher properties...")
    stream_metrics = stream_cipher_detection()
    
    print("\nAnalysis complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()