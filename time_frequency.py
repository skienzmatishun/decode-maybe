import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def read_binary_file(file_path):
    """Reads a binary file and returns its content as a numpy array of bytes"""
    with open(file_path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def compute_bigram_frequencies(data):
    """Computes bigram frequency distribution"""
    bigrams = [tuple(data[i:i+2]) for i in range(len(data)-1)]
    unique, counts = np.unique(bigrams, axis=0, return_counts=True)
    freq_matrix = np.zeros((256, 256), dtype=np.float32)
    
    for (a, b), count in zip(unique, counts):
        freq_matrix[a][b] = count / len(bigrams)  # Normalize to probability
    
    return freq_matrix

def plot_heatmap(matrix, title, ax, cmap='viridis'):
    """Plots a heatmap on the given axes"""
    im = ax.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=np.max(matrix))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('Second Byte')
    ax.set_ylabel('First Byte')
    ax.set_xticks(np.arange(0, 256, 32))
    ax.set_yticks(np.arange(0, 256, 32))

def compare_heatmaps(raw_path, encrypted_path, output_filename='heatmap_comparison.png'):
    """Generates and saves heatmaps comparing raw and encrypted files"""
    # Read and truncate data to same length
    raw_data = read_binary_file(raw_path)
    encrypted_data = read_binary_file(encrypted_path)
    min_len = min(len(raw_data), len(encrypted_data))
    raw_data = raw_data[:min_len]
    encrypted_data = encrypted_data[:min_len]

    # Compute bigram frequencies
    raw_freq = compute_bigram_frequencies(raw_data)
    encrypted_freq = compute_bigram_frequencies(encrypted_data)

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot individual heatmaps
    plot_heatmap(raw_freq, 'Raw Audio Bigram Heatmap', axes[0])
    plot_heatmap(encrypted_freq, 'Encrypted Audio Bigram Heatmap', axes[1])

    # Plot difference heatmap
    diff = np.abs(raw_freq - encrypted_freq)
    plot_heatmap(diff, 'Difference Heatmap', axes[2], cmap='plasma')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    raw_path = 'left.raw'
    modified_folder = './modified_raw'
    encrypted_files = glob.glob(os.path.join(modified_folder, '*.raw'))
    
    # Create output directory if it doesn't exist
    output_directory = './time_freq_png'
    os.makedirs(output_directory, exist_ok=True)
    
    for encrypted_file in encrypted_files:
        base_name = os.path.splitext(os.path.basename(encrypted_file))[0]
        output_filename = os.path.join(output_directory, f"{base_name}_comparison.png")
        compare_heatmaps(raw_path, encrypted_file, output_filename)