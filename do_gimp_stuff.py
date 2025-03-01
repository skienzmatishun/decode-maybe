from gimpfu import *
import os
import numpy as np

def extract_color_patterns(image_path, output_file):
    """Extract color patterns from spectrograms for encryption pattern analysis"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    # Get image dimensions
    width = drawable.width
    height = drawable.height
    
    # Sample areas of interest (frequency bands)
    bands = [(0, height//4), (height//4, height//2), (height//2, 3*height//4), (3*height//4, height)]
    results = []
    
    for band_start, band_end in bands:
        band_data = []
        for x in range(0, width, 2):  # Sample every other pixel for efficiency
            for y in range(band_start, band_end, 2):
                # Get pixel color
                pixel = pdb.gimp_drawable_get_pixel(drawable, x, y)
                band_data.append(pixel[1])  # Use color values
        
        # Calculate statistics for this frequency band
        band_avg = np.mean(band_data)
        band_std = np.std(band_data)
        band_entropy = entropy(band_data)
        results.append((band_avg, band_std, band_entropy))
    
    # Write results to a file
    with open(output_file, 'w') as f:
        f.write("Band,Average,StdDev,Entropy\n")
        for i, (avg, std, ent) in enumerate(results):
            f.write(f"Band{i},{avg:.2f},{std:.2f},{ent:.4f}\n")
    
    pdb.gimp_image_delete(image)

def entropy(data):
    """Calculate Shannon entropy of data"""
    values, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    return -np.sum(probs * np.log2(probs + 1e-10))

def batch_analyze_spectrograms(input_dir, output_dir):
    """Process all spectrograms in a directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(("stft_comparison.png", "spectrogram.png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_analysis.csv")
            extract_color_patterns(input_path, output_path)
            
            # Create difference heatmap for encrypted vs original
            if "stft_comparison" in filename:
                original_path = os.path.join(input_dir, "left_spectrogram.png")
                if os.path.exists(original_path):
                    create_difference_heatmap(original_path, input_path, 
                                           os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_diff.png"))

def create_difference_heatmap(original_path, encrypted_path, output_path):
    """Create heatmap showing differences between spectrograms"""
    orig_img = pdb.gimp_file_load(original_path, original_path)
    enc_img = pdb.gimp_file_load(encrypted_path, encrypted_path)
    
    # Create new image for difference
    width = min(orig_img.width, enc_img.width)
    height = min(orig_img.height, enc_img.height)
    diff_img = pdb.gimp_image_new(width, height, RGB)
    diff_layer = pdb.gimp_layer_new(diff_img, width, height, RGB_IMAGE, "Difference", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(diff_img, diff_layer, 0)
    
    # Calculate differences pixel by pixel
    for x in range(width):
        for y in range(height):
            orig_pixel = pdb.gimp_drawable_get_pixel(orig_img.active_layer, x, y)
            enc_pixel = pdb.gimp_drawable_get_pixel(enc_img.active_layer, x, y)
            
            # Calculate color difference and map to heatmap (red = high difference)
            diff = sum([abs(orig_pixel[1][i] - enc_pixel[1][i]) for i in range(3)]) / 3
            diff_color = (min(255, int(diff * 2)), max(0, 255 - int(diff * 2)), 0)  # Red-green gradient
            
            pdb.gimp_drawable_set_pixel(diff_layer, x, y, 3, [diff_color[0], diff_color[1], diff_color[2]])
    
    # Save the result
    pdb.file_png_save(diff_img, diff_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    pdb.gimp_image_delete(orig_img)
    pdb.gimp_image_delete(enc_img)
    pdb.gimp_image_delete(diff_img)

register(
    "python_fu_audio_encryption_analysis",
    "Analyze audio spectrograms for encryption patterns",
    "Extract features from spectrograms to identify encryption patterns",
    "AudioEncrypt", "AudioEncrypt", "2025",
    "<Image>/Filters/Python-Fu/Audio Encryption Analysis...",
    "*",
    [
        (PF_DIRNAME, "input_dir", "Spectrogram Directory", ""),
        (PF_DIRNAME, "output_dir", "Output Directory", "spectrogram_analysis"),
    ],
    [],
    batch_analyze_spectrograms,
    menu="<Image>/Filters/Python-Fu"
)

main()