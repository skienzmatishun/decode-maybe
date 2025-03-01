from gimpfu import *
import os
import numpy as np

def analyze_frequency_cipher(image_path, output_path):
    """Analyze spectrograms for frequency band shift patterns"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    width = drawable.width
    height = drawable.height
    
    # Create histogram bins for each frequency band
    num_bands = 16
    band_height = height // num_bands
    band_histograms = []
    band_offsets = []
    
    for band in range(num_bands):
        y_start = band * band_height
        y_end = (band + 1) * band_height
        
        # Create histogram for this band
        histogram = np.zeros(256, dtype=int)
        
        # Sample pixels in this frequency band
        for y in range(y_start, min(y_end, height)):
            for x in range(0, width, 4):  # Sample every 4th pixel for efficiency
                pixel = pdb.gimp_drawable_get_pixel(drawable, x, y)[1][0]  # Red channel
                histogram[pixel] += 1
        
        # Find potential circular shift offset using autocorrelation
        shifts = np.zeros(256, dtype=float)
        for shift in range(256):
            shifted = np.roll(histogram, shift)
            correlation = np.sum(histogram * shifted)
            shifts[shift] = correlation
        
        # Normalize and find best offset (excluding zero shift)
        shifts[0] = 0  # Ignore zero shift
        best_shift = np.argmax(shifts)
        
        band_histograms.append(histogram)
        band_offsets.append(best_shift)
    
    # Create visualization image
    result_img = pdb.gimp_image_new(800, 600, RGB)
    result_layer = pdb.gimp_layer_new(result_img, 800, 600, RGB_IMAGE, "Frequency Analysis", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(result_img, result_layer, 0)
    pdb.gimp_context_set_foreground((0, 0, 0))
    pdb.gimp_drawable_fill(result_layer, FOREGROUND_FILL)
    
    # Draw band shift visualization
    pdb.gimp_context_set_foreground((255, 255, 255))
    text_layer = pdb.gimp_text_fontname(result_img, None, 10, 10, "Frequency Band Shift Analysis", 0, True, 16, PIXELS, "Sans")
    pdb.gimp_floating_sel_anchor(text_layer)
    
    # Draw histogram visualizations
    for band in range(num_bands):
        # Draw band label and offset
        y_pos = 40 + band * 30
        text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_pos, 
                                           "Band {} : Offset {} (0x{:02x})".format(band, band_offsets[band], band_offsets[band]), 
                                           0, True, 12, PIXELS, "Sans")
        pdb.gimp_floating_sel_anchor(text_layer)
        
        # Draw mini histogram
        max_count = max(band_histograms[band]) if max(band_histograms[band]) > 0 else 1
        for x in range(256):
            height = int((band_histograms[band][x] / max_count) * 20)
            if height > 0:
                # Color based on value
                hue = (x / 256) * 360
                if hue < 120:
                    r, g, b = 255, int((hue / 120) * 255), 0
                elif hue < 240:
                    r, g, b = int(((240 - hue) / 120) * 255), 255, int(((hue - 120) / 120) * 255)
                else:
                    r, g, b = 0, int(((360 - hue) / 120) * 255), 255
                
                pdb.gimp_context_set_foreground((r, g, b))
                pdb.gimp_image_select_rectangle(result_img, CHANNEL_OP_REPLACE, 
                                              200 + x * 2, y_pos, 2, height)
                pdb.gimp_edit_fill(result_layer, FOREGROUND_FILL)
    
    pdb.gimp_selection_none(result_img)
    
    # Draw key analysis
    y_pos = 40 + num_bands * 30 + 20
    text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_pos, "Potential Encryption Keys:", 0, True, 14, PIXELS, "Sans")
    pdb.gimp_floating_sel_anchor(text_layer)
    
    # Analyze offset patterns
    y_pos += 25
    offsets = np.array(band_offsets)
    
    # Check for repeating offset pattern
    key_patterns = []
    for pattern_len in range(1, min(8, num_bands // 2)):
        matches = 0
        for i in range(num_bands - pattern_len):
            if np.array_equal(offsets[i:i+pattern_len], offsets[i+pattern_len:i+2*pattern_len]):
                matches += 1
        
        if matches > 0:
            key_patterns.append((pattern_len, matches))
    
    # Generate key sequences from offsets
    for pattern_len, matches in sorted(key_patterns, key=lambda x: x[1], reverse=True):
        pattern = offsets[:pattern_len]
        text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_pos, 
                                           "Key length: {}, Confidence: {}".format(pattern_len, matches), 
                                           0, True, 12, PIXELS, "Sans")
        pdb.gimp_floating_sel_anchor(text_layer)
        y_pos += 20
        
        text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_pos, 
                                           "Key bytes: {}".format(' '.join("{:02x}".format(x) for x in pattern)), 
                                           0, True, 12, PIXELS, "Sans")
        pdb.gimp_floating_sel_anchor(text_layer)
        y_pos += 30
    
    # If no patterns found, try constant offset
    if not key_patterns:
        # Check if all offsets are similar (within 10%)
        mean_offset = np.mean(offsets)
        if np.all(np.abs(offsets - mean_offset) < 0.1 * mean_offset):
            text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_pos, 
                                              "Constant offset: {} (0x{:02x})".format(int(mean_offset), int(mean_offset)), 
                                              0, True, 12, PIXELS, "Sans")
            pdb.gimp_floating_sel_anchor(text_layer)
            y_pos += 30
    
    # Draw correlation matrix to detect frequency band relationships
    matrix_size = 256
    correlation_img = pdb.gimp_image_new(matrix_size, matrix_size, RGB)
    correlation_layer = pdb.gimp_layer_new(correlation_img, matrix_size, matrix_size, RGB_IMAGE, "Correlation Matrix", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(correlation_img, correlation_layer, 0)
    pdb.gimp_context_set_foreground((0, 0, 0))
    pdb.gimp_drawable_fill(correlation_layer, FOREGROUND_FILL)
    
    # Calculate correlation between original value (x) and encrypted value (y)
    cross_correlation = np.zeros((256, 256), dtype=int)
    for y in range(0, height, 4):
        for x in range(0, width-4, 4):
            original = pdb.gimp_drawable_get_pixel(drawable, x, y)[1][0]
            encrypted = pdb.gimp_drawable_get_pixel(drawable, x+4, y)[1][0]
            cross_correlation[original, encrypted] += 1
    
    # Normalize and draw
    max_correlation = np.max(cross_correlation) if np.max(cross_correlation) > 0 else 1
    for y in range(256):
        for x in range(256):
            if cross_correlation[x, y] > 0:
                intensity = int((cross_correlation[x, y] / max_correlation) * 255)
                pdb.gimp_drawable_set_pixel(correlation_layer, x, y, 3, (0, intensity, 0))
    
    # Save correlation matrix
    corr_path = output_path.replace('.png', '_correlation.png')
    pdb.file_png_save(correlation_img, correlation_layer, corr_path, corr_path, 0, 9, 0, 0, 0, 0, 0)
    
    # Save results
    pdb.file_png_save(result_img, result_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    
    # Save binary key file for testing
    key_file = output_path.replace('.png', '.key')
    with open(key_file, 'wb') as f:
        for offset in band_offsets:
            f.write(bytes([offset]))
    
    # Save detailed analysis text
    with open(output_path.replace('.png', '.txt'), 'w') as f:
        f.write("Frequency Band Encryption Analysis\n")
        f.write("================================\n\n")
        
        f.write("Band Offsets:\n")
        for i, offset in enumerate(band_offsets):
            f.write("Band {}: {} (0x{:02x})\n".format(i, offset, offset))
        
        f.write("\nPotential Key Patterns:\n")
        if key_patterns:
            for pattern_len, matches in sorted(key_patterns, key=lambda x: x[1], reverse=True):
                pattern = offsets[:pattern_len]
                with open("output.txt", "w") as f:
                    f.write("Key length: {}, Confidence: {}\n".format(pattern_len, matches))
                    f.write("Key bytes: {}\n\n".format(' '.join(f'{x:02x}' for x in pattern)))
        else:
            f.write("No clear repeating patterns found\n")
            
            # Check if all offsets are similar (within 10%)
            mean_offset = np.mean(offsets)
            if np.all(np.abs(offsets - mean_offset) < 0.1 * mean_offset):
                f.write("Constant offset: {} (0x{:02x})\n".format(int(mean_offset), int(mean_offset)))
    
    pdb.gimp_image_delete(image)
    pdb.gimp_image_delete(result_img)
    pdb.gimp_image_delete(correlation_img)

def batch_analyze_images(input_dir, output_dir):
    """Process all spectrograms in directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_frequency.png")
            
            analyze_frequency_cipher(input_path, output_path)

register(
    "python_fu_audio_frequency_cipher_analyzer",
    "Analyze spectrograms for frequency-band encryption patterns",
    "Identify potential encryption keys based on frequency-dependent patterns",
    "AudioCrypt", "AudioCrypt", "2025",
    "<Image>/Filters/Python-Fu/Audio Frequency Cipher Analyzer...",
    "*",
    [
        (PF_DIRNAME, "input_dir", "Spectrogram Directory", ""),
        (PF_DIRNAME, "output_dir", "Output Directory", "frequency_analysis"),
    ],
    [],
    batch_analyze_images,
    menu="<Image>/Filters/Python-Fu"
)

main()