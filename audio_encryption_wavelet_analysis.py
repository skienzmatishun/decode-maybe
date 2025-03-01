from gimpfu import *
import os
import numpy as np
import struct

def extract_wavelet_features(image_path, output_path):
    """Extract wavelet-like features from spectrograms to identify encryption patterns"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    width = drawable.width
    height = drawable.height
    
    # Create output image for pattern analysis
    result_img = pdb.gimp_image_new(256, 256, RGB)
    result_layer = pdb.gimp_layer_new(result_img, 256, 256, RGB_IMAGE, "Encryption Analysis", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(result_img, result_layer, 0)
    
    # Initialize pattern arrays
    horizontal_diffs = np.zeros(256, dtype=int)
    vertical_diffs = np.zeros(256, dtype=int)
    xor_patterns = np.zeros((256, 256), dtype=int)
    
    # Sample the image at different scales (similar to wavelet analysis)
    scales = [1, 2, 4, 8, 16]
    
    for scale in scales:
        for y in range(0, height-scale, scale):
            for x in range(0, width-scale, scale):
                # Get pixels at current position and neighbors
                center = pdb.gimp_drawable_get_pixel(drawable, x, y)[1][0]  # Red channel
                right = pdb.gimp_drawable_get_pixel(drawable, x+scale, y)[1][0]
                bottom = pdb.gimp_drawable_get_pixel(drawable, x, y+scale)[1][0]
                
                # Calculate differences
                h_diff = (right - center) % 256
                v_diff = (bottom - center) % 256
                xor_val = center ^ right
                
                # Accumulate statistics
                horizontal_diffs[h_diff] += 1
                vertical_diffs[v_diff] += 1
                xor_patterns[center, xor_val] += 1
    
    # Find most common horizontal difference (potential encryption key)
    h_key = np.argmax(horizontal_diffs)
    v_key = np.argmax(vertical_diffs)
    
    # Create visualization of possible encryption patterns
    # X-axis: original value, Y-axis: encrypted value
    max_count = np.max(xor_patterns)
    for x in range(256):
        for y in range(256):
            count = xor_patterns[x, y]
            if count > 0:
                # Normalize and convert to color
                intensity = int(255 * count / max_count)
                color = (0, intensity, intensity)  # Cyan gradient
                pdb.gimp_drawable_set_pixel(result_layer, x, y, 3, color)
    
    # Mark the most likely encryption key values
    pdb.gimp_context_set_foreground((255, 0, 0))  # Red
    for i in range(256):
        if horizontal_diffs[i] > max(horizontal_diffs) * 0.7:
            # Draw vertical line for potential key
            pdb.gimp_paintbrush_default(result_layer, 2, [i, 0, i, 255])
    
    # Save visualization
    pdb.file_png_save(result_img, result_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    
    # Save potential key data
    with open(output_path.replace('.png', '.bin'), 'wb') as f:
        # Write most common differences as potential keys
        for i in range(10):
            idx = np.argpartition(horizontal_diffs, -i-1)[-i-1]
            if horizontal_diffs[idx] > max(horizontal_diffs) * 0.3:
                f.write(struct.pack('B', idx))
    
    # Save analysis text
    with open(output_path.replace('.png', '.txt'), 'w') as f:
        f.write(f"Most likely horizontal key: {h_key} (0x{h_key:02x})\n")
        f.write(f"Most likely vertical key: {v_key} (0x{v_key:02x})\n")
        
        # List top 5 most common horizontal differences
        f.write("\nTop horizontal differences:\n")
        top_h = np.argsort(horizontal_diffs)[-5:][::-1]
        for i, idx in enumerate(top_h):
            f.write(f"{i+1}. Value: {idx} (0x{idx:02x}), Count: {horizontal_diffs[idx]}\n")
        
        # Check for potential byte substitution
        f.write("\nPotential substitution cipher mappings:\n")
        for i in range(256):
            j = np.argmax(xor_patterns[i,:])
            confidence = xor_patterns[i,j] / sum(xor_patterns[i,:]) if sum(xor_patterns[i,:]) > 0 else 0
            if confidence > 0.5:
                f.write(f"Original: {i} (0x{i:02x}) → Encrypted: {j} (0x{j:02x}), Confidence: {confidence:.2f}\n")
    
    pdb.gimp_image_delete(image)
    pdb.gimp_image_delete(result_img)

def attempt_decrypt_preview(image_path, key_value, output_path):
    """Create a preview of decryption attempt using the specified key"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    width = drawable.width
    height = drawable.height
    
    # Create output image for decryption preview
    decrypted = pdb.gimp_image_duplicate(image)
    dec_layer = decrypted.active_layer
    
    # Apply potential decryption using XOR with key
    for y in range(height):
        for x in range(width):
            pixel = pdb.gimp_drawable_get_pixel(drawable, x, y)[1]
            # Apply XOR decryption to all channels
            new_pixel = [(p ^ key_value) % 256 for p in pixel]
            pdb.gimp_drawable_set_pixel(dec_layer, x, y, len(pixel), new_pixel)
    
    # Save decryption preview
    pdb.file_png_save(decrypted, dec_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    pdb.gimp_image_delete(image)
    pdb.gimp_image_delete(decrypted)

def batch_wavelet_analysis(input_dir, output_dir):
    """Process all spectrograms in directory with wavelet-like analysis"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_wavelet.png")
            
            # Run wavelet analysis
            extract_wavelet_features(input_path, output_path)
            
            # Read the key file if it was generated
            key_file = output_path.replace('.png', '.bin')
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    keys = list(f.read())
                    
                    # Create preview decryptions for top key
                    if keys:
                        decrypt_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_decrypted.png")
                        attempt_decrypt_preview(input_path, keys[0], decrypt_path)

register(
    "python_fu_audio_wavelet_encryption_analyzer",
    "Analyze spectrograms using wavelet-like techniques to identify encryption patterns",
    "Extracts multi-scale features to identify potential encryption keys and patterns",
    "AudioCrypt", "AudioCrypt", "2025",
    "<Image>/Filters/Python-Fu/Audio Wavelet Encryption Analyzer...",
    "*",
    [
        (PF_DIRNAME, "input_dir", "Spectrogram Directory", ""),
        (PF_DIRNAME, "output_dir", "Output Directory", "wavelet_analysis"),
    ],
    [],
    batch_wavelet_analysis,
    menu="<Image>/Filters/Python-Fu"
)

main()