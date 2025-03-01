from gimpfu import *
import os
import numpy as np
from collections import Counter

def analyze_phase_patterns(image_path, output_path):
    """Analyze spectrogram for phase encryption patterns"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    width = drawable.width
    height = drawable.height
    
    # Extract color gradients for transition pattern detection
    transitions = []
    for x in range(width-1):
        for y in range(height-1):
            # Get 2x2 grid of pixels
            pixels = [
                pdb.gimp_drawable_get_pixel(drawable, x, y)[1],
                pdb.gimp_drawable_get_pixel(drawable, x+1, y)[1],
                pdb.gimp_drawable_get_pixel(drawable, x, y+1)[1],
                pdb.gimp_drawable_get_pixel(drawable, x+1, y+1)[1]
            ]
            
            # Calculate transitions between adjacent pixels
            for i in range(3):  # RGB values
                transitions.append((pixels[0][i] - pixels[1][i]) % 256)
                transitions.append((pixels[0][i] - pixels[2][i]) % 256)
    
    # Find most common transitions (possible encryption keys)
    counter = Counter(transitions)
    common_patterns = counter.most_common(10)
    
    # Generate pattern map
    pattern_img = pdb.gimp_image_new(256, 256, RGB)
    pattern_layer = pdb.gimp_layer_new(pattern_img, 256, 256, RGB_IMAGE, "Transition Pattern", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(pattern_img, pattern_layer, 0)
    
    # Fill with black
    pdb.gimp_context_set_foreground((0, 0, 0))
    pdb.gimp_drawable_fill(pattern_layer, FOREGROUND_FILL)
    
    # Plot transition frequency as heatmap
    for i in range(256):
        count = counter.get(i, 0)
        intensity = min(255, int(count * 255 / (counter.most_common(1)[0][1] + 1)))
        
        # Draw vertical line with intensity proportional to frequency
        for j in range(256):
            color = (intensity, 0, 255 - intensity)  # Purple to red gradient
            pdb.gimp_drawable_set_pixel(pattern_layer, i, j, 3, color)
    
    # Add markers for most common transitions
    for value, count in common_patterns:
        pdb.gimp_context_set_foreground((255, 255, 0))  # Yellow
        pdb.gimp_paintbrush_default(pattern_layer, 2, [value, 0, value, 255])
    
    # Save results
    pdb.file_png_save(pattern_img, pattern_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    
    # Create text file with most common transitions
    with open(output_path.replace('.png', '.txt'), 'w') as f:
        f.write("Potential encryption keys (byte difference values):\n")
        for value, count in common_patterns:
            f.write(f"Value: {value}, Frequency: {count}, Hex: 0x{value:02x}\n")
    
    pdb.gimp_image_delete(image)
    pdb.gimp_image_delete(pattern_img)

def create_transform_matrix(image_path, output_path):
    """Create transform matrix visualization from spectrogram"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    # Create byte transformation matrix (256x256)
    matrix_img = pdb.gimp_image_new(256, 256, RGB)
    matrix_layer = pdb.gimp_layer_new(matrix_img, 256, 256, RGB_IMAGE, "Transform Matrix", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(matrix_img, matrix_layer, 0)
    
    # Fill with black
    pdb.gimp_context_set_foreground((0, 0, 0))
    pdb.gimp_drawable_fill(matrix_layer, FOREGROUND_FILL)
    
    # Sample image to build frequency map
    transformations = {}
    width = drawable.width
    height = drawable.height
    
    # Get sample of pixel transitions
    for x in range(0, width-8, 8):
        for y in range(0, height-1, 1):
            # Get consecutive pixels
            p1 = pdb.gimp_drawable_get_pixel(drawable, x, y)[1][0]  # Use red channel
            p2 = pdb.gimp_drawable_get_pixel(drawable, x+8, y)[1][0]
            
            # Accumulate potential transformations
            key = (p1, p2)
            transformations[key] = transformations.get(key, 0) + 1
    
    # Find most likely transformations and plot
    max_freq = max(transformations.values()) if transformations else 1
    for (input_byte, output_byte), freq in transformations.items():
        # Color intensity based on frequency
        intensity = min(255, int(freq * 255 / max_freq))
        color = (0, intensity, 0)  # Green with varying intensity
        
        # Plot point in transform matrix
        pdb.gimp_drawable_set_pixel(matrix_layer, input_byte, output_byte, 3, color)
    
    # Save matrix visualization
    pdb.file_png_save(matrix_img, matrix_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    pdb.gimp_image_delete(image)
    pdb.gimp_image_delete(matrix_img)

def batch_process_spectrograms(input_dir, output_dir):
    """Process all spectrograms in directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg")):
            input_path = os.path.join(input_dir, filename)
            pattern_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pattern.png")
            matrix_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_matrix.png")
            
            analyze_phase_patterns(input_path, pattern_path)
            create_transform_matrix(input_path, matrix_path)

register(
    "python_fu_audio_encryption_pattern_detector",
    "Detect encryption patterns in spectrograms",
    "Analyze spectrograms to identify potential encryption keys and transformation patterns",
    "AudioCrypt", "AudioCrypt", "2025",
    "<Image>/Filters/Python-Fu/Audio Encryption Pattern Detector...",
    "*",
    [
        (PF_DIRNAME, "input_dir", "Spectrogram Directory", ""),
        (PF_DIRNAME, "output_dir", "Output Directory", "pattern_analysis"),
    ],
    [],
    batch_process_spectrograms,
    menu="<Image>/Filters/Python-Fu"
)

main()