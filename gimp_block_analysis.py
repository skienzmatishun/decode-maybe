from gimpfu import *
import os
import numpy as np
from collections import defaultdict

def analyze_block_patterns(image_path, output_path):
    """Analyze spectrograms for block-based encryption patterns"""
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    
    width = drawable.width
    height = drawable.height
    
    # Define block sizes to test
    block_sizes = [2, 4, 8, 16, 32]
    block_patterns = {}
    
    for block_size in block_sizes:
        patterns = defaultdict(int)
        
        # Analyze blocks for patterns
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                # Extract block pixel values
                block_signature = []
                for by in range(block_size):
                    for bx in range(block_size):
                        if by == 0 and bx < 4:  # Sample first row only to keep signature manageable
                            pixel = pdb.gimp_drawable_get_pixel(drawable, x + bx, y + by)[1][0]  # Red channel
                            block_signature.append(pixel)
                
                # Hash the block signature
                block_hash = tuple(block_signature)
                patterns[block_hash] += 1
        
        # Save the most common patterns for this block size
        block_patterns[block_size] = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Create result image - visualization of likely block sizes
    result_img = pdb.gimp_image_new(512, 512, RGB)
    result_layer = pdb.gimp_layer_new(result_img, 512, 512, RGB_IMAGE, "Block Analysis", 100, NORMAL_MODE)
    pdb.gimp_image_add_layer(result_img, result_layer, 0)
    pdb.gimp_context_set_foreground((0, 0, 0))
    pdb.gimp_drawable_fill(result_layer, FOREGROUND_FILL)
    
    # Visualize block pattern frequency for each block size
    y_offset = 10
    for block_size, patterns in block_patterns.items():
        # Skip if no patterns found
        if not patterns:
            continue
            
        # Draw block size label
        pdb.gimp_context_set_foreground((255, 255, 255))
        text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_offset, f"Block size: {block_size}", 0, TRUE, 14, PIXELS, "Sans")
        pdb.gimp_floating_sel_anchor(text_layer)
        y_offset += 25
        
        # Calculate how common the top patterns are
        max_count = patterns[0][1] if patterns else 0
        
        # Draw histogram bars for top patterns
        for i, (pattern, count) in enumerate(patterns):
            bar_height = 20
            bar_width = int((count / max_count) * 400)
            
            # Color based on pattern prevalence
            intensity = int((count / max_count) * 255)
            pdb.gimp_context_set_foreground((0, intensity, 255 - intensity))
            
            # Draw bar
            pdb.gimp_image_select_rectangle(result_img, CHANNEL_OP_REPLACE, 50, y_offset, bar_width, bar_height)
            pdb.gimp_edit_fill(result_layer, FOREGROUND_FILL)
            pdb.gimp_selection_none(result_img)
            
            # Draw pattern values
            pattern_str = " ".join([f"{p:02x}" for p in pattern[:4]])
            pdb.gimp_context_set_foreground((255, 255, 255))
            text_layer = pdb.gimp_text_fontname(result_img, None, 460, y_offset+2, f"Count: {count}", 0, TRUE, 12, PIXELS, "Sans")
            pdb.gimp_floating_sel_anchor(text_layer)
            
            text_layer = pdb.gimp_text_fontname(result_img, None, 55, y_offset+2, pattern_str, 0, TRUE, 12, PIXELS, "Sans")
            pdb.gimp_floating_sel_anchor(text_layer)
            
            y_offset += bar_height + 5
        
        y_offset += 20
    
    # Create pattern grid visualization
    best_block_size = max(block_patterns.items(), key=lambda x: x[1][0][1] if x[1] else 0)[0] if block_patterns else 8
    
    # Create grid visualization using best block size
    y_offset += 20
    text_layer = pdb.gimp_text_fontname(result_img, None, 10, y_offset, f"Block Pattern Grid (size {best_block_size})", 0, TRUE, 14, PIXELS, "Sans")
    pdb.gimp_floating_sel_anchor(text_layer)
    y_offset += 25
    
    # Create block pattern grid visualization
    grid_width = width // best_block_size
    grid_height = height // best_block_size
    grid_scale = min(400 // grid_width, 200 // grid_height)
    
    # Create grid of block patterns
    pattern_map = np.zeros((grid_height, grid_width), dtype=int)
    pattern_dict = {}
    next_pattern_id = 1
    
    for y in range(0, height - best_block_size + 1, best_block_size):
        for x in range(0, width - best_block_size + 1, best_block_size):
            # Extract block signature
            block_signature = []
            for by in range(block_size):
                for bx in range(block_size):
                    if by == 0 and bx < 4:
                        pixel = pdb.gimp_drawable_get_pixel(drawable, x + bx, y + by)[1][0]
                        block_signature.append(pixel)
            
            block_hash = tuple(block_signature)
            
            # Assign pattern ID
            if block_hash not in pattern_dict:
                pattern_dict[block_hash] = next_pattern_id
                next_pattern_id += 1
            
            pattern_map[y // best_block_size, x // best_block_size] = pattern_dict[block_hash]
    
    # Draw the pattern grid
    num_patterns = len(pattern_dict)
    for gy in range(grid_height):
        for gx in range(grid_width):
            pattern_id = pattern_map[gy, gx]
            
            # Color based on pattern ID
            if num_patterns > 1:
                hue = (pattern_id / num_patterns) * 360
                # Convert HSV to RGB (simplified)
                if hue < 60:
                    r, g, b = 255, int((hue / 60) * 255), 0
                elif hue < 120:
                    r, g, b = int(((120 - hue) / 60) * 255), 255, 0
                elif hue < 180:
                    r, g, b = 0, 255, int(((hue - 120) / 60) * 255)
                elif hue < 240:
                    r, g, b = 0, int(((240 - hue) / 60) * 255), 255
                elif hue < 300:
                    r, g, b = int(((hue - 240) / 60) * 255), 0, 255
                else:
                    r, g, b = 255, 0, int(((360 - hue) / 60) * 255)
            else:
                r, g, b = 128, 128, 128
            
            pdb.gimp_context_set_foreground((r, g, b))
            pdb.gimp_image_select_rectangle(result_img, CHANNEL_OP_REPLACE, 
                                          50 + gx * grid_scale, 
                                          y_offset + gy * grid_scale, 
                                          grid_scale, grid_scale)
            pdb.gimp_edit_fill(result_layer, FOREGROUND_FILL)
    
    pdb.gimp_selection_none(result_img)
    
    # Save analysis results
    pdb.file_png_save(result_img, result_layer, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    
    # Save text analysis
    with open(output_path.replace('.png', '.txt'), 'w') as f:
        f.write("Block Pattern Analysis Results\n")
        f.write("===========================\n\n")
        
        f.write(f"Best block size candidate: {best_block_size}\n\n")
        
        for block_size, patterns in block_patterns.items():
            f.write(f"Block size {block_size}:\n")
            for i, (pattern, count) in enumerate(patterns):
                f.write(f"  Pattern {i+1}: Values: {' '.join([f'{p:02x}' for p in pattern[:4]])}, Count: {count}\n")
            f.write("\n")
        
        f.write("\nPotential encryption insights:\n")
        
        # Analyze potential block cipher behavior
        pattern_diversity = len(pattern_dict) / (grid_width * grid_height)
        f.write(f"Pattern diversity: {pattern_diversity:.2f}\n")
        if pattern_diversity > 0.8:
            f.write("High pattern diversity suggests possible CBC mode or stream cipher\n")
        elif pattern_diversity < 0.2:
            f.write("Low pattern diversity suggests possible ECB mode block cipher\n")
        else:
            f.write("Medium pattern diversity could indicate structured data encryption\n")
    
    pdb.gimp_image_delete(image)
    pdb.gimp_image_delete(result_img)

def batch_process_images(input_dir, output_dir):
    """Process all spectrograms in directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".png", ".jpg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_blocks.png")
            
            analyze_block_patterns(input_path, output_path)

register(
    "python_fu_audio_encryption_block_analyzer",
    "Analyze audio spectrograms for block-based encryption patterns",
    "Identify block size and patterns in encrypted spectrograms",
    "AudioCrypt", "AudioCrypt", "2025",
    "<Image>/Filters/Python-Fu/Audio Encryption Block Analyzer...",
    "*",
    [
        (PF_DIRNAME, "input_dir", "Spectrogram Directory", ""),
        (PF_DIRNAME, "output_dir", "Output Directory", "block_analysis"),
    ],
    [],
    batch_process_images,
    menu="<Image>/Filters/Python-Fu"
)

main()