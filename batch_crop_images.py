from gimpfu import *
import os

def crop_image(image_path, output_path, x1, y1, x2, y2):
    # Load the image
    image = pdb.gimp_file_load(image_path, image_path)
    drawable = image.active_layer
    # Define the crop rectangle (x1, y1, width, height)
    width = x2 - x1
    height = y2 - y1
    # Crop the image
    pdb.gimp_image_crop(image, width, height, x1, y1)
    # Save the cropped image
    pdb.file_png_save(image, drawable, output_path, output_path, 0, 9, 0, 0, 0, 0, 0)
    # Close the image
    pdb.gimp_image_delete(image)

def batch_crop_images(input_dir, output_dir, x1, y1, x2, y2):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith("stft_comparison.png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            crop_image(input_path, output_path, x1, y1, x2, y2)

# Register the function with GIMP
register(
    "python_fu_batch_crop_images",
    "Batch Crop Images",
    "Batch crop images to specified coordinates",
    "Ryan",
    "Ryan",
    "2025",
    "<Image>/Filters/Python-Fu/Batch Crop Images...",
    "*",
    [
        (PF_DIRNAME, "input_dir", "Input Directory", ""),
        (PF_DIRNAME, "output_dir", "Output Directory", "cropped_stft"),
        (PF_INT, "x1", "X1 Coordinate", 80),
        (PF_INT, "y1", "Y1 Coordinate", 512),
        (PF_INT, "x2", "X2 Coordinate", 965),
        (PF_INT, "y2", "Y2 Coordinate", 520),
    ],
    [],
    batch_crop_images,
    menu="<Image>/Filters/Python-Fu"
)

main()