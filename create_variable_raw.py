import random
import os
import datetime

def flip_bits_in_file(input_file, output_file, flip_percentage=0.01):
    """
    Reads a raw audio file, flips bits randomly, and saves the modified data.

    Args:
        input_file (str): Path to the input raw audio file.
        output_file (str): Path to save the modified raw audio file.
        flip_percentage (float): Approximate percentage of bits to flip (0.0 to 1.0).
    """

    with open(input_file, 'rb') as f_in:
        data = bytearray(f_in.read())

    num_bytes_to_flip = int(len(data) * flip_percentage)
    
    for _ in range(num_bytes_to_flip):
        byte_index = random.randint(0, len(data) - 1)
        bit_index = random.randint(0, 7)  # 8 bits in a byte

        # Flip the bit
        data[byte_index] ^= (1 << bit_index)

    with open(output_file, 'wb') as f_out:
        f_out.write(data)

def generate_random_flipped_files(input_file, output_dir, num_files, min_percentage=0.001, max_percentage=0.1):
    """
    Generates multiple versions of the input file with random bit flips,
    using a random percentage within a given range.

    Args:
        input_file (str): Path to the input raw audio file.
        output_dir (str): Directory to save the generated files.
        num_files (int): Number of files to generate.
        min_percentage (float): Minimum percentage of bits to flip (0.0 to 1.0).
        max_percentage (float): Maximum percentage of bits to flip (0.0 to 1.0).
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(num_files):
        percentage = random.uniform(min_percentage, max_percentage)  # Generate random percentage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_filename}_flipped_{percentage:.4f}_{i+1:03d}_{timestamp}.raw"
        output_file = os.path.join(output_dir, output_filename)
        flip_bits_in_file(input_file, output_file, percentage)
        print(f"Generated: {output_file}")

# --- Example Usage ---
input_file = "right.raw"
output_dir = "modified_raw"

num_files = 5000  # Generate 50 files with random percentages
min_percentage = 0.001  # Minimum percentage to flip
max_percentage = 0.1    # Maximum percentage to flip

generate_random_flipped_files(input_file, output_dir, num_files, min_percentage, max_percentage)