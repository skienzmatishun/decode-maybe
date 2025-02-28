import os
import json

def list_python_files(path):
    """
    Lists all `.py` files in the given directory.

    Args:
    path (str): The path to the directory.
    """

    python_files = {}

    # Iterate over each item in the directory
    for item in os.listdir(path):
        # Construct the full path to the item
        item_path = os.path.join(path, item)

        # Check if the item is a Python file
        if os.path.isfile(item_path) and item.endswith(".py"):
            with open(item_path, "r") as file:
                python_files[item] = file.read()

    return python_files

def main():
    project_goal_text = """Project Goal
    The project aims to reverse engineer an encryption method used on audio files. We are given:
    A hex frequency graph of a raw audio sample (clear peaks at 00h and FFh).
    A hex frequency graph of the encrypted audio (gap at 0Dh, peak at 9Dh).
    The goal is to:
        Determine the transformation applied to encrypt the audio.
        Develop a decryption method to revert the encrypted audio to its original state.
        Validate the method by comparing frequency distributions and restoring the audio's clear drumbeat.
        Apply the method to a larger dataset of encrypted audio files.

    We have left.raw, which is unencryped. right.raw is encrypted."""

    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current Directory: {current_dir}")

    # List the `.py` files in the current directory
    python_files = list_python_files(current_dir)

    # Add the project goal to the dictionary
    output_data = {
        "project_description": project_goal_text,
        "python_files": python_files
    }

    # Write the data to a .txt file in JSON format
    with open("python_files.txt", "w") as json_file:
        json.dump(output_data, json_file, indent=4)

if __name__ == "__main__":
    main()