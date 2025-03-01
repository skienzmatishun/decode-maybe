#!/usr/bin/env python3

import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pesq import pesq
from pystoi import stoi
from handle_raw_audio import read_raw_audio
import pandas as pd
from dotenv import load_dotenv
import glob

load_dotenv()  # Load environment variables from .env file

# Configuration
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
RESULTS_DIR = "./validation_results"
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))

def play_audio(audio, name):
    """Play audio with automatic gain control"""
    try:
        audio_int16 = np.int16(audio * (32767 / np.max(np.abs(audio))))
        print("\nPlaying {}...".format(name))
        sd.play(audio_int16, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print("Playback error: {}".format(str(e)))

def dynamic_time_warping(original, decrypted, basename):
    """Calculate DTW alignment between signals"""
    original_resampled = original[::16]
    decrypted_resampled = decrypted[::16]
    
    distance, path = fastdtw(original_resampled, decrypted_resampled, dist=euclidean)
    
    plt.figure(figsize=(10, 6))
    plt.plot([p[0] for p in path], [p[1] for p in path], 'r--', alpha=0.5)
    plt.title("DTW Alignment Path")
    plt.xlabel("Original Signal Index")
    plt.ylabel("Decrypted Signal Index")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "dtw_alignment_{}.png".format(basename)))
    plt.close()
    
    return distance / len(path)

def perceptual_evaluation(original, decrypted):
    """Calculate objective quality metrics"""
    min_len = min(len(original), len(decrypted))
    original = original[:min_len]
    decrypted = decrypted[:min_len]
    
    pesq_score = -1
    stoi_score = -1
    
    try:
        pesq_score = pesq(SAMPLE_RATE, original, decrypted, 'wb')
    except Exception as e:
        print("PESQ calculation error: {}".format(str(e)))
    
    try:
        stoi_score = stoi(original, decrypted, SAMPLE_RATE)
    except Exception as e:
        print("STOI calculation error: {}".format(str(e)))
    
    return pesq_score, stoi_score

def analyze_decrypted_file(raw_audio, decrypted_path, basename):
    file_results = {"filename": basename}
    
    decrypted_audio = read_raw_audio(decrypted_path)
    min_len = min(len(raw_audio), len(decrypted_audio))
    
    play_audio(raw_audio[:min_len], "Original Audio")
    play_audio(decrypted_audio[:min_len], "Decrypted Audio")
    
    dtw_score = dynamic_time_warping(raw_audio[:min_len], decrypted_audio[:min_len], basename)
    file_results["dtw_score"] = dtw_score
    
    pesq_score, stoi_score = perceptual_evaluation(raw_audio[:min_len], decrypted_audio[:min_len])
    file_results["pesq"] = pesq_score
    file_results["stoi"] = stoi_score
    
    return file_results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_audio = read_raw_audio(RAW_AUDIO_PATH)
    
    results = []
    
    for decrypted_dir in DECRYPTED_DIRS:
        decrypted_files = glob.glob(os.path.join(decrypted_dir, "*.raw"))
        
        if not decrypted_files:
            print(f"No decrypted files found in {decrypted_dir}")
            continue
        
        for decrypted_path in decrypted_files:
            basename = os.path.splitext(os.path.basename(decrypted_path))[0]
            print("\nAnalyzing {}...".format(decrypted_path))
            
            try:
                file_results = analyze_decrypted_file(raw_audio, decrypted_path, basename)
                results.append(file_results)
            except Exception as e:
                print("Error analyzing {}: {}".format(decrypted_path, str(e)))
    
    if not results:
        print("No valid decrypted files were processed.")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "validation_results.csv"), index=False)
    
    summary = {
        "average_dtw": df.dtw_score.mean(),
        "average_pesq": df.pesq.mean(),
        "average_stoi": df.stoi.mean(),
        "best_file": df.loc[df.stoi.idxmax(), 'filename']
    }
    
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write("{}: {}\n".format(k, v))
    
    print("\nValidation complete. Results saved to {}".format(RESULTS_DIR))

if __name__ == "__main__":
    main()