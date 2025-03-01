import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
from handle_raw_audio import read_raw_audio
from dotenv import load_dotenv
import glob

load_dotenv()  # Load environment variables from .env file

# Constants
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
OUTPUT_DIR = "./time_freq_analysis"
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))  # Ensure SAMPLE_RATE is an integer

# CQT parameters
CQT_BINS = 84
HOP_LENGTH = 512
BINS_PER_OCTAVE = 12

# Mel parameters
N_MELS = 128

def compute_cqt(audio):
    return librosa.amplitude_to_db(
        np.abs(librosa.cqt(
            audio,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=CQT_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        )),
        ref=np.max
    )

def compute_mel_spectrogram(audio):
    return librosa.amplitude_to_db(
        librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH
        ),
        ref=np.max
    )

def extract_temporal_features(audio):
    return {
        'zcr': librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=HOP_LENGTH)[0],
        'rms': librosa.feature.rms(y=audio, frame_length=2048, hop_length=HOP_LENGTH)[0]
    }

def plot_comparison(raw_feat, enc_feat, feat_name, filename, y_axis_type):
    fig, ax = plt.subplots(3, 1, figsize=(16, 18))  # Increased figure size
    
    # Modified string formatting
    ax[0].set_title('Raw Audio {}'.format(feat_name))
    librosa.display.specshow(raw_feat, 
                           sr=SAMPLE_RATE, 
                           hop_length=HOP_LENGTH,
                           x_axis='time',
                           y_axis=y_axis_type,
                           ax=ax[0])
    
    ax[1].set_title('Encrypted Audio {}'.format(feat_name))
    librosa.display.specshow(enc_feat, 
                           sr=SAMPLE_RATE, 
                           hop_length=HOP_LENGTH,
                           x_axis='time',
                           y_axis=y_axis_type,
                           ax=ax[1])
    
    ax[2].set_title('Difference ({})'.format(feat_name))
    diff = raw_feat - enc_feat
    im = librosa.display.specshow(diff, 
                                sr=SAMPLE_RATE, 
                                hop_length=HOP_LENGTH,
                                x_axis='time',
                                y_axis=y_axis_type,
                                ax=ax[2])
    
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
    plt.savefig(filename)
    plt.close()

def analyze_audio(raw_audio, enc_audio, basename):
    file_output_dir = os.path.join(OUTPUT_DIR, basename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # CQT Analysis
    raw_cqt = compute_cqt(raw_audio)
    enc_cqt = compute_cqt(enc_audio)
    plot_comparison(raw_cqt, enc_cqt, 'CQT', 
                    os.path.join(file_output_dir, '{}_cqt_comparison.png'.format(basename)),
                    y_axis_type='cqt_note')
    
    # Mel-Spectrogram Analysis
    raw_mel = compute_mel_spectrogram(raw_audio)
    enc_mel = compute_mel_spectrogram(enc_audio)
    plot_comparison(raw_mel, enc_mel, 'Mel Spectrogram',
                    os.path.join(file_output_dir, '{}_mel_comparison.png'.format(basename)),
                    y_axis_type='mel')
    
    # Temporal Features
    raw_temp = extract_temporal_features(raw_audio)
    enc_temp = extract_temporal_features(enc_audio)
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    times = librosa.times_like(raw_temp['zcr'], sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    
    ax[0].plot(times, raw_temp['zcr'], label='Raw')
    ax[0].plot(times, enc_temp['zcr'], label='Encrypted')
    ax[0].set_title('Zero Crossing Rate')
    ax[0].legend()
    
    ax[1].plot(times, raw_temp['rms'], label='Raw')
    ax[1].plot(times, enc_temp['rms'], label='Encrypted')
    ax[1].set_title('RMS Energy')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(file_output_dir, '{}_temporal_features.png'.format(basename)))
    plt.close()
    
    pd.DataFrame({
        'time': times,
        'raw_zcr': raw_temp['zcr'],
        'enc_zcr': enc_temp['zcr'],
        'raw_rms': raw_temp['rms'],
        'enc_rms': enc_temp['rms']
    }).to_csv(os.path.join(file_output_dir, '{}_temporal_features.csv'.format(basename)), index=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_audio = read_raw_audio(RAW_AUDIO_PATH)
    
    # Process each encrypted RAW file
    encrypted_files = glob.glob(os.path.join(ENCRYPTED_DIR, "*.raw"))
    if not encrypted_files:
        print("No encrypted files found in {}".format(ENCRYPTED_DIR))
    else:
        for encrypted_file in encrypted_files:
            basename = os.path.splitext(os.path.basename(encrypted_file))[0]
            filepath = os.path.join(ENCRYPTED_DIR, encrypted_file)
            enc_audio = read_raw_audio(filepath)
            min_len = min(len(raw_audio), len(enc_audio))
            analyze_audio(raw_audio[:min_len], enc_audio[:min_len], basename)
    
    # Process each decrypted RAW file from each directory in DECRYPTED_DIRS
    for decrypted_dir in DECRYPTED_DIRS:
        print(f"Checking directory: {decrypted_dir}")
        decrypted_files = glob.glob(os.path.join(decrypted_dir, "*.raw"))
        
        if not decrypted_files:
            print(f"No decrypted files found in {decrypted_dir}")
            continue
        
        for decrypted_file in decrypted_files:
            basename = os.path.splitext(os.path.basename(decrypted_file))[0]
            filepath = os.path.join(decrypted_dir, decrypted_file)
            dec_audio = read_raw_audio(filepath)
            min_len = min(len(raw_audio), len(dec_audio))
            analyze_audio(raw_audio[:min_len], dec_audio[:min_len], basename)

    print("Analysis complete. Results saved to: {}".format(OUTPUT_DIR))

if __name__ == "__main__":
    main()