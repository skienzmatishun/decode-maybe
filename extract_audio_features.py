import librosa
import numpy as np
import os

def extract_audio_features(audio_file, sr=22050):
    """
    Extracts MFCCs, Chroma features, Spectral Centroid, and Spectral Bandwidth from an audio file.

    Args:
        audio_file (str): Path to the audio file.
        sr (int): Sample rate of the audio file.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    try:
        y, _ = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return None

    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)   # Adjust n_mfcc as needed
    features['mfccs'] = np.mean(mfccs.T, axis=0)  # Average MFCCs over time

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma'] = np.mean(chroma.T, axis=0)  # Average chroma over time

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr) 
    features['spectral_centroid'] = np.mean(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth)

    return features