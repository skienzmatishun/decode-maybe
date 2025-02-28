import librosa
import soundfile as sf
import numpy as np

def read_raw_audio(file_path, sr=16000, channels=1):
    """Reads raw audio data and resamples it."""
    try:
        # Provide samplerate and channels to sf.read for raw files
        data, samplerate = sf.read(file_path, samplerate=sr, channels=channels, format='RAW', subtype='PCM_16')  # Explicitly define format as RAW and subtype
    except Exception as e:
        raise Exception(f"Error reading audio file {file_path}: {e}")

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert to mono if stereo (though should be mono already if channels=1 above)

    # Convert to floating-point (e.g., float32) and normalize
    data = data.astype(np.float32)  # Convert to float32
    data = data / 32768.0         # Normalize if it was 16-bit integer audio (adjust if needed)
                                   # 32768.0 is 2**15, the max positive value for int16

    return librosa.resample(data, orig_sr=samplerate if samplerate else sr, target_sr=sr)  # Use 'sr' from function argument, not potentially undefined 'samplerate' from sf.read