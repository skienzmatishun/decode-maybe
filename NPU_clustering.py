# ml_analysis.py
# ml_analysis.py
import os
import numpy as np
import pandas as pd
import librosa  # Added missing import
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models
from handle_raw_audio import read_raw_audio

AUTOENCODER_EPOCHS = 50
BATCH_SIZE = 32
RAW_AUDIO_PATH = os.getenv("RAW_AUDIO_PATH")
ENCRYPTED_DIR = os.getenv("ENCRYPTED_DIR")
DECRYPTED_DIRS = os.getenv("DECRYPTED_DIRS").split(",")
RESULTS_DIR = "./ml_results"
ENCODING_DIM = 64

def entropy(signal):
    """Calculate Shannon entropy of a signal"""
    counts = np.bincount(signal.astype(np.uint8), minlength=256)
    probabilities = counts / len(signal)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def extract_features(audio):
    """Feature extraction optimized for M-series NPU"""
    features = {
        'histogram': np.bincount(audio.astype(np.uint8), minlength=256),
        'entropy': entropy(audio),
        'mean': np.mean(audio),
        'std': np.std(audio),
        'zcr': np.mean(librosa.feature.zero_crossing_rate(audio.astype(float)))
    }
    return np.concatenate([
        features['histogram'],
        [features['entropy'], features['mean'], features['std'], features['zcr']]
    ])

def prepare_clustering_data():
    """Prepare data for clustering analysis"""
    features = []
    files = []
    
    for fname in os.listdir(ENCRYPTED_DIR):
        if fname.endswith('.raw'):
            audio = read_raw_audio(os.path.join(ENCRYPTED_DIR, fname))
            features.append(extract_features(audio))
            files.append(fname)
    
    return np.array(features), files

def clustering_analysis():
    """K-means clustering of encrypted files"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare data
    X, files = prepare_clustering_data()
    X = StandardScaler().fit_transform(X)
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    # Save results
    pd.DataFrame({
        'file': files,
        'cluster': clusters,
        'pca1': X_pca[:,0],
        'pca2': X_pca[:,1]
    }).to_csv(os.path.join(RESULTS_DIR, 'clustering_results.csv'), index=False)
    
    return clusters

def prepare_classification_data():
    """Prepare dataset for raw vs encrypted classification"""
    X, y = [], []
    
    # Raw samples
    raw_audio = read_raw_audio(RAW_AUDIO_PATH)
    X.append(extract_features(raw_audio))
    y.append(0)
    
    # Encrypted samples
    for fname in os.listdir(ENCRYPTED_DIR):
        if fname.endswith('.raw'):
            audio = read_raw_audio(os.path.join(ENCRYPTED_DIR, fname))
            X.append(extract_features(audio))
            y.append(1)
    
    return np.array(X), np.array(y)

def classification_analysis():
    """Train classifier to detect encryption"""
    X, y = prepare_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model and results
    pd.DataFrame({
        'feature': ['feature_' + str(i) for i in range(X.shape[1])],
        'importance': clf.feature_importances_
    }).to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
    
    return accuracy

def build_autoencoder(input_dim):
    """Neural network optimized for M-series NPU"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dense(ENCODING_DIM, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def autoencoder_analysis():
    """Train autoencoder to learn encryption patterns"""
    # Prepare data pairs
    raw = read_raw_audio(RAW_AUDIO_PATH)
    encrypted_files = [f for f in os.listdir(ENCRYPTED_DIR) if f.endswith('.raw')]
    
    # Create dataset
    X_train, X_val = [], []
    for fname in encrypted_files[:len(encrypted_files)//2]:
        enc = read_raw_audio(os.path.join(ENCRYPTED_DIR, fname))
        X_train.append(np.stack([raw[:len(enc)], enc]).T)
    
    for fname in encrypted_files[len(encrypted_files)//2:]:
        enc = read_raw_audio(os.path.join(ENCRYPTED_DIR, fname))
        X_val.append(np.stack([raw[:len(enc)], enc]).T)
    
    # Train autoencoder
    autoencoder = build_autoencoder(2)
    history = autoencoder.fit(
        np.concatenate(X_train),
        np.concatenate(X_train),
        validation_data=(np.concatenate(X_val), np.concatenate(X_val)),
        epochs=AUTOENCODER_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    
    # Save model and training history
    autoencoder.save(os.path.join(RESULTS_DIR, 'autoencoder_model'))
    pd.DataFrame(history.history).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)
    
    return autoencoder

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Performing clustering analysis...")
    clustering_analysis()
    
    print("\nRunning classification analysis...")
    accuracy = classification_analysis()
    print(f"Classification Accuracy: {accuracy:.2f}")
    
    print("\nTraining autoencoder...")
    autoencoder_analysis()
    
    print("\nML analysis complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()