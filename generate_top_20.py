import os
import json
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import jaccard_score, cosine_similarity

# Configuration
RAW_AUDIO_PATH = "left.raw"
ENCRYPTED_DIR = "modified_raw"
DECRYPTED_DIR = "decrypted_files"
REPORT_DIR = "analysis_reports"

# Ensure directories exist
os.makedirs(REPORT_DIR, exist_ok=True)

def load_histogram(file_path):
    """Load histogram data from JSON report files"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['histogram']), data.get('method', 'unknown')

def calculate_similarity(raw_hist, decrypted_hist):
    """Calculate multiple similarity metrics between histograms"""
    # Add 1 to avoid zero probabilities
    js_div = entropy(raw_hist + 1, decrypted_hist + 1, base=2)
    cosine = cosine_similarity([raw_hist], [decrypted_hist])[0][0]
    jaccard = jaccard_score(np.where(raw_hist>0, 1, 0), 
                            np.where(decrypted_hist>0, 1, 0))
    return {
        'jensen_shannon': js_div,
        'cosine_similarity': cosine,
        'jaccard_index': jaccard
    }

def main():
    # Load raw audio histogram
    raw_data = np.fromfile(RAW_AUDIO_PATH, dtype=np.uint8)
    raw_hist = np.bincount(raw_data, minlength=256)

    # Collect all decrypted files
    decrypted_files = []
    for root, _, files in os.walk(DECRYPTED_DIR):
        for file in files:
            if file.endswith('.raw'):
                decrypted_files.append(os.path.join(root, file))
    
    results = []
    for decrypt_file in decrypted_files:
        try:
            # Load decrypted data
            decrypted_data = np.fromfile(decrypt_file, dtype=np.uint8)
            decrypted_hist = np.bincount(decrypted_data, minlength=256)
            
            # Calculate similarities
            metrics = calculate_similarity(raw_hist, decrypted_hist)
            
            # Get method from filename or metadata
            method = os.path.basename(decrypt_file).split('_')[0]
            
            results.append({
                'file': decrypt_file,
                'method': method,
                'jensen_shannon': metrics['jensen_shannon'],
                'cosine_similarity': metrics['cosine_similarity'],
                'jaccard_index': metrics['jaccard_index']
            })
        except Exception as e:
            print(f"Error processing {decrypt_file}: {str(e)}")

    # Calculate overall score (higher is better)
    for r in results:
        r['score'] = (
            r['cosine_similarity'] * 0.4 +
            r['jaccard_index'] * 0.3 +
            (1 - r['jensen_shannon']) * 0.3
        )

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)

    # Generate report
    report_path = os.path.join(REPORT_DIR, "top_20_decrypted_report.txt")
    with open(report_path, 'w') as report:
        report.write("Top 20 Decrypted Files Report\n")
        report.write("============================\n\n")
        
        report.write("Evaluation Criteria:\n")
        report.write("- Cosine Similarity (40%)\n")
        report.write("- Jaccard Index (30%)\n")
        report.write("- Inverted Jensen-Shannon Divergence (30%)\n\n")
        
        report.write("Top 20 Candidates:\n")
        for i, res in enumerate(results[:20]):
            report.write(f"\n{i+1}. File: {res['file']}\n")
            report.write(f"   Method: {res['method']}\n")
            report.write(f"   Score: {res['score']:.4f}\n")
            report.write(f"   Cosine Similarity: {res['cosine_similarity']:.4f}\n")
            report.write(f"   Jaccard Index: {res['jaccard_index']:.4f}\n")
            report.write(f"   Jensen-Shannon Divergence: {res['jensen_shannon']:.4f}\n")

    print(f"Report saved to {report_path}")
    print("Key findings:")
    print(f"- Best candidate: {results[0]['file']} (Score: {results[0]['score']:.4f})")
    print(f"- Worst in top 20: {results[19]['file']} (Score: {results[19]['score']:.4f})")

if __name__ == "__main__":
    main()