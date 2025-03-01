import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import glob
import json
import time
from datetime import datetime

# Import functions from existing scripts
sys.path.append('.')
from transition_matrix import read_raw_audio, compute_histogram, validate_decryption
from pca_analysis import compute_features_from_sliding_window, compute_pca

# Constants
RAW_AUDIO_PATH = "./left.raw"
ENCRYPTED_RAW_DIR = "./modified_raw"
DECRYPTED_DIR = "./decryption_analysis"
OUTPUT_DIR = "./orchestration_results"
REPORT_PATH = os.path.join(OUTPUT_DIR, "decryption_report.md")

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [OUTPUT_DIR, DECRYPTED_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_script(script_name):
    """Run a Python script and return its output."""
    print(f"Running {script_name}...")
    try:
        result = subprocess.run(['python', script_name], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        print(f"Successfully executed {script_name}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")
        print(f"Script output: {e.output}")
        print(f"Script error: {e.stderr}")
        return None

def analyze_histograms():
    """Analyze histograms of raw and decrypted files to calculate similarity metrics."""
    print("Analyzing histograms...")
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    raw_hist = compute_histogram(raw_data)
    
    # Find all decrypted files
    decrypted_files = glob.glob(os.path.join(DECRYPTED_DIR, "*.raw"))
    
    results = []
    for decrypted_file in decrypted_files:
        file_name = os.path.basename(decrypted_file)
        decrypted_data = read_raw_audio(decrypted_file)
        decrypted_hist = compute_histogram(decrypted_data)
        
        # Compute validation metrics
        metrics = validate_decryption(raw_hist, decrypted_hist)
        
        results.append({
            "file_name": file_name,
            "path": decrypted_file,
            "cosine_similarity": metrics["cosine_similarity"],
            "pearson_correlation": metrics["pearson_correlation"]
        })
    
    return results

def perform_pca_comparison():
    """Perform PCA comparison between raw and decrypted files."""
    print("Performing PCA comparison...")
    raw_data = read_raw_audio(RAW_AUDIO_PATH)
    
    # Set parameters for feature extraction
    window_size = 100
    step_size = 50
    
    # Extract features from raw audio
    raw_features = compute_features_from_sliding_window(raw_data, window_size, step_size)
    raw_pca, _ = compute_pca(raw_features)
    
    # Find all decrypted files
    decrypted_files = glob.glob(os.path.join(DECRYPTED_DIR, "*.raw"))
    
    results = []
    for decrypted_file in decrypted_files:
        file_name = os.path.basename(decrypted_file)
        
        # Extract features from decrypted audio
        decrypted_data = read_raw_audio(decrypted_file)
        decrypted_features = compute_features_from_sliding_window(decrypted_data, window_size, step_size)
        
        if decrypted_features.shape[0] < 2:
            print(f"Not enough data points for PCA on {file_name}, skipping...")
            continue
            
        decrypted_pca, _ = compute_pca(decrypted_features)
        
        # Calculate average distance between PCA components
        min_len = min(len(raw_pca), len(decrypted_pca))
        if min_len > 0:
            distances = []
            for i in range(min_len):
                dist = np.linalg.norm(raw_pca[i] - decrypted_pca[i])
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            results.append({
                "file_name": file_name,
                "path": decrypted_file,
                "pca_distance": avg_distance
            })
    
    return results

def parse_ml_results():
    """Parse results from the ML analysis."""
    print("Parsing ML results...")
    try:
        clustering_results = pd.read_csv(os.path.join("./ml_results", "clustering_results.csv"))
        feature_importance = pd.read_csv(os.path.join("./ml_results", "feature_importance.csv"))
        
        # Find the cluster that most likely contains correct decryptions
        # This is a simplification - in a real scenario, we would need more robust logic
        cluster_counts = clustering_results['cluster'].value_counts()
        target_cluster = cluster_counts.index[0]  # Assuming the largest cluster is most likely correct
        
        target_files = clustering_results[clustering_results['cluster'] == target_cluster]['file'].tolist()
        
        results = []
        for file in target_files:
            # Find corresponding decryption file
            decrypted_file = os.path.join(DECRYPTED_DIR, f"decrypted_{file}")
            if os.path.exists(decrypted_file):
                results.append({
                    "file_name": f"decrypted_{file}",
                    "path": decrypted_file,
                    "cluster": target_cluster,
                    "ml_score": 1.0  # Placeholder score, would be more sophisticated in real implementation
                })
        
        return results
    except Exception as e:
        print(f"Error parsing ML results: {e}")
        return []

def combine_results(histogram_results, pca_results, ml_results):
    """Combine results from different analyses to rank decrypted files."""
    print("Combining analysis results...")
    
    # Create a dictionary to hold all metrics for each file
    all_files = {}
    
    # Add histogram results
    for result in histogram_results:
        file_name = result["file_name"]
        if file_name not in all_files:
            all_files[file_name] = {
                "path": result["path"],
                "metrics": {}
            }
        all_files[file_name]["metrics"]["cosine_similarity"] = result["cosine_similarity"]
        all_files[file_name]["metrics"]["pearson_correlation"] = result["pearson_correlation"]
    
    # Add PCA results
    for result in pca_results:
        file_name = result["file_name"]
        if file_name not in all_files:
            all_files[file_name] = {
                "path": result["path"],
                "metrics": {}
            }
        # Invert the distance so higher is better (like the other metrics)
        all_files[file_name]["metrics"]["pca_similarity"] = 1.0 / (1.0 + result["pca_distance"])
    
    # Add ML results
    for result in ml_results:
        file_name = result["file_name"]
        if file_name not in all_files:
            all_files[file_name] = {
                "path": result["path"],
                "metrics": {}
            }
        all_files[file_name]["metrics"]["ml_score"] = result["ml_score"]
    
    # Calculate composite score (weighted average of available metrics)
    weights = {
        "cosine_similarity": 0.35,
        "pearson_correlation": 0.35,
        "pca_similarity": 0.2,
        "ml_score": 0.1
    }
    
    for file_name, data in all_files.items():
        total_weight = 0
        weighted_sum = 0
        
        for metric, weight in weights.items():
            if metric in data["metrics"]:
                weighted_sum += data["metrics"][metric] * weight
                total_weight += weight
        
        # Calculate final score
        if total_weight > 0:
            data["composite_score"] = weighted_sum / total_weight
        else:
            data["composite_score"] = 0
    
    # Convert to list and sort by composite score
    results_list = []
    for file_name, data in all_files.items():
        results_list.append({
            "file_name": file_name,
            "path": data["path"],
            "metrics": data["metrics"],
            "composite_score": data["composite_score"]
        })
    
    # Sort by composite score in descending order
    results_list.sort(key=lambda x: x["composite_score"], reverse=True)
    
    return results_list

def generate_report(top_files):
    """Generate a comprehensive report of the decryption analysis."""
    print("Generating final report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(REPORT_PATH, 'w') as f:
        f.write("# Audio Decryption Analysis Report\n\n")
        f.write(f"**Generated on:** {timestamp}\n\n")
        
        f.write("## Project Overview\n\n")
        f.write("This report summarizes the results of our audio decryption analysis. ")
        f.write("The goal was to reverse engineer an encryption method used on audio files ")
        f.write("and identify the most likely correctly decrypted files.\n\n")
        
        f.write("## Analysis Methods\n\n")
        f.write("The following analysis methods were combined to evaluate decrypted files:\n\n")
        f.write("1. **Histogram Analysis**: Comparing byte frequency distributions using cosine similarity and Pearson correlation\n")
        f.write("2. **PCA Analysis**: Comparing principal components of audio data\n")
        f.write("3. **Machine Learning Analysis**: Clustering and classification of audio features\n\n")
        
        f.write("## Top 10 Most Likely Correctly Decrypted Files\n\n")
        f.write("| Rank | File Name | Composite Score | Cosine Similarity | Pearson Correlation | PCA Similarity | ML Score |\n")
        f.write("|------|-----------|----------------|-------------------|---------------------|---------------|----------|\n")
        
        for i, file_data in enumerate(top_files[:10], 1):
            metrics = file_data["metrics"]
            cosine = metrics.get("cosine_similarity", "N/A")
            pearson = metrics.get("pearson_correlation", "N/A")
            pca = metrics.get("pca_similarity", "N/A")
            ml = metrics.get("ml_score", "N/A")
            
            if cosine != "N/A":
                cosine = f"{cosine:.4f}"
            if pearson != "N/A":
                pearson = f"{pearson:.4f}"
            if pca != "N/A":
                pca = f"{pca:.4f}"
            if ml != "N/A":
                ml = f"{ml:.4f}"
                
            f.write(f"| {i} | {file_data['file_name']} | {file_data['composite_score']:.4f} | ")
            f.write(f"{cosine} | {pearson} | {pca} | {ml} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        f.write("### Histogram Comparison\n\n")
        f.write("Histogram comparison measures how closely the byte frequency distribution of ")
        f.write("decrypted files matches the original raw audio. Higher similarity indicates ")
        f.write("better decryption.\n\n")
        
        f.write("### PCA Comparison\n\n")
        f.write("PCA comparison analyzes the structural patterns in the audio data. ")
        f.write("Closer PCA components suggest that the decrypted audio has similar ")
        f.write("structural properties to the original.\n\n")
        
        f.write("### Machine Learning Analysis\n\n")
        f.write("ML techniques were used to cluster and classify audio files based on ")
        f.write("extracted features. Files in clusters similar to the raw audio are ")
        f.write("more likely to be correctly decrypted.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Based on our comprehensive analysis, the top-ranked files are the most ")
        f.write("likely to be correctly decrypted. We recommend focusing further analysis ")
        f.write("on these files, particularly the top 3, which show the strongest evidence ")
        f.write("of successful decryption.\n")
    
    print(f"Report generated at: {REPORT_PATH}")
    return REPORT_PATH

def visualize_top_results(top_files):
    """Create visualizations for the top decrypted files."""
    print("Creating visualizations...")
    
    # Prepare data for visualization
    file_names = [f["file_name"] for f in top_files[:10]]
    scores = [f["composite_score"] for f in top_files[:10]]
    
    # Create bar chart of composite scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(file_names, scores, color='skyblue')
    plt.xlabel('Decrypted Files')
    plt.ylabel('Composite Score')
    plt.title('Top 10 Decrypted Files by Composite Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Save the visualization
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_decrypted_files.png'))
    plt.close()
    
    # Create radar chart for top 3 files
    metrics = ['Cosine Similarity', 'Pearson Correlation', 'PCA Similarity', 'ML Score']
    
    # Prepare data
    top3_data = []
    for file_data in top_files[:3]:
        file_metrics = []
        for metric in ['cosine_similarity', 'pearson_correlation', 'pca_similarity', 'ml_score']:
            value = file_data['metrics'].get(metric, 0)
            file_metrics.append(value)
        top3_data.append(file_metrics)
    
    # Create radar chart  
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, data in enumerate(top3_data):
        data += data[:1]  # Close the polygon
        ax.plot(angles, data, linewidth=2, label=f"File {i+1}: {top_files[i]['file_name']}")
        ax.fill(angles, data, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Metric Comparison for Top 3 Files')
    ax.legend(loc='upper right')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'top3_radar_chart.png'))
    plt.close()
    
    print("Visualizations created.")

def main():
    start_time = time.time()
    print("Starting decryption analysis orchestration...")
    
    # Create necessary directories
    create_directories()
    
    # Run existing scripts
    run_script("transition_matrix.py")
    run_script("pca_analysis.py")
    run_script("NPU_clustering.py")
    
    # Perform analysis on decrypted files
    histogram_results = analyze_histograms()
    pca_results = perform_pca_comparison()
    ml_results = parse_ml_results()
    
    # Combine results and rank files
    all_results = combine_results(histogram_results, pca_results, ml_results)
    
    # Generate report with top files
    report_path = generate_report(all_results)
    
    # Create visualizations
    visualize_top_results(all_results)
    
    end_time = time.time()
    print(f"Analysis complete! Time taken: {end_time - start_time:.2f} seconds")
    print(f"Report saved to: {report_path}")
    print(f"Check {OUTPUT_DIR} for visualization files.")

if __name__ == "__main__":
    main()