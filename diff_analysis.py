import numpy as np
from math import gcd
from functools import reduce

def read_file(filename):
    """Reads a binary file and returns its content as a NumPy array of bytes"""
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def create_transition_matrix(raw, encrypted):
    """Creates a byte transition matrix from raw to encrypted data"""
    min_len = min(len(raw), len(encrypted))
    transition = np.zeros((256, 256), dtype=int)
    for r, e in zip(raw[:min_len], encrypted[:min_len]):
        transition[r, e] += 1
    return transition

def report_common_transitions(transition, top=10):
    """Reports the most common byte transitions in the transition matrix"""
    transitions = []
    for r in range(256):
        for e in range(256):
            count = transition[r, e]
            if count > 0:
                transitions.append((r, e, count))
    transitions.sort(key=lambda x: (-x[2], x[0], x[1]))
    return transitions[:top]

def calculate_similarity(a, b):
    """Calculates similarity between two byte arrays using Chi-squared statistic"""
    hist_a = np.bincount(a, minlength=256)
    hist_b = np.bincount(b, minlength=256)
    smooth = 1e-10
    chi2 = np.sum(((hist_a - hist_b) ** 2) / (hist_b + smooth))
    return 1 / (1 + chi2)

def brute_force_xor_key(raw, encrypted, window_size=10000):
    """Brute-forces XOR keys and scores decrypted data using histogram similarity"""
    best_key = None
    best_similarity = -1
    min_len = min(len(raw), len(encrypted))
    window_size = min(window_size, min_len)
    
    raw_window = raw[:window_size]
    encrypted_window = encrypted[:window_size]
    
    for key in range(256):
        decrypted_window = encrypted_window ^ key
        similarity = calculate_similarity(decrypted_window, raw_window)
        if similarity > best_similarity:
            best_similarity = similarity
            best_key = key
    return best_key, best_similarity

def kasiski(encrypted_data, sequence_length=3):
    """Estimates key length using Kasiski examination"""
    sequences = {}
    for i in range(len(encrypted_data) - sequence_length):
        seq = tuple(encrypted_data[i:i+sequence_length])
        if seq in sequences:
            sequences[seq].append(i)
        else:
            sequences[seq] = [i]
    distances = []
    for indices in sequences.values():
        if len(indices) >= 2:
            for j in range(1, len(indices)):
                distances.append(indices[j] - indices[0])
    if not distances:
        return None
    current_gcd = distances[0]
    for d in distances[1:]:
        current_gcd = gcd(current_gcd, d)
        if current_gcd == 1:
            break
    return current_gcd

def main():
    """Main entry point for the script"""
    raw_data = read_file('left.raw')
    encrypted_data = read_file('right.raw')
    
    # Differential Analysis
    transition = create_transition_matrix(raw_data, encrypted_data)
    common_transitions = report_common_transitions(transition)
    print("\nTop 10 byte transitions (raw -> encrypted):")
    for r, e, count in common_transitions:
        print(f"0x{r:02X} -> 0x{e:02X}: {count} occurrences")
    
    # Key Space Exhaustion (Brute-force XOR)
    key, similarity = brute_force_xor_key(raw_data, encrypted_data, window_size=20000)
    print(f"\nBest XOR Key: 0x{key:02X} (Similarity Score: {similarity:.4f})")
    
    # Kasiski Examination
    key_length = kasiski(encrypted_data)
    print(f"\nEstimated Key Length (Kasiski): {key_length}")

if __name__ == "__main__":
    main()