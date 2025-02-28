# Audio Encryption Reverse Engineering

## Project Description
Reverse-engineer encryption method used on audio files to determine transformation patterns, develop decryption methods, validate results through frequency distribution analysis, and apply solutions to larger datasets.

## Goals
- Identify encryption transformation (e.g., XOR, modular arithmetic, byte mapping)
- Develop decryption algorithm
- Validate decryption through frequency/entropy analysis
- Apply method to encrypted dataset

## Setup
### Dependencies
numpy, matplotlib, librosa, scipy, skimage, pywt, soundfile

### Installation
pip install -r requirements.txt

## Usage
### Workflow
- 1. Generate modified samples: `python create_variable_raw.py`
- 2. Analyze histograms: `python images_modified_raw.py`
- 3. Brute-force XOR analysis: `python brut-force-xor.py`
- 4. Frequency domain analysis: `python frequency_analysis.py`
- 5. Heatmap comparison: `python heatmap_compare.py`
- 6. Generate report: `python make_report.py`

### Directories
**Input:** left.raw (plaintext), modified_raw/ (encrypted samples)
**Output:** analysis_reports/, frequency_domain_analysis/, entropy_analysis_results/, decrypted_files/

## Key Files
### Analysis Scripts
| File | Purpose |
|------|---------|
| `images_modified_raw.py` | Generate histograms for raw/encrypted audio comparison |
| `brut-force-xor.py` | Brute-force XOR decryption attempts (0-255 keys) |
| `frequency_analysis.py` | STFT spectrogram analysis for time-frequency patterns |
| `heatmap_compare.py` | Visualize byte transformation patterns |
| `analyzz.py` | Advanced cryptanalysis (XOR, modular arithmetic, byte mapping) |

### Helper Scripts
| File | Purpose |
|------|---------|
| `create_variable_raw.py` | generate modified samples |
| `transition_matrix.py` | decryption validation |
| `pca_analysis.py` | dimensionality reduction analysis |

## Validation
### Metrics
Jensen-Shannon divergence, Cosine similarity, Mutual information, Entropy analysis

### Success Criteria
Decrypted file frequency distribution matches original (left.raw) within 10% tolerance

## License
MIT License - Copyright (c) 2023 Audio Encryption Project

## Acknowledgments
- Librosa community for audio processing tools
- SciPy ecosystem for statistical functions
- Matplotlib team for visualization capabilities
