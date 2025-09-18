## Project 17: Automated ICA Component Labeling for EEG

### Overview
Train a classifier to label ICA components as brain vs artifact using features from component maps and spectra.

### Instructions
Provide a feature CSV with labels (brain/artifact). Run:
```bash
python ica_label_cli.py --features features.csv --label_col label --out_dir outputs/ica_label
```

### Learning: Component Features
Spectral slopes, peaks at line noise, kurtosis, and topography-based spatial features help discriminate artifacts (EOG/EMG) from neural components.

