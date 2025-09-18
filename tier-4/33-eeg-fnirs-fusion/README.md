## Project 33: Multimodal Fusion EEG+fNIRS

### Overview
Fuse EEG and fNIRS features to improve classification over unimodal baselines using early or late fusion.

### Instructions
Provide `eeg.npy` (N, Fe), `fnirs.npy` (N, Fh), and labels `y.npy`. Run:
```bash
python fusion_cli.py --eeg eeg.npy --fnirs fnirs.npy --labels y.npy --out_dir outputs/fusion
```

### Learning: Fusion Strategies
Early fusion concatenates features; late fusion combines classifier outputs. Calibration and normalization across modalities are key.

