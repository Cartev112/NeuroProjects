## Project 4: Sleep Staging Baseline (Classical Features + ML)

### Overview
This project implements a classical sleep staging baseline on Sleep-EDF: extract per-epoch features (bandpower, Hjorth parameters, entropy), then train an SVM/RandomForest. It includes cross-subject evaluation and confusion matrices.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Prepare Sleep-EDF locally (not required to run code scaffolds now).
3) Run:
```bash
python sleep_cli.py \
  --data_root /path/to/sleep_edf \
  --out_dir outputs/sleep_baseline \
  --classifier svm
```

### Learning: Sleep Stages and Classical Features
Sleep staging assigns 30-second epochs to W, N1, N2, N3, REM. Classical baselines compute hand-crafted features per epoch:

- Spectral bandpower (delta/theta/alpha/sigma/beta) using Welch PSD.
- Hjorth parameters (activity, mobility, complexity) capturing signal shape.
- Entropy and statistics (variance, kurtosis), and spindle-related sigma power.

We then train a simple classifier with stratified cross-subject splitting. This provides a reliable baseline before deep models. The code emphasizes clear feature extraction and reproducible evaluation.

