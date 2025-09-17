## Project 5: Motor Imagery Classification with CSP + LDA

### Overview
This project reproduces a classic BCI baseline: classify left vs right hand motor imagery using Common Spatial Patterns (CSP) for feature extraction and Linear Discriminant Analysis (LDA) for classification. It supports data loading via MOABB or pre-extracted epochs.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Prepare a motor imagery dataset (e.g., BNCI 2014-001, BCI IV 2a).
3) Run:
```bash
python mi_cli.py \
  --data_root /path/to/data \
  --out_dir outputs/mi_csp_lda
```

### Learning: CSP and LDA in EEG Decoding
Common Spatial Patterns learns spatial filters that maximize variance for one class while minimizing it for the other. Applying these filters projects multi-channel EEG into a low-dimensional space where log-variance features are highly discriminative. LDA then finds a linear boundary maximizing class separation under Gaussian assumptions with shared covariance.

In code, we band-pass filter around the mu/beta bands (e.g., 8–30 Hz), epoch around motor imagery trials, compute class-wise covariance matrices, fit CSP filters, compute log-variance features, and train/test an LDA with cross-validation.

