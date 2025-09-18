## Project 11: Sleep Staging with a Compact CNN/CRNN

### Overview
Train a compact deep model (CNN/CRNN) on per-epoch EEG for sleep staging and compare against classical baselines. Outputs per-stage F1, confusion matrix, and training curves.

### Instructions
Prepare an `npz` with `epochs` (N, C, T) and `labels` (N,). Run:
```bash
python sleep_deep_cli.py \
  --data_npz /path/to/sleep_epochs.npz \
  --out_dir outputs/sleep_deep \
  --model cnn
```

### Learning: Why Deep Models for Sleep?
Sleep stages exhibit specific temporal–spectral patterns (spindles, K-complexes). CNNs learn band-limited patterns across channels; CRNNs add temporal context. Class imbalance requires careful metrics (per-class F1) and possibly focal loss or weighting.

