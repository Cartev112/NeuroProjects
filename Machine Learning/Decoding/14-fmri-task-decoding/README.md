## Project 14: fMRI Task Decoding

### Overview
Decode task conditions from GLM beta maps or parcel-wise time series using linear classifiers with nested CV.

### Instructions
Provide `X.npy` (N, P) and `y.npy` labels. Run:
```bash
python task_decode_cli.py --features X.npy --labels y.npy --out_dir outputs/task_decoding
```

### Learning: Decoding vs. Encoding
Decoding predicts conditions from brain data, revealing information content. Regularization and nested CV avoid overfitting given high-dimensional features.

