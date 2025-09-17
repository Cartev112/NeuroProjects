## Project 12: Braindecode Benchmark on MOABB Datasets

### Overview
Run compact convnets (ShallowFBCSPNet/Deep4Net/EEGNet) on a preprocessed EEG dataset to compare performance across models using a unified training loop.

### Instructions
Provide an `npz` with trials `X` (N, C, T) and labels `y`. Run:
```bash
python bench_cli.py --data_npz /path/to/mi_epochs.npz --out_dir outputs/bench
```

### Learning: Model Comparisons
Different architectures capture spectral–spatial patterns with varying inductive biases. This scaffold standardizes preprocessing and evaluation to isolate model effects.

