## Project 28: Cross-Subject Domain Adaptation for Motor Imagery

### Overview
Train on source subjects and adapt to a new subject with minimal labels using CORAL alignment on features.

### Instructions
Provide source `X_s, y_s` and target `X_t, y_t` (few labels). Run:
```bash
python adapt_cli.py --source source.npz --target target.npz --out_dir outputs/adapt
```

### Learning: CORAL Alignment
CORAL aligns second-order statistics by whitening source features and re-coloring to target covariance, reducing domain shift.

