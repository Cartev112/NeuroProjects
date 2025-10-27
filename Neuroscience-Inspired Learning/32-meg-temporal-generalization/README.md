## Project 32: MEG Temporal Generalization Matrices

### Overview
Train time-specific decoders and test across all time points to build temporal generalization matrices (TGMs).

### Instructions
Provide epochs `X` (N, C, T) and labels `y`. Run:
```bash
python tgm_cli.py --data_npz Xy.npz --out_dir outputs/tgm
```

### Learning: Temporal Dynamics
TGMs reveal when information is stable vs transient over time; off-diagonal performance indicates temporal generalization.

