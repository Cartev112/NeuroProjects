## Project 27: Self-Supervised Pretraining for EEG (Contrastive)

### Overview
Pretrain an encoder on unlabeled EEG with SimCLR-style contrastive learning, then evaluate with a linear probe. Includes simple time-series augmentations.

### Instructions
Provide `npz` with `X` (N, C, T) unlabeled for pretraining and labeled `X_l`, `y` for probing. Run:
```bash
python ssl_cli.py --unlabeled_npz unlabeled.npz --labeled_npz labeled.npz --out_dir outputs/ssl
```

### Learning: Contrastive Learning on EEG
Two augmented views of the same epoch are encouraged to have similar embeddings, while different epochs are pushed apart. Augmentations include jitter, scaling, time masking, and channel dropout. A linear probe evaluates representation quality.

