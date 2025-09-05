## Project 31: Hierarchical Predictive Processing & Metacognition (HP3M)

### Overview
Link predictive coding theory to conscious access and confidence by modeling precision-weighted prediction errors and their mapping to EEG/MEG signatures (MMN, P3b) and subjective reports.

Core components:
- **Variational State-Space Model (SSM)**: Recurrent encoder-decoder with latent prediction, precision, and PE dynamics.
- **Predictive Coding Readouts**: Map latent PE/precision to ERP components (MMN, P3b) and time-frequency markers (beta/gamma).
- **Metacognition**: Drift-diffusion model (DDM) for choice + RT; meta-d' for confidence calibration.
- **Self-Supervised Pretraining**: Temporal contrastive learning on raw epochs for robust representations.

### Data
- Neural input `--data`: `.npy` or `.npz` with shape `(N, C, T)` or `(N, T)`.
- Optional behavioral:
  - `--choices`: binary (N,) for DDM
  - `--rts`: reaction times (N,) for DDM
  - `--confidence`: ratings (N,) for meta-d'

### Install
```bash
pip install numpy scipy
```

### Usage

Basic SSM training and latent extraction:
```bash
python hp3m_cli.py \
  --data X.npy \
  --sfreq 250 \
  --latent_dim 32 --n_layers 2 --epochs 20 \
  --out_dir outputs/hp3m
```

With contrastive pretraining:
```bash
python hp3m_cli.py \
  --data X.npy --sfreq 250 \
  --pretrain --pretrain_epochs 5 \
  --latent_dim 32 --epochs 20 \
  --out_dir outputs/hp3m_pretrain
```

Extract ERP and TF features:
```bash
python hp3m_cli.py \
  --data X.npy --sfreq 250 \
  --extract_erp --extract_tf \
  --out_dir outputs/hp3m_features
```

Full pipeline with metacognition:
```bash
python hp3m_cli.py \
  --data X.npy --sfreq 250 \
  --choices choices.npy --rts rts.npy --confidence conf.npy \
  --pretrain --extract_erp --extract_tf \
  --latent_dim 32 --epochs 20 \
  --out_dir outputs/hp3m_full
```

### Outputs
- `latents.npz` — Latent states: `h`, `pred`, `precision`, `pe` each (N, T, ...)
- `erp_correlations.json` — Correlations between latent PE/precision and ERP peaks (MMN, P3b)
- `tf_power.npz` — Time-frequency power envelopes (beta, gamma) if requested
- `ddm.json` — DDM parameters (drift, boundary, ndt) if choices/RTs provided
- `metacog.json` — Meta-d' and confidence prediction correlation
- `summary.json` — Run metadata

### Notes
- SSM uses simplified RNN-based encoder-decoder; for production, use PyTorch/JAX with proper backprop.
- Contrastive pretraining computes InfoNCE-style loss but doesn't backprop (diagnostic only in NumPy).
- DDM fitting uses Nelder-Mead on simplified likelihood.
- Meta-d' computed via binned confidence and type-2 SDT.

### Roadmap
- Hierarchical Bayesian PCN with explicit prediction/error nodes.
- Subject-specific vs group-level latent priors.
- Active inference with action selection.
- Integration with attention manipulation experiments.
