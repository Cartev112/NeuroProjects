## Project 30: Global Neuronal Workspace — Ignition, Broadcasting, and RSA (GNW-Ignition)

### Overview
Detect putative "ignition" events and quantify large-scale broadcasting dynamics in neural time series (EEG/MEG), with optional time-resolved RSA against DNN features.

Core components:
- Ignition detection from Global Field Power (GFP) using z-threshold and duration.
- Effective connectivity via pairwise Granger (VAR) with broadcasting index per node.
- Optional RSA timecourse comparing neural and model representational geometry.

### Data
- Neural input `--data`: `.npy` or `.npz` containing `X` with shape `(N, C, T)`.
- Optional channel names: text file with one name per line.
- Optional DNN/model features for RSA: `(N, P)`.
- Optional neural features for RSA (if not using channels as features): `(N, T, F)`.

### Install
- Requirements:
```
numpy>=1.22
scipy>=1.8
```

### Usage
Basic run (ignition + EC/broadcasting):
```bash
python gnw_cli.py \
  --data X.npy \
  --sfreq 250 \
  --preproc zscore,avgref,bandpass,notch \
  --l_freq 1 --h_freq 40 --notch 60 \
  --z_thresh 2.5 --min_samples 15 --smooth 5 \
  --lags 5 \
  --out_dir outputs/gnw
```

Use HMM-based ignition (requires hmmlearn; falls back to z-threshold if unavailable):
```bash
pip install hmmlearn  # optional
python gnw_cli.py --data X.npy --sfreq 250 --ignition_method hmm --out_dir outputs/gnw_hmm
```

Fixed windows instead of ignition-based windows:
```bash
python gnw_cli.py --data X.npy --sfreq 250 --win_sec 1.0 --step_sec 0.5 --out_dir outputs/gnw_fixed
```

Add RSA timecourse against DNN features:
```bash
python gnw_cli.py \
  --data X.npy --sfreq 250 --out_dir outputs/gnw_rsa \
  --rsa_dnn Z.npy \
  # optionally: --rsa_neural neural_feats.npy  # (N,T,F)
```

### Outputs
- `ignition.json` — Detected ignition episodes per epoch: `[[[t0, t1], ...], ...]`
- `ec_broadcast.csv` — For each analyzed window, broadcasting index per channel/node.
- `rsa_timecourse.csv` — If RSA requested, Spearman correlation per time index.
- `summary.json` — Metadata and counts.

### Notes
- Pairwise Granger is a simple baseline; can be replaced with multivariate VAR or spectral GC.
- Broadcasting index = sum of outgoing directed strength; customize as needed.
- RSA uses correlation distance RDMs and Spearman correlation of upper triangles.
- HMM ignition uses GaussianHMM(n_components=2) on GFP to pick high-mean state segments.

### Roadmap
- Support source-reconstructed data and ROI-level broadcasting.
- HMM-based ignition refinement and state occupancy stats.
- Spectral Granger/TE for frequency-resolved broadcasting.
