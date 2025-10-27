## Project 2: EEG Power Spectral Density (PSD) Explorer

### Overview
This project computes and visualizes EEG power spectra across channels and conditions, including topographic maps of canonical frequency bands (delta, theta, alpha, beta). It supports Welch and multitaper PSDs and produces a compact HTML/PNG report.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Provide a BIDS EEG dataset or use MNE sample datasets.
3) Run:
```bash
python psd_cli.py \
  --bids_root /path/to/bids \
  --subject sub-01 \
  --task rest \
  --out_dir outputs/sub-01_task-rest \
  --method welch
```

Outputs: PSD arrays (`.npy`), bandpower CSV, channel spectra plots, and optional scalp topographies.

### Learning: Spectral Analysis of EEG
The PSD estimates the distribution of signal power across frequencies. For EEG, canonical bands relate to physiological states: delta (1–4 Hz), theta (4–8), alpha (8–12), beta (12–30). Two common estimators are:

- Welch: averages periodograms from overlapping windows to reduce variance.
- Multitaper: uses Slepian tapers to balance bias–variance and reduce spectral leakage.

In this code, we compute per-channel PSDs, aggregate bandpower per band, and visualize channel-wise spectra and topographies (using standard montages). This enables quick inspection of alpha peaks, line noise residuals, or atypical spectra.

