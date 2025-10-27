## Project 19: EEG Encoding Model for Auditory Envelopes

### Overview
Fit linear time-lagged encoding models (mTRF-style) to predict EEG from stimulus envelopes; compute temporal response functions (TRFs) and R².

### Instructions
Provide an `npz` with `eeg` (N, T, C), `env` (N, T), and `sfreq`. Run:
```bash
python encoding_cli.py --data_npz /path/to/audio_eeg.npz --out_dir outputs/encoding
```

### Learning: Encoding with Time Lags
We construct a design matrix of the envelope shifted over lags (e.g., -100..400 ms) and fit ridge regression per channel. The coefficients across lags form the TRF. Predictive performance (R²) is evaluated via cross-validation.

