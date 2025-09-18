## Project 9: MEG Time–Frequency Representations (TFR)

### Overview
Compute time–frequency decompositions for MEG sensor-level data using Morlet wavelets or multitaper methods. Average across trials and visualize induced/evoked power changes relative to baseline.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Provide an MEG dataset with events (e.g., MNE sample). 
3) Run:
```bash
python meg_tfr_cli.py \
  --bids_root /path/to/bids \
  --subject sub-01 \
  --task faces \
  --event_id face:5 scrambled:6 \
  --out_dir outputs/tfr
```

### Learning: Time–Frequency Analysis
Wavelet transforms trade temporal and spectral resolution to capture transient oscillatory power changes. MEG’s high temporal resolution allows tracking frequency-specific responses to stimuli. Baseline-normalized TFRs help visualize event-related synchronization/desynchronization (ERS/ERD). The code provides a compact, extensible pipeline.

