## Project 3: ERP Pipeline for Oddball Paradigms

### Overview
This project implements an event-related potential (ERP) pipeline for a simple oddball paradigm: preprocessing, epoching, baseline correction, robust averaging, and peak detection (P300/N100). It outputs component amplitudes and latencies to CSV and produces publication-ready figures.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Use any BIDS EEG oddball dataset or MNE sample data with events.
3) Run:
```bash
python erp_cli.py \
  --bids_root /path/to/bids \
  --subject sub-01 \
  --task oddball \
  --event_id standard:1 deviant:2 \
  --tmin -0.2 --tmax 0.8 \
  --baseline -0.2 0 \
  --out_dir outputs/sub-01
```

### Learning: What is an ERP?
ERPs are time-locked averages of EEG responses to discrete events. Averaging across trials improves SNR by canceling unrelated activity. Key steps:

- Epoching: segment continuous EEG around events to form trials.
- Baseline correction: subtract pre-stimulus mean to remove slow drifts.
- Robust averaging: weights trials to downweight outliers and reduce artifact influence.
- Component quantification: detect peaks (e.g., P300 around 300 ms at parietal electrodes), measure amplitude and latency.

The code applies these steps using MNE, performs condition contrasts (deviant − standard), and exports component measurements for downstream stats.

