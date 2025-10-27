## Project 24: fMRI Inter-Subject Correlation (ISC)

### Overview
Compute ISC during naturalistic stimuli and summarize ROI-wise correlations.

### Instructions
Provide ROI time series `ts.npy` (subjects × time × ROIs). Run:
```bash
python isc_cli.py --ts ts.npy --out_dir outputs/isc
```

### Learning: ISC
ISC measures the similarity of time series across subjects, indicating shared stimulus-driven responses.

