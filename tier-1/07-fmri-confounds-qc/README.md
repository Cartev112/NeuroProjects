## Project 7: fMRI Confounds and Motion QC

### Overview
Extract image quality metrics and motion confounds from fMRI data and summarize per subject/session. Optionally parse outputs from fMRIPrep or compute basic metrics from raw NIfTI files.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Provide an fMRI dataset (BIDS preferred) or fMRIPrep derivatives.
3) Run:
```bash
python fmri_qc_cli.py \
  --bids_root /path/to/bids \
  --out_dir outputs/qc
```

### Learning: Motion and Confounds in fMRI
Head motion and physiological noise can severely bias fMRI analyses. Standard practice includes computing framewise displacement (FD), DVARS, and other confounds (global signal, CSF/WM signals). This code summarizes these measures and provides simple pass/fail criteria and visualizations.

