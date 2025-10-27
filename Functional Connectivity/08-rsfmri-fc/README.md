## Project 8: Resting-state fMRI Functional Connectivity Matrices

### Overview
Compute ROI-based resting-state functional connectivity (FC) matrices and visualize networks. Supports standard atlases via Nilearn and outputs correlation and precision matrices, plus reliability across sessions if available.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Provide BIDS or NIfTI rs-fMRI and select an atlas.
3) Run:
```bash
python rsfmri_fc_cli.py \
  --func bold.nii.gz \
  --mask brain_mask.nii.gz \
  --atlas harvard_oxford \
  --out_dir outputs/fc
```

### Learning: Functional Connectivity
FC measures statistical dependence between regional time series, commonly Pearson correlation. Precision matrices (inverse covariance) estimate conditional dependencies under a Gaussian graphical model. Steps: denoise (regress confounds if available), extract ROI signals, compute correlation/precision, and assess network structure. The code provides a concise template for this workflow with Nilearn.

