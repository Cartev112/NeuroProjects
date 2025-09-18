## Project 10: Structural MRI Brain Age Baseline

### Overview
Predict chronological age from structural MRI-derived features. This baseline uses region-wise features (e.g., cortical thickness/volume from FreeSurfer or FastSurfer outputs) and trains gradient boosting or linear models, reporting MAE and bias-corrected brain-age delta.

### Instructions
1) Install dependencies from `tier-1/requirements.txt`.
2) Prepare a CSV of features per subject (rows) and age/covariates (columns).
3) Run:
```bash
python brain_age_cli.py \
  --features_csv features.csv \
  --age_col age \
  --out_dir outputs/brain_age
```

### Learning: Brain Age Modeling
Brain age models map neuroanatomical features to age, capturing normative development/aging. A key step is bias correction: naive models show age-dependent error (regression to the mean). After cross-validated predictions, we fit a linear model of predicted vs true age and remove the bias to estimate brain-age delta (predicted − true) that is less confounded by age.

