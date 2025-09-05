## Project 31: Normative Modeling of Cortical Thickness (GPR)

### Overview
Fit a Gaussian Process Regression normative model for cortical thickness vs age/sex and compute subject-level deviation z-scores.

### Instructions
Provide CSV with thickness features and covariates. Run:
```bash
python normative_cli.py --features thickness.csv --target_col age --out_dir outputs/normative
```

### Learning: Normative Modeling
Normative models estimate expected values given covariates; deviations identify atypical individuals. GPR captures nonlinear trajectories and uncertainty.

