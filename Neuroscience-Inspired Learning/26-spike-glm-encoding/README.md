## Project 26: Spike Train GLM (LNP) Encoding

### Overview
Fit Poisson GLMs to spike trains using stimulus or kinematic regressors with time lags; evaluate predictive log-likelihood.

### Instructions
Provide binned spikes `y.npy` (T,) and regressors `X.npy` (T, P). Run:
```bash
python spike_glm_cli.py --X X.npy --y y.npy --out_dir outputs/spike_glm
```

### Learning: LNP Models
The Linear-Nonlinear-Poisson model captures stimulus-response relationships with a log link. Time-lagged design matrices and refractory terms extend the model to history dependence.

