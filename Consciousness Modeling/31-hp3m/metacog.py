"""
Metacognition models: Drift-Diffusion Model (DDM) and meta-d' for confidence.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def fit_ddm_simple(choices, rts, drift_init=1.0, bound_init=1.0, ndt_init=0.3):
    """Fit simple DDM to binary choices and RTs.
    
    choices: (N,) binary 0/1
    rts: (N,) reaction times in seconds
    
    Returns: dict with drift, boundary, ndt
    """
    choices = np.asarray(choices)
    rts = np.asarray(rts)
    
    def nll(params):
        drift, bound, ndt = params
        if bound <= 0 or ndt < 0 or ndt > np.min(rts):
            return 1e10
        
        # Simple approximation: RT ~ ndt + bound/drift + noise
        # Probability of choice 1 ~ sigmoid(drift)
        p1 = 1.0 / (1.0 + np.exp(-drift))
        p_choice = np.where(choices == 1, p1, 1 - p1)
        p_choice = np.clip(p_choice, 1e-8, 1 - 1e-8)
        
        # RT likelihood (simplified Gaussian around mean)
        mean_rt = ndt + bound / (np.abs(drift) + 0.1)
        sd_rt = 0.1
        rt_lik = norm.pdf(rts, loc=mean_rt, scale=sd_rt)
        rt_lik = np.clip(rt_lik, 1e-8, np.inf)
        
        return -np.sum(np.log(p_choice) + np.log(rt_lik))
    
    res = minimize(nll, [drift_init, bound_init, ndt_init], method='Nelder-Mead')
    drift, bound, ndt = res.x
    return {'drift': float(drift), 'boundary': float(bound), 'ndt': float(ndt), 'nll': float(res.fun)}


def compute_meta_dprime(choices, confidence, n_bins=4):
    """Compute meta-d' (type-2 sensitivity) from choices and confidence ratings.
    
    choices: (N,) binary 0/1 (correctness)
    confidence: (N,) continuous or discrete ratings
    
    Returns: meta_d' estimate
    """
    choices = np.asarray(choices)
    confidence = np.asarray(confidence)
    
    # Bin confidence into quartiles
    bins = np.percentile(confidence, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    conf_binned = np.digitize(confidence, bins) - 1
    
    # For each confidence bin, compute hit rate and false alarm rate
    hr = []
    far = []
    for b in range(n_bins):
        mask = conf_binned == b
        if np.sum(mask) == 0:
            continue
        hits = np.sum(choices[mask] == 1)
        total = np.sum(mask)
        hr.append(hits / total if total > 0 else 0.5)
        far.append((total - hits) / total if total > 0 else 0.5)
    
    if len(hr) < 2:
        return np.nan
    
    # Type-2 d' approximation
    hr = np.clip(hr, 0.01, 0.99)
    far = np.clip(far, 0.01, 0.99)
    z_hr = norm.ppf(hr)
    z_far = norm.ppf(far)
    meta_d = float(np.mean(z_hr - z_far))
    
    return meta_d


def predict_confidence_from_latents(latents, confidence, method='precision'):
    """Predict confidence from latent precision or PE magnitude.
    
    latents: dict with 'precision', 'pe'
    confidence: (N,) ground truth confidence
    
    Returns: dict with correlation and predictions
    """
    if method == 'precision':
        if 'precision' not in latents:
            return {'correlation': np.nan, 'predictions': None}
        prec = latents['precision']  # (N, T, 1)
        pred_conf = np.mean(prec, axis=(1, 2))
    elif method == 'pe':
        if 'pe' not in latents:
            return {'correlation': np.nan, 'predictions': None}
        pe = latents['pe']  # (N, T, D)
        # Inverse of PE magnitude
        pred_conf = -np.mean(np.abs(pe), axis=(1, 2))
    else:
        return {'correlation': np.nan, 'predictions': None}
    
    mask = np.isfinite(pred_conf) & np.isfinite(confidence)
    if np.sum(mask) < 3:
        return {'correlation': np.nan, 'predictions': pred_conf}
    
    corr = np.corrcoef(pred_conf[mask], confidence[mask])[0, 1]
    return {'correlation': float(corr), 'predictions': pred_conf}
