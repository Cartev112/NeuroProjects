"""
ERP and time-frequency feature extraction for predictive coding readouts.
"""
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def extract_erp_peaks(X, sfreq, time_windows=None):
    """Extract ERP peak amplitudes in specified time windows.
    
    X: (N, C, T) or (N, T) epochs
    sfreq: sampling frequency
    time_windows: dict {name: (t_start, t_end)} in seconds
    
    Returns: dict {name: (N,) peak amplitudes}
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    
    if time_windows is None:
        # Default MMN and P3b windows
        time_windows = {
            'mmn': (0.15, 0.25),  # 150-250 ms
            'p3b': (0.30, 0.50),  # 300-500 ms
        }
    
    out = {}
    for name, (t0, t1) in time_windows.items():
        i0 = int(t0 * sfreq)
        i1 = int(t1 * sfreq)
        i0 = max(0, i0)
        i1 = min(T, i1)
        if i1 <= i0:
            out[name] = np.full(N, np.nan)
            continue
        # Average over channels, then find peak in window
        erp = X[:, :, i0:i1].mean(axis=1)  # (N, window_len)
        peaks = np.max(np.abs(erp), axis=1)
        out[name] = peaks
    
    return out


def extract_tf_power(X, sfreq, bands=None):
    """Extract time-frequency power in specified bands using Hilbert.
    
    X: (N, C, T)
    sfreq: sampling frequency
    bands: dict {name: (f_low, f_high)}
    
    Returns: dict {name: (N, T) average power envelope}
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    
    if bands is None:
        bands = {
            'beta': (13, 30),
            'gamma': (30, 80),
        }
    
    out = {}
    for name, (fl, fh) in bands.items():
        nyq = 0.5 * sfreq
        Wn = [fl / nyq, fh / nyq]
        b, a = butter(4, Wn, btype='band')
        
        power_all = []
        for n in range(N):
            ch_power = []
            for c in range(C):
                filt = filtfilt(b, a, X[n, c])
                analytic = hilbert(filt)
                env = np.abs(analytic)
                ch_power.append(env)
            # Average over channels
            avg_power = np.mean(ch_power, axis=0)
            power_all.append(avg_power)
        
        out[name] = np.array(power_all)  # (N, T)
    
    return out


def map_latent_to_erp(latents, erp_features, method='correlation'):
    """Compute trial-wise correlation between latent PE/precision and ERP peaks.
    
    latents: dict with 'pe', 'precision' each (N, T, ...)
    erp_features: dict {name: (N,) peak values}
    
    Returns: dict {latent_name: {erp_name: correlation}}
    """
    out = {}
    for lat_name in ['pe', 'precision']:
        if lat_name not in latents:
            continue
        lat = latents[lat_name]  # (N, T, ...)
        # Summarize latent over time (mean absolute)
        lat_summary = np.mean(np.abs(lat), axis=(1, 2)) if lat.ndim == 3 else np.mean(np.abs(lat), axis=1)
        
        out[lat_name] = {}
        for erp_name, erp_vals in erp_features.items():
            if len(erp_vals) != len(lat_summary):
                continue
            mask = np.isfinite(erp_vals) & np.isfinite(lat_summary)
            if np.sum(mask) < 3:
                out[lat_name][erp_name] = np.nan
                continue
            corr = np.corrcoef(lat_summary[mask], erp_vals[mask])[0, 1]
            out[lat_name][erp_name] = corr
    
    return out
