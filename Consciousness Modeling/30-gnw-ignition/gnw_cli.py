import argparse
import os
import json
import csv
import numpy as np
from scipy.stats import spearmanr

from preprocessing import zscore, average_reference, bandpass_filter, notch_filter
from ignition import global_field_power, detect_ignition
try:
    from hmm_ignition import detect_ignition_hmm
except Exception:
    detect_ignition_hmm = None
from ec import granger_matrix, broadcasting_index


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _load_array(path: str, key_preference=('X',)):
    if path.endswith('.npy'):
        arr = np.load(path)
    elif path.endswith('.npz'):
        npz = np.load(path)
        key = None
        for k in key_preference:
            if k in npz.files:
                key = k
                break
        if key is None:
            key = npz.files[0]
        arr = npz[key]
    else:
        raise ValueError('Input must be .npy or .npz')
    return np.asarray(arr)


def _windows(T: int, win_samples: int, step_samples: int):
    if win_samples <= 0:
        win_samples = T
    if step_samples <= 0:
        step_samples = T
    if win_samples >= T:
        yield 0, T
        return
    for start in range(0, T - win_samples + 1, step_samples):
        yield start, start + win_samples
    if (T - win_samples) % step_samples != 0:
        yield max(0, T - win_samples), T


def compute_rsa_timecourse(neural: np.ndarray, dnn: np.ndarray):
    """Compute time-resolved RSA between neural and DNN features.
    neural: (N, T, F) trials x time x features
    dnn:    (N, P) trials x model features
    Returns: (T,) Spearman correlations
    """
    neural = np.asarray(neural)
    dnn = np.asarray(dnn)
    N, T, F = neural.shape
    if dnn.shape[0] != N:
        raise ValueError('neural and dnn must have same N')
    # Precompute DNN RDM
    Z = dnn - dnn.mean(axis=0, keepdims=True)
    Zd = np.corrcoef(Z, rowvar=True)
    RDM_d = 1.0 - Zd
    iu = np.triu_indices(N, k=1)
    v_d = RDM_d[iu]
    out = np.zeros(T, dtype=float)
    for t in range(T):
        Xt = neural[:, t, :]
        Xt = Xt - Xt.mean(axis=0, keepdims=True)
        C = np.corrcoef(Xt, rowvar=True)
        RDM = 1.0 - C
        v = RDM[iu]
        rho, _ = spearmanr(v, v_d)
        out[t] = rho if np.isfinite(rho) else np.nan
    return out


def main():
    p = argparse.ArgumentParser(description='GNW-Ignition: ignition detection, effective connectivity, broadcasting, and RSA')
    # Data
    p.add_argument('--data', required=True, help='Neural data .npy/.npz (X: (N,C,T) epochs x channels x time)')
    p.add_argument('--sfreq', type=float, default=None, help='Sampling frequency (Hz) for filtering and time params')
    p.add_argument('--channels', default=None, help='Optional channel names txt (one per line)')
    # Preproc
    p.add_argument('--preproc', default=None, help='Comma-separated: zscore,avgref,bandpass,notch')
    p.add_argument('--l_freq', type=float, default=None)
    p.add_argument('--h_freq', type=float, default=None)
    p.add_argument('--notch', type=float, default=None)
    # Ignition
    p.add_argument('--ignition_method', default='z', choices=['z','hmm'], help='Ignition detection method: z (threshold) or hmm')
    p.add_argument('--z_thresh', type=float, default=2.5)
    p.add_argument('--min_samples', type=int, default=10)
    p.add_argument('--smooth', type=int, default=5)
    p.add_argument('--win_sec', type=float, default=None, help='If set, analyze fixed windows, else ignition-based windows')
    p.add_argument('--step_sec', type=float, default=None)
    # EC/Granger
    p.add_argument('--lags', type=int, default=5, help='VAR lags for Granger')
    # RSA
    p.add_argument('--rsa_dnn', default=None, help='DNN features (N,P) for RSA (optional)')
    p.add_argument('--rsa_neural', default=None, help='Override neural features (N,T,F). If omitted, use X as features (F=C).')
    # Output
    p.add_argument('--out_dir', required=True)

    args = p.parse_args()
    _ensure_outdir(args.out_dir)

    X = _load_array(args.data, key_preference=('X',))  # (N,C,T)
    if X.ndim != 3:
        raise ValueError('X must be (N,C,T)')
    N, C, T = X.shape

    if args.channels and os.path.isfile(args.channels):
        with open(args.channels, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if len(names) != C:
            raise ValueError('channels file length does not match C')
    else:
        names = [f'ch{c:02d}' for c in range(C)]

    # Preprocessing
    if args.preproc:
        steps = [s.strip().lower() for s in args.preproc.split(',') if s.strip()]
        Xp = X.copy()
        if 'avgref' in steps:
            Xp = average_reference(Xp)
        if 'bandpass' in steps:
            if args.sfreq is None:
                raise ValueError('bandpass requires --sfreq')
            Xp = bandpass_filter(Xp, args.sfreq, args.l_freq, args.h_freq)
        if 'notch' in steps and args.notch is not None:
            if args.sfreq is None:
                raise ValueError('notch requires --sfreq')
            Xp = notch_filter(Xp, args.sfreq, args.notch)
        if 'zscore' in steps:
            Xp = zscore(Xp, axis=-1)
        X = Xp

    # Ignition detection
    gfp = global_field_power(X)  # (N,T)
    if args.ignition_method == 'hmm':
        if detect_ignition_hmm is None:
            print('Warning: hmmlearn not available; falling back to z-threshold ignition')
            episodes = detect_ignition(gfp, z_thresh=args.z_thresh, min_samples=args.min_samples, smooth=args.smooth)
        else:
            episodes = detect_ignition_hmm(gfp, n_components=2, min_samples=args.min_samples, smooth=args.smooth)
    else:
        episodes = detect_ignition(gfp, z_thresh=args.z_thresh, min_samples=args.min_samples, smooth=args.smooth)
    with open(os.path.join(args.out_dir, 'ignition.json'), 'w', encoding='utf-8') as f:
        json.dump({'episodes': episodes, 'z_thresh': args.z_thresh, 'min_samples': args.min_samples, 'smooth': args.smooth}, f, indent=2)

    # Windowing for EC
    win_list = []  # list of (epoch, t0, t1)
    if args.sfreq is not None and args.win_sec is not None:
        win_samples = int(round(args.win_sec * args.sfreq))
        step_samples = int(round((args.step_sec if args.step_sec is not None else args.win_sec) * args.sfreq))
        for n in range(N):
            for t0, t1 in _windows(T, win_samples, step_samples):
                win_list.append((n, t0, t1))
    else:
        # ignition-based: take first episode per epoch, and analyze that interval
        for n in range(N):
            if len(episodes[n]) > 0:
                t0, t1 = episodes[n][0]
                win_list.append((n, max(0, t0), min(T, t1)))

    # EC and broadcasting per window
    ec_csv = os.path.join(args.out_dir, 'ec_broadcast.csv')
    fieldnames = ['epoch', 't0', 't1'] + [f'bi_{nm}' for nm in names]
    with open(ec_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (n, t0, t1) in win_list:
            seg = X[n, :, t0:t1]
            if seg.shape[1] <= args.lags + 1:
                continue
            G = granger_matrix(seg, lags=args.lags)
            bi = broadcasting_index(G)
            row = {'epoch': n, 't0': int(t0), 't1': int(t1)}
            row.update({f'bi_{names[i]}': float(bi[i]) for i in range(C)})
            writer.writerow(row)

    # RSA timecourse (optional)
    if args.rsa_dnn is not None:
        Z = _load_array(args.rsa_dnn, key_preference=('Z',))  # (N,P)
        if args.rsa_neural is not None:
            neural = _load_array(args.rsa_neural, key_preference=('X',))  # (N,T,F)
        else:
            # use channels as features
            neural = np.transpose(X, (0, 2, 1))  # (N,T,C)
        rsa = compute_rsa_timecourse(neural, Z)
        with open(os.path.join(args.out_dir, 'rsa_timecourse.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['time_index', 'rsa_spearman'])
            for t, val in enumerate(rsa):
                w.writerow([t, float(val) if np.isfinite(val) else ''])

    # Summary
    meta = {
        'N': int(N), 'C': int(C), 'T': int(T),
        'lags': int(args.lags), 'sfreq': args.sfreq,
        'used_windows': len(win_list),
        'rsa': bool(args.rsa_dnn is not None)
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Wrote ignition.json, ec_broadcast.csv, and summary.json')


if __name__ == '__main__':
    main()
