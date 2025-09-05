import argparse
import os
import json
import csv
import numpy as np
import sys

# allow importing metrics from parent folder
THIS_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from complexity_metrics import (  # type: ignore
    lz_complexity,
    permutation_entropy,
    sample_entropy,
    multiscale_entropy,
)
from preprocessing import (
    zscore,
    average_reference,
    bandpass_filter,
    notch_filter,
    compute_band_signals,
)

try:
    from control_sim import (
        ensure_stable,
        compute_controllability,
        simulate_linear_impulses,
    )
except Exception:
    # control_sim may not exist if Phase 2 files are not present yet
    ensure_stable = None
    compute_controllability = None
    simulate_linear_impulses = None


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _load_array(path: str):
    if path.endswith('.npy'):
        arr = np.load(path)
    elif path.endswith('.npz'):
        npz = np.load(path)
        key = 'X' if 'X' in npz.files else npz.files[0]
        arr = npz[key]
    else:
        raise ValueError('Input must be .npy or .npz')
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x[None, None, :]
    elif x.ndim == 2:
        # assume (C, T)
        if x.shape[0] < x.shape[1]:
            x = x[None, :, :]
        else:
            # if (T, C) transpose
            x = x.T[None, :, :]
    elif x.ndim == 3:
        # assume (N, C, T) else try to permute
        N, A, B = x.shape
        # if last axis is not time by heuristic, attempt swap
        if A > B:
            # could be (N, T, C)
            x = np.transpose(x, (0, 2, 1))
    else:
        raise ValueError('Expected array with ndim in {1,2,3}')
    return x  # (N, C, T)


def _windows(T: int, win_samples: int, step_samples: int):
    if win_samples <= 0:
        win_samples = T
    if step_samples <= 0:
        step_samples = T
    if win_samples >= T:
        yield 0, T
        return
    for start in range(0, T - win_samples + 1, step_samples):
        end = start + win_samples
        yield start, end
    if (T - win_samples) % step_samples != 0:
        # add final window ending at T
        yield max(0, T - win_samples), T


def _compute_metrics_1d(
    x1d: np.ndarray,
    metrics: list,
    pe_order: int,
    mse_scales: int,
    lz_bin: str = 'median',
    lz_q: float = 0.5,
    pe_tie: str = 'noise',
):
    out = {}
    if 'lzc' in metrics:
        out['lzc'] = float(lz_complexity(x1d, binarize_method=lz_bin, q=lz_q))
    if 'pe' in metrics:
        out['pe'] = float(permutation_entropy(x1d, order=pe_order, tie_strategy=pe_tie))
    if 'se' in metrics:
        out['se'] = float(sample_entropy(x1d))
    if 'mse' in metrics:
        mse = multiscale_entropy(x1d, scales=mse_scales)
        for i, v in enumerate(np.asarray(mse).ravel(), start=1):
            out[f'mse_{i}'] = None if np.isnan(v) else float(v)
    return out


def main():
    p = argparse.ArgumentParser(description='Compute perturbation complexity metrics from EEG/MEG arrays and optional simulations')
    p.add_argument('--data', required=True, help='Path to .npy or .npz (expects X key) with shape (N,C,T) or (C,T) or (T,)')
    p.add_argument('--out_dir', required=True, help='Output directory')
    p.add_argument('--sfreq', type=float, default=None, help='Sampling frequency in Hz (optional; used for reporting only)')
    p.add_argument('--win_sec', type=float, default=None, help='Window length in seconds; if omitted, uses full length')
    p.add_argument('--step_sec', type=float, default=None, help='Step length in seconds; if omitted, uses full length')
    p.add_argument('--metrics', default='lzc,pe,se,mse', help='Comma-separated metrics among lzc,pe,se,mse')
    p.add_argument('--pe_order', type=int, default=3, help='Permutation entropy order')
    p.add_argument('--mse_scales', type=int, default=5, help='Number of scales for MSE')
    p.add_argument('--channels', default=None, help='Optional path to txt file with channel names (one per line)')
    # Phase 1 enhancements
    p.add_argument('--preproc', default=None, help='Comma-separated preprocessing: zscore,avgref,bandpass,notch')
    p.add_argument('--l_freq', type=float, default=None, help='Bandpass low cut (Hz)')
    p.add_argument('--h_freq', type=float, default=None, help='Bandpass high cut (Hz)')
    p.add_argument('--notch', type=float, default=None, help='Notch frequency (Hz), e.g., 50 or 60')
    p.add_argument('--bands', default=None, help='Optional bands to compute: comma-separated subset of delta,theta,alpha,beta,gamma')
    p.add_argument('--roi_map', default=None, help='Optional CSV mapping channels to ROI with columns: channel,roi')
    p.add_argument('--lz_bin', default='median', choices=['median','mean','quantile'], help='Binarization for LZC')
    p.add_argument('--lz_q', type=float, default=0.5, help='Quantile for LZC when lz_bin=quantile')
    p.add_argument('--pe_tie', default='noise', choices=['noise','ordinal'], help='Tie handling for permutation entropy')
    # Phase 2 options
    p.add_argument('--connectome', default=None, help='Path to connectome adjacency matrix (.npy) [nodes x nodes]')
    p.add_argument('--stim', default='all', help='Stim nodes: "all", comma-separated indices (0-based), or channel/ROI names if mapping provided')
    p.add_argument('--sim_T', type=int, default=1000, help='Simulation length (samples)')
    p.add_argument('--sim_scale', type=float, default=0.95, help='Spectral radius scaling to ensure stability (<=1)')
    args = p.parse_args()

    _ensure_outdir(args.out_dir)
    X = _load_array(args.data)  # (N,C,T)
    N, C, T = X.shape

    if args.channels is not None and os.path.isfile(args.channels):
        with open(args.channels, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if len(names) != C:
            raise ValueError('channels file length does not match C')
    else:
        names = [f'ch{c:02d}' for c in range(C)]

    metrics = [m.strip().lower() for m in args.metrics.split(',') if m.strip()]
    metrics = [m for m in metrics if m in {'lzc', 'pe', 'se', 'mse'}]
    if not metrics:
        raise ValueError('No valid metrics selected')

    if args.sfreq is not None and args.win_sec is not None:
        win_samples = int(round(args.win_sec * args.sfreq))
    else:
        win_samples = T
    if args.sfreq is not None and args.step_sec is not None:
        step_samples = int(round(args.step_sec * args.sfreq))
    else:
        step_samples = T

    # preprocessing
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

    bands = None
    if args.bands:
        if args.sfreq is None:
            raise ValueError('--bands requires --sfreq')
        bands = [b.strip().lower() for b in args.bands.split(',') if b.strip()]
        band_signals = compute_band_signals(X, args.sfreq, bands)
    else:
        band_signals = {}

    # ROI mapping
    roi_map = None
    if args.roi_map and os.path.isfile(args.roi_map):
        roi_map = {}
        import csv as _csv
        with open(args.roi_map, 'r', encoding='utf-8') as rf:
            rdr = _csv.DictReader(rf)
            for row in rdr:
                ch = row['channel'].strip()
                roi = row['roi'].strip()
                idx = names.index(ch) if ch in names else None
                if idx is None:
                    continue
                roi_map.setdefault(roi, []).append(idx)

    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    fieldnames = ['epoch', 'channel', 'band', 'level', 't0', 't1']
    # preview to collect full header when including mse
    tmp = _compute_metrics_1d(
        X[0, 0, :min(T, max(4, win_samples))], metrics, args.pe_order, args.mse_scales, args.lz_bin, args.lz_q, args.pe_tie
    )
    fieldnames += list(tmp.keys())

    rows = []
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        def process_block(Xblock: np.ndarray, band_label: str):
            for n in range(N):
                for c in range(C):
                    x = Xblock[n, c]
                    for t0, t1 in _windows(T, win_samples, step_samples):
                        seg = x[t0:t1]
                        met = _compute_metrics_1d(
                            seg, metrics, args.pe_order, args.mse_scales, args.lz_bin, args.lz_q, args.pe_tie
                        )
                        row = {
                            'epoch': n,
                            'channel': names[c],
                            'band': band_label,
                            'level': 'channel',
                            't0': int(t0),
                            't1': int(t1),
                        }
                        row.update(met)
                        writer.writerow(row)
                        rows.append(row)

        # full-band
        process_block(X, 'full')
        # bands
        for bname, Xb in band_signals.items():
            process_block(Xb, bname)

    # summary
    summary = {'n_epochs': int(N), 'n_channels': int(C), 'n_samples': int(T), 'sfreq': args.sfreq}
    if rows:
        keys = [k for k in fieldnames if k not in {'epoch', 'channel', 'band', 'level', 't0', 't1'}]
        for k in keys:
            vals = [r[k] for r in rows if (r.get(k) is not None)]
            try:
                arr = np.array(vals, dtype=float)
                summary[f'mean_{k}'] = float(np.nanmean(arr))
                summary[f'std_{k}'] = float(np.nanstd(arr))
            except Exception:
                pass
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # ROI aggregation (optional)
    if roi_map:
        roi_csv = os.path.join(args.out_dir, 'roi_metrics.csv')
        with open(roi_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # aggregate by averaging channels within ROI before metrics
            for n in range(N):
                for band_label, Xblock in [('full', X)] + list(band_signals.items()):
                    for roi, idxs in roi_map.items():
                        xb = np.nanmean(Xblock[n, idxs, :], axis=0)
                        for t0, t1 in _windows(T, win_samples, step_samples):
                            seg = xb[t0:t1]
                            met = _compute_metrics_1d(
                                seg, metrics, args.pe_order, args.mse_scales, args.lz_bin, args.lz_q, args.pe_tie
                            )
                            row = {
                                'epoch': n,
                                'channel': roi,
                                'band': band_label,
                                'level': 'roi',
                                't0': int(t0),
                                't1': int(t1),
                            }
                            row.update(met)
                            writer.writerow(row)

    # Phase 2: optional connectome simulations
    if args.connectome is not None:
        if simulate_linear_impulses is None:
            print('Warning: control_sim module not available; skipping simulations')
        else:
            A = np.load(args.connectome)
            A = ensure_stable(A, target_radius=args.sim_scale)
            metrics_keys = list(tmp.keys())
            sim_csv = os.path.join(args.out_dir, 'sim_metrics.csv')
            with open(sim_csv, 'w', newline='', encoding='utf-8') as f:
                sfields = ['stim_node'] + metrics_keys
                writer = csv.DictWriter(f, fieldnames=sfields)
                writer.writeheader()
                # determine stim nodes
                if args.stim == 'all':
                    stim_nodes = list(range(A.shape[0]))
                else:
                    try:
                        stim_nodes = [int(s) for s in args.stim.split(',')]
                    except Exception:
                        # fallback if names provided and match channels or ROIs
                        stim_nodes = []
                for node in stim_nodes:
                    resp = simulate_linear_impulses(A, node, T=args.sim_T)
                    met = _compute_metrics_1d(
                        resp, metrics, args.pe_order, args.mse_scales, args.lz_bin, args.lz_q, args.pe_tie
                    )
                    row = {'stim_node': int(node)}
                    row.update(met)
                    writer.writerow(row)
            # controllability summary
            if compute_controllability is not None:
                cont = compute_controllability(A)
            else:
                cont = {}
            with open(os.path.join(args.out_dir, 'sim_summary.json'), 'w', encoding='utf-8') as f:
                json.dump({'controllability': cont, 'sim_T': int(args.sim_T)}, f, indent=2)

    print(f'Wrote {csv_path} and summary.json')


if __name__ == '__main__':
    main()
