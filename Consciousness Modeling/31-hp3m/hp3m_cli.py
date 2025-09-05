import argparse
import os
import json
import csv
import numpy as np

from ssm import PredictiveSSM
from features import extract_erp_peaks, extract_tf_power, map_latent_to_erp
from metacog import fit_ddm_simple, compute_meta_dprime, predict_confidence_from_latents
from pretrain import pretrain_ssm_contrastive


def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _load_array(path: str, key='X'):
    if path.endswith('.npy'):
        arr = np.load(path)
    elif path.endswith('.npz'):
        npz = np.load(path)
        arr = npz[key] if key in npz.files else npz[npz.files[0]]
    else:
        raise ValueError('Input must be .npy or .npz')
    return np.asarray(arr)


def main():
    p = argparse.ArgumentParser(description='HP3M: Hierarchical Predictive Processing & Metacognition')
    # Data
    p.add_argument('--data', required=True, help='Neural data (N,C,T) or (N,T)')
    p.add_argument('--choices', default=None, help='Binary choices (N,) for DDM (optional)')
    p.add_argument('--rts', default=None, help='Reaction times (N,) for DDM (optional)')
    p.add_argument('--confidence', default=None, help='Confidence ratings (N,) for meta-d\' (optional)')
    p.add_argument('--sfreq', type=float, default=250.0, help='Sampling frequency (Hz)')
    # SSM
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    # Pretrain
    p.add_argument('--pretrain', action='store_true', help='Run contrastive pretraining')
    p.add_argument('--pretrain_epochs', type=int, default=5)
    # ERP/TF
    p.add_argument('--extract_erp', action='store_true', help='Extract ERP peaks (MMN, P3b)')
    p.add_argument('--extract_tf', action='store_true', help='Extract TF power (beta, gamma)')
    # Output
    p.add_argument('--out_dir', required=True)

    args = p.parse_args()
    _ensure_outdir(args.out_dir)

    # Load data
    X = _load_array(args.data, key='X')
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    input_dim = C

    # Flatten to (N, T*C) for SSM input (simplified)
    X_flat = X.reshape(N, T, C)

    # Initialize SSM
    ssm = PredictiveSSM(input_dim=input_dim, latent_dim=args.latent_dim, n_layers=args.n_layers, lr=args.lr)

    # Pretrain (optional)
    if args.pretrain:
        print('Running contrastive pretraining...')
        ssm = pretrain_ssm_contrastive(ssm, X_flat, n_epochs=args.pretrain_epochs, verbose=True)

    # Train SSM
    print('Training SSM...')
    ssm.fit(X_flat, n_epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    # Extract latents
    print('Extracting latents...')
    latents = ssm.extract_latents(X_flat)

    # Save latents
    np.savez(
        os.path.join(args.out_dir, 'latents.npz'),
        h=latents['h'],
        pred=latents['pred'],
        precision=latents['precision'],
        pe=latents['pe'],
    )

    # ERP features
    erp_corr = {}
    if args.extract_erp:
        print('Extracting ERP features...')
        erp_feats = extract_erp_peaks(X, args.sfreq)
        erp_corr = map_latent_to_erp(latents, erp_feats)
        with open(os.path.join(args.out_dir, 'erp_correlations.json'), 'w') as f:
            json.dump(erp_corr, f, indent=2)

    # TF features
    if args.extract_tf:
        print('Extracting TF power...')
        tf_power = extract_tf_power(X, args.sfreq)
        np.savez(os.path.join(args.out_dir, 'tf_power.npz'), **tf_power)

    # Metacognition
    ddm_res = None
    meta_d = None
    conf_pred = None

    if args.choices is not None and args.rts is not None:
        print('Fitting DDM...')
        choices = _load_array(args.choices, key='choices')
        rts = _load_array(args.rts, key='rts')
        ddm_res = fit_ddm_simple(choices, rts)
        with open(os.path.join(args.out_dir, 'ddm.json'), 'w') as f:
            json.dump(ddm_res, f, indent=2)

    if args.confidence is not None:
        print('Computing meta-d\'...')
        confidence = _load_array(args.confidence, key='confidence')
        if args.choices is not None:
            choices = _load_array(args.choices, key='choices')
            meta_d = compute_meta_dprime(choices, confidence)
        
        # Predict confidence from latents
        conf_pred = predict_confidence_from_latents(latents, confidence, method='precision')
        
        with open(os.path.join(args.out_dir, 'metacog.json'), 'w') as f:
            json.dump({
                'meta_d': float(meta_d) if meta_d is not None else None,
                'confidence_correlation': conf_pred['correlation'] if conf_pred else None,
            }, f, indent=2)

    # Summary
    summary = {
        'N': int(N), 'C': int(C), 'T': int(T),
        'latent_dim': int(args.latent_dim),
        'n_layers': int(args.n_layers),
        'epochs': int(args.epochs),
        'pretrain': bool(args.pretrain),
        'erp_extracted': bool(args.extract_erp),
        'tf_extracted': bool(args.extract_tf),
        'ddm_fitted': ddm_res is not None,
        'meta_d_computed': meta_d is not None,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Done. Outputs in {args.out_dir}')


if __name__ == '__main__':
    main()
