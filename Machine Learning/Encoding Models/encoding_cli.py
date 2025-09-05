import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def lag_matrix(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    # x: (T,)
    T = len(x)
    X = []
    for L in lags:
        if L < 0:
            X.append(np.r_[x[-L:], np.zeros(-L)])
        elif L > 0:
            X.append(np.r_[np.zeros(L), x[:-L]])
        else:
            X.append(x.copy())
    return np.vstack(X).T


def parse_args():
    p = argparse.ArgumentParser(description="EEG encoding for audio envelope")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lag_min_ms", type=int, default=-100)
    p.add_argument("--lag_max_ms", type=int, default=400)
    p.add_argument("--lag_step_ms", type=int, default=10)
    p.add_argument("--alpha", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arr = np.load(args.data_npz)
    eeg = arr["eeg"]  # (N, T, C)
    env = arr["env"]  # (N, T)
    sfreq = float(arr["sfreq"])  # Hz

    lags_ms = np.arange(args.lag_min_ms, args.lag_max_ms + 1, args.lag_step_ms)
    lags = (lags_ms * 1e-3 * sfreq).astype(int)

    n_trials, T, n_ch = eeg.shape
    r2 = np.zeros(n_ch)
    coefs = np.zeros((n_ch, len(lags)))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    X_all = [lag_matrix(env[i], lags) for i in range(n_trials)]
    for ch in range(n_ch):
        y_all = [eeg[i, :, ch] for i in range(n_trials)]
        y_true, y_pred = [], []
        coef_sum = np.zeros(len(lags))
        for tr_idx, te_idx in kf.split(X_all):
            X_tr = np.vstack([X_all[i] for i in tr_idx])
            y_tr = np.hstack([y_all[i] for i in tr_idx])
            X_te = np.vstack([X_all[i] for i in te_idx])
            y_te = np.hstack([y_all[i] for i in te_idx])
            model = Ridge(alpha=args.alpha)
            model.fit(X_tr, y_tr)
            y_pred.append(model.predict(X_te))
            y_true.append(y_te)
            coef_sum += model.coef_
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        r2[ch] = r2_score(y_true, y_pred)
        coefs[ch] = coef_sum / kf.get_n_splits()

    pd.DataFrame({"channel": np.arange(n_ch), "r2": r2}).to_csv(out / "r2_per_channel.csv", index=False)
    np.save(out / "trf_coefs.npy", coefs)
    np.save(out / "lags_ms.npy", lags_ms)
    print(f"Saved TRFs and R2 to {out}")


if __name__ == "__main__":
    main()

