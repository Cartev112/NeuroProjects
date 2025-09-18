import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def bandpower_features(X: np.ndarray, sfreq: float, band=(70, 150)):
    import scipy.signal as sg
    N, C, T = X.shape
    feats = []
    for i in range(N):
        f, Pxx = sg.welch(X[i], fs=sfreq, nperseg=int(sfreq * 0.5), axis=-1)
        mask = (f >= band[0]) & (f <= band[1])
        bp = Pxx[..., mask].mean(axis=-1)
        feats.append(bp)
    return np.asarray(feats)


def parse_args():
    p = argparse.ArgumentParser(description="ECoG high-gamma decoding")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sfreq", type=float, default=1000.0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arr = np.load(args.data_npz)
    X, y = arr["X"], arr["y"].astype(int)
    Xf = bandpower_features(X, args.sfreq)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, Xf, y, cv=cv, scoring="balanced_accuracy")
    pd.Series(scores).to_csv(out / "ba_cv.csv", index=False)
    print(scores.mean())


if __name__ == "__main__":
    main()

