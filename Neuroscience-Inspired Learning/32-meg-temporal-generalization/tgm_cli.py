import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score


def parse_args():
    p = argparse.ArgumentParser(description="Temporal Generalization Matrix")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arr = np.load(args.data_npz)
    X, y = arr["X"], arr["y"].astype(int)
    N, C, T = X.shape
    clf = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    tgm = np.zeros((T, T))
    for tr_t in range(T):
        X_tr = X[:, :, tr_t]
        clf.fit(X_tr, y)
        for te_t in range(T):
            X_te = X[:, :, te_t]
            y_pred = clf.predict(X_te)
            tgm[tr_t, te_t] = balanced_accuracy_score(y, y_pred)
    np.save(out / "tgm.npy", tgm)
    pd.DataFrame(tgm).to_csv(out / "tgm.csv", index=False)
    print(f"Saved TGM to {out}")


if __name__ == "__main__":
    main()

