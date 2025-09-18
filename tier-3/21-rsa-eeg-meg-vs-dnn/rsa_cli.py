import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def rdm(X):
    # X: (N, F) -> RDM (N,N) using 1 - Pearson
    Xc = X - X.mean(0, keepdims=True)
    Xc /= (Xc.std(0, keepdims=True) + 1e-9)
    sim = Xc @ Xc.T / X.shape[1]
    d = 1 - sim
    return d


def parse_args():
    p = argparse.ArgumentParser(description="Time-resolved RSA")
    p.add_argument("--neural", required=True)
    p.add_argument("--dnn", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(args.neural)  # (N, T, F)
    Z = np.load(args.dnn)     # (N, P)
    R_model = rdm(Z).reshape(-1)
    N, T, F = X.shape
    corrs = []
    for t in range(T):
        R_t = rdm(X[:, t, :]).reshape(-1)
        c = np.corrcoef(R_t, R_model)[0, 1]
        corrs.append(c)
    pd.DataFrame({"time": np.arange(T), "rsa_corr": corrs}).to_csv(out / "rsa_timecourse.csv", index=False)
    print(f"Saved RSA timecourse to {out}")


if __name__ == "__main__":
    main()

