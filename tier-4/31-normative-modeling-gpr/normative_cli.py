import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold


def parse_args():
    p = argparse.ArgumentParser(description="Normative modeling with GPR")
    p.add_argument("--features", required=True)
    p.add_argument("--target_col", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    y = df[args.target_col].to_numpy().astype(float)
    X = df.drop(columns=[args.target_col]).to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = np.zeros_like(y)
    y_std = np.zeros_like(y)
    for tr, te in kf.split(X):
        kern = 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kern, alpha=1e-6, normalize_y=True)
        gpr.fit(X[tr], y[tr])
        m, s = gpr.predict(X[te], return_std=True)
        y_pred[te] = m
        y_std[te] = s
    z = (y - y_pred) / (y_std + 1e-6)
    pd.DataFrame({"y": y, "y_pred": y_pred, "y_std": y_std, "z": z}).to_csv(out / "normative.csv", index=False)
    print(f"Saved normative outputs to {out}")


if __name__ == "__main__":
    main()

