import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def parse_args():
    p = argparse.ArgumentParser(description="fMRI visual encoding")
    p.add_argument("--X", required=True)
    p.add_argument("--Y", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(args.X)
    Y = np.load(args.Y)
    n_vox = Y.shape[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    alphas = np.logspace(-2, 3, 10)
    r2 = np.zeros(n_vox)
    for v in range(n_vox):
        y = Y[:, v]
        y_true, y_pred = [], []
        for tr, te in cv.split(X):
            model = RidgeCV(alphas=alphas, cv=3)
            model.fit(X[tr], y[tr])
            y_pred.append(model.predict(X[te]))
            y_true.append(y[te])
        r2[v] = r2_score(np.concatenate(y_true), np.concatenate(y_pred))
    pd.DataFrame({"voxel": np.arange(n_vox), "r2": r2}).to_csv(out / "r2_voxels.csv", index=False)
    print(f"Saved voxelwise R2 to {out}")


if __name__ == "__main__":
    main()

