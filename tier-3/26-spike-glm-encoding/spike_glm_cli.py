import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold


def parse_args():
    p = argparse.ArgumentParser(description="Poisson GLM for spike trains")
    p.add_argument("--X", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(args.X)
    y = np.load(args.y)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    ll = []
    for tr, te in kf.split(X):
        model = PoissonRegressor(alpha=1.0, max_iter=1000)
        model.fit(X[tr], y[tr])
        mu = model.predict(X[te])
        mu = np.clip(mu, 1e-6, None)
        # Poisson log-likelihood
        ll.append(float(np.mean(y[te] * np.log(mu) - mu)))
    pd.Series(ll).to_csv(out / "loglik_cv.csv", index=False)
    print(np.mean(ll))


if __name__ == "__main__":
    main()

