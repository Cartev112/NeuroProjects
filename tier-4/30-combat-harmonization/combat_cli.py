import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def combat_harmonize(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    # Simple mean/variance standardization per batch as a placeholder
    Xh = np.zeros_like(X)
    for b in np.unique(batch):
        m = batch == b
        mu = X[m].mean(0)
        sd = X[m].std(0) + 1e-6
        Xh[m] = (X[m] - mu) / sd
    return Xh


def parse_args():
    p = argparse.ArgumentParser(description="ComBat harmonization study")
    p.add_argument("--features", required=True)
    p.add_argument("--site_col", required=True)
    p.add_argument("--label_col", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    y = df[args.label_col].astype(int).to_numpy()
    site = df[args.site_col].to_numpy()
    X = df.drop(columns=[args.label_col, args.site_col]).to_numpy()

    logo = LeaveOneGroupOut()
    base_scores, harm_scores = [], []
    for tr, te in logo.split(X, y, groups=site):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
        base_scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))

        Xh = combat_harmonize(X, site)
        clf2 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xh[tr], y[tr])
        harm_scores.append(balanced_accuracy_score(y[te], clf2.predict(Xh[te])))

    pd.Series({"base_mean": np.mean(base_scores), "harm_mean": np.mean(harm_scores)}).to_csv(out / "results.csv")
    print(np.mean(base_scores), np.mean(harm_scores))


if __name__ == "__main__":
    main()

