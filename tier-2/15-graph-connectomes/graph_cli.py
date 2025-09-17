import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser(description="Graph features from FC")
    p.add_argument("--corr", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def graph_features(corrs: np.ndarray) -> np.ndarray:
    # corrs: (N, R, R)
    n, r, _ = corrs.shape
    feats = []
    for i in range(n):
        c = np.abs(corrs[i])
        deg = c.sum(axis=1)
        strength = np.square(c).sum(axis=1)
        eigvals = np.linalg.eigvalsh(c)
        feats.append(np.hstack([deg, strength, eigvals]))
    return np.asarray(feats)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    corrs = np.load(args.corr)
    y = np.load(args.labels).astype(int)
    X = graph_features(corrs)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    pd.Series(scores).to_csv(out / "ba_cv.csv", index=False)
    print(f"Balanced acc: {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()

