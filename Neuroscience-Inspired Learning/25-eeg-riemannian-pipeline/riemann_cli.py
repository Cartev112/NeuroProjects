import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def parse_args():
    p = argparse.ArgumentParser(description="Riemannian EEG pipeline")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arr = np.load(args.data_npz)
    X, y = arr["X"], arr["y"].astype(int)
    pipe = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    pd.Series(scores).to_csv(out / "ba_cv.csv", index=False)
    print(scores.mean())


if __name__ == "__main__":
    main()

