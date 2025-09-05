import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def parse_args():
    p = argparse.ArgumentParser(description="EEG+fNIRS fusion classifier")
    p.add_argument("--eeg", required=True)
    p.add_argument("--fnirs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X_eeg = np.load(args.eeg)
    X_hbo = np.load(args.fnirs)
    y = np.load(args.labels).astype(int)
    # Early fusion
    X = np.hstack([X_eeg, X_hbo])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    pd.Series(scores).to_csv(out / "fusion_ba_cv.csv", index=False)
    print(scores.mean())


if __name__ == "__main__":
    main()

