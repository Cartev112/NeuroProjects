import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score


def parse_args():
    p = argparse.ArgumentParser(description="fMRI task decoding")
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(args.features)
    y = np.load(args.labels).astype(int)
    base = LinearSVC(dual=False, class_weight="balanced")
    param = {"C": np.logspace(-3, 3, 7)}
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = GridSearchCV(base, {"C": param["C"]}, cv=inner, scoring="balanced_accuracy")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for tr, te in outer.split(X, y):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        scores.append(balanced_accuracy_score(y[te], y_pred))
    pd.Series(scores).to_csv(out / "balanced_accuracy_cv.csv", index=False)
    print(f"Balanced acc: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


if __name__ == "__main__":
    main()

