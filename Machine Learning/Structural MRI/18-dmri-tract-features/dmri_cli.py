import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


def parse_args():
    p = argparse.ArgumentParser(description="dMRI tract features classifier")
    p.add_argument("--features", required=True)
    p.add_argument("--label_col", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    y = df[args.label_col].astype(int)
    X = df.drop(columns=[args.label_col])
    clf = GradientBoostingClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    pd.Series(scores).to_csv(out / "ba_cv.csv", index=False)
    print(scores.mean())


if __name__ == "__main__":
    main()

