import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


def parse_args():
    p = argparse.ArgumentParser(description="ICA component labeling")
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
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    pd.Series(scores).to_csv(out / "ba_cv.csv", index=False)
    print(f"Balanced acc: {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()

