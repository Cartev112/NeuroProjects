import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import Xdawn


def parse_args():
    p = argparse.ArgumentParser(description="P300 detection with XDAWN+LDA")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_components", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arr = np.load(args.data_npz)
    X, y = arr["X"], arr["y"].astype(int)

    xd = Xdawn(n_components=args.n_components)
    # Flatten time after spatial filtering for LDA
    X_xd = xd.fit_transform(X, y)
    n = X_xd.shape[0]
    X_feat = X_xd.reshape(n, -1)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(lda, X_feat, y, cv=cv, scoring="roc_auc")
    pd.Series(scores).to_csv(out / "auc_cv.csv", index=False)
    print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()

