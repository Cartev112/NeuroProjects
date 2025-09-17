import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from mne.decoding import CSP


def parse_args():
    p = argparse.ArgumentParser(description="Motor imagery CSP+LDA baseline")
    p.add_argument("--data_root", required=True, help="Path to epochs npz containing X,y")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_components", type=int, default=6)
    return p.parse_args()


def load_epochs_npz(path: str):
    arr = np.load(Path(path) / "mi_epochs.npz")
    return arr["X"], arr["y"], arr.get("sfreq", None)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_epochs_npz(args.data_root)
    # X shape: (n_trials, n_channels, n_times)
    csp = CSP(n_components=args.n_components, reg=None, log=True, cov_est="epoch")
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    pipe = Pipeline([("csp", csp), ("lda", lda)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    pd.DataFrame({"balanced_accuracy": scores}).to_csv(out / "cv_scores.csv", index=False)
    print(f"Mean balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()

