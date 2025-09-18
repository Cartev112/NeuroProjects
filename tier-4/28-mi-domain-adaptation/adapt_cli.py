import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def coral_transform(Xs, Xt):
    # Center
    Xs_c = Xs - Xs.mean(0, keepdims=True)
    Xt_c = Xt - Xt.mean(0, keepdims=True)
    # Covariances
    Cs = np.cov(Xs_c, rowvar=False) + 1e-6 * np.eye(Xs.shape[1])
    Ct = np.cov(Xt_c, rowvar=False) + 1e-6 * np.eye(Xt.shape[1])
    # Whiten/color
    Us, Ss, _ = np.linalg.svd(Cs)
    Ut, St, _ = np.linalg.svd(Ct)
    Ws = Us @ np.diag(1.0 / np.sqrt(Ss)) @ Us.T
    Wt = Ut @ np.diag(np.sqrt(St)) @ Ut.T
    return (Xs_c @ Ws) @ Wt + Xt.mean(0, keepdims=True)


def parse_args():
    p = argparse.ArgumentParser(description="CORAL domain adaptation for MI")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    S = np.load(args.source); T = np.load(args.target)
    Xs, ys = S["X_s"], S["y_s"].astype(int)
    Xt, yt = T["X_t"], T["y_t"].astype(int)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xs, ys)
    base = balanced_accuracy_score(yt, clf.predict(Xt))

    Xs_a = coral_transform(Xs, Xt)
    clf2 = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf2.fit(Xs_a, ys)
    adapted = balanced_accuracy_score(yt, clf2.predict(Xt))

    pd.Series({"baseline_ba": base, "adapted_ba": adapted}).to_csv(out / "results.csv")
    print(f"Baseline BA: {base:.3f} | Adapted BA: {adapted:.3f}")


if __name__ == "__main__":
    main()

