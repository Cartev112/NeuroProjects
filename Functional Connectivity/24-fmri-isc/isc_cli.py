import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compute inter-subject correlation")
    p.add_argument("--ts", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ts = np.load(args.ts)  # (S, T, R)
    S, T, R = ts.shape
    isc = np.zeros(R)
    for r in range(R):
        X = ts[:, :, r]
        mean_other = (np.sum(X, axis=0, keepdims=True) - X) / (S - 1)
        corr = [np.corrcoef(X[i], mean_other[i].ravel())[0, 1] for i in range(S)]
        isc[r] = float(np.mean(corr))
    pd.DataFrame({"roi": np.arange(R), "isc": isc}).to_csv(out / "isc_roi.csv", index=False)
    print(f"Saved ISC to {out}")


if __name__ == "__main__":
    main()

