import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Source localization benchmark (simulated)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--snr_db", nargs=3, type=float, default=[-10, 0, 10])
    return p.parse_args()


def simulate_data(n_sensors=64, n_sources=1, snr_db=0.0):
    rng = np.random.default_rng(0)
    leadfield = rng.normal(size=(n_sensors, n_sources))
    source = rng.normal(size=(n_sources,))
    sensors = leadfield @ source
    noise = rng.normal(scale=np.linalg.norm(sensors) / (10 ** (snr_db / 20)), size=sensors.shape)
    return leadfield, sensors + noise, source


def fit_min_norm(leadfield, sensors, alpha):
    A = leadfield
    inv = np.linalg.pinv(A.T @ A + alpha * np.eye(A.shape[1])) @ A.T
    return inv @ sensors


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for snr in args.snr_db:
        L, y, s = simulate_data(snr_db=snr)
        for alpha in [0.0, 0.1, 1.0, 10.0]:
            s_hat = fit_min_norm(L, y, alpha)
            err = float(np.linalg.norm(s - s_hat))
            rows.append({"snr_db": snr, "alpha": alpha, "l2_error": err})
    pd.DataFrame(rows).to_csv(out / "bench.csv", index=False)
    print(f"Saved benchmark to {out}")


if __name__ == "__main__":
    main()

