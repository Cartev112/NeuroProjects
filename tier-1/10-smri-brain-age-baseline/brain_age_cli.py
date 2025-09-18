import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor


def parse_args():
    p = argparse.ArgumentParser(description="Brain age baseline from sMRI features")
    p.add_argument("--features_csv", required=True)
    p.add_argument("--age_col", required=True)
    p.add_argument("--model", choices=["ridge", "xgb"], default="ridge")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def bias_correct(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # Fit y_pred = a*y_true + b and compute corrected predictions
    X = np.vstack([y_true, np.ones_like(y_true)]).T
    a, b = np.linalg.lstsq(X, y_pred, rcond=None)[0]
    y_pred_corr = (y_pred - b) / (a + 1e-12)
    return y_pred_corr


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    y = df[args.age_col].to_numpy().astype(float)
    X = df.drop(columns=[args.age_col])

    if args.model == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
    else:
        model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    mae = mean_absolute_error(y, y_pred)
    y_pred_corr = bias_correct(y, y_pred)
    mae_corr = mean_absolute_error(y, y_pred_corr)
    delta = y_pred_corr - y

    pd.DataFrame({"y_true": y, "y_pred": y_pred, "y_pred_corr": y_pred_corr, "delta": delta}).to_csv(out / "predictions.csv", index=False)
    with open(out / "metrics.txt", "w") as f:
        f.write(f"MAE: {mae:.3f}\n")
        f.write(f"MAE_bias_corrected: {mae_corr:.3f}\n")

    print(f"Saved brain age outputs to {out}")


if __name__ == "__main__":
    main()

