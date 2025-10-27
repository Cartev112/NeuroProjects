import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


class ShallowNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        self.conv_spat = nn.Conv2d(40, 40, (in_ch, 1), groups=40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(40, n_classes)

    def forward(self, x):
        # x: (N, C, T) -> (N,1,C,T)
        x = x.unsqueeze(1)
        x = torch.square(torch.tanh(self.conv_spat(self.conv_time(x))))
        x = self.pool(x)
        x = self.drop(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


def parse_args():
    p = argparse.ArgumentParser(description="Braindecode-like benchmark")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    return p.parse_args()


def train_eval(model, train_loader, X_test, y_test, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1).cpu().numpy()
    return accuracy_score(y_test.numpy(), y_pred)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arr = np.load(args.data_npz)
    X = torch.tensor(arr["X"], dtype=torch.float32)
    y = torch.tensor(arr["y"].astype(int))
    n_classes = int(y.max().item()) + 1
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    results = {}
    model = ShallowNet(in_ch=X.shape[1], n_classes=n_classes)
    results["ShallowNet"] = train_eval(model, train_loader, X_te, y_te, epochs=args.epochs)
    pd.Series(results).to_csv(out / "benchmark.csv")
    print(results)


if __name__ == "__main__":
    main()

