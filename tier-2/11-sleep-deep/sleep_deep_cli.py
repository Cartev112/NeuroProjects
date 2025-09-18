import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        z = self.net(x)
        z = z.squeeze(-1)
        return self.fc(z)


def parse_args():
    p = argparse.ArgumentParser(description="Sleep staging deep model")
    p.add_argument("--data_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.data_npz)
    X = arr["epochs"]  # (N, C, T)
    y = arr["labels"].astype(int)
    n_classes = int(np.max(y)) + 1

    # Simple split for demo
    n = len(y)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    train_idx = idx[: int(0.8 * n)]
    test_idx = idx[int(0.8 * n) :]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    model = SimpleCNN(in_ch=X.shape[1], n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        print(f"epoch {epoch+1}: loss={total/len(train_idx):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        y_pred = logits.argmax(dim=1).cpu().numpy()
    report = classification_report(y_test.numpy(), y_pred, output_dict=True)
    cm = confusion_matrix(y_test.numpy(), y_pred)
    pd.DataFrame(report).to_csv(out / "classification_report.csv")
    pd.DataFrame(cm).to_csv(out / "confusion_matrix.csv", index=False)
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()

