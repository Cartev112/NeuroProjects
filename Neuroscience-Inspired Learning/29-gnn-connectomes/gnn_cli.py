import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class GCN(nn.Module):
    def __init__(self, n_nodes, in_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.cls = nn.Linear(64, n_classes)
        self.I = torch.eye(n_nodes)

    def forward(self, A, X):
        A_hat = A + self.I.to(A.device)
        D = torch.diag(1.0 / torch.sqrt(A_hat.sum(1) + 1e-6))
        An = D @ A_hat @ D
        H = torch.relu(self.fc1(An @ X))
        H = torch.relu(self.fc2(An @ H))
        g = H.mean(dim=0)  # global mean pooling
        return self.cls(g)


def parse_args():
    p = argparse.ArgumentParser(description="GNN on connectomes")
    p.add_argument("--A", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    A = np.load(args.A)  # (N, R, R)
    y = np.load(args.labels).astype(int)
    N, R, _ = A.shape
    # Node features: degree only for simplicity
    X = A.sum(axis=2)  # (N, R)
    n_classes = int(y.max()) + 1

    idx = np.arange(N); np.random.default_rng(0).shuffle(idx)
    tr = idx[: int(0.8 * N)]; te = idx[int(0.8 * N) :]

    model = GCN(n_nodes=R, in_dim=1, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train(); opt.zero_grad()
        loss = 0.0
        for i in tr:
            Ai = torch.tensor(A[i], dtype=torch.float32)
            Xi = torch.tensor(X[i][:, None], dtype=torch.float32)
            yi = torch.tensor([y[i]], dtype=torch.long)
            logits = model(Ai, Xi)
            loss = loss + crit(logits.unsqueeze(0), yi)
        loss.backward(); opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in te:
            Ai = torch.tensor(A[i], dtype=torch.float32)
            Xi = torch.tensor(X[i][:, None], dtype=torch.float32)
            logits = model(Ai, Xi)
            preds.append(int(logits.argmax().item()))
    acc = accuracy_score(y[te], np.array(preds))
    (out / "metrics.txt").write_text(f"accuracy: {acc:.3f}\n")
    print(acc)


if __name__ == "__main__":
    main()

