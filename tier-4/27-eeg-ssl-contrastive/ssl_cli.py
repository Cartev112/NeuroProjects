import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def augment(x, p_drop=0.1):
    # x: (C, T)
    x = x.copy()
    # Jitter
    x += 0.01 * np.random.randn(*x.shape)
    # Scaling
    x *= (0.9 + 0.2 * np.random.rand())
    # Time masking
    T = x.shape[1]
    t0 = np.random.randint(0, max(1, T - T // 10))
    x[:, t0 : t0 + T // 10] = 0
    # Channel dropout
    mask = np.random.rand(x.shape[0]) > p_drop
    x = x * mask[:, None]
    return x


class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 9, padding=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.out = nn.Linear(128, 128)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return nn.functional.normalize(self.out(z), dim=1)


def nt_xent(z1, z2, tau=0.1):
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T
    N = z1.size(0)
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels + N, labels])
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim[~mask].view(2 * N, -1)
    logits = sim / tau
    return nn.functional.cross_entropy(logits, labels)


def parse_args():
    p = argparse.ArgumentParser(description="EEG self-supervised contrastive pretraining")
    p.add_argument("--unlabeled_npz", required=True)
    p.add_argument("--labeled_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    U = np.load(args.unlabeled_npz)["X"]  # (N, C, T)
    L = np.load(args.labeled_npz)
    X_l, y = L["X_l"], L["y"].astype(int)

    in_ch = U.shape[1]
    enc = Encoder(in_ch)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

    # Pretraining
    for epoch in range(args.epochs):
        idx = np.random.permutation(len(U))
        for i in range(0, len(U), args.batch_size):
            batch = U[idx[i : i + args.batch_size]]
            x1 = np.stack([augment(x) for x in batch])
            x2 = np.stack([augment(x) for x in batch])
            x1 = torch.tensor(x1, dtype=torch.float32)
            x2 = torch.tensor(x2, dtype=torch.float32)
            opt.zero_grad()
            z1, z2 = enc(x1), enc(x2)
            loss = nt_xent(z1, z2)
            loss.backward(); opt.step()
        print(f"epoch {epoch+1}")

    # Linear probe
    with torch.no_grad():
        Z = enc(torch.tensor(X_l, dtype=torch.float32)).numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(Z, y, test_size=0.2, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    (out / "metrics.txt").write_text(f"linear_probe_acc: {acc:.3f}\n")
    print(f"Linear probe acc: {acc:.3f}")


if __name__ == "__main__":
    main()

