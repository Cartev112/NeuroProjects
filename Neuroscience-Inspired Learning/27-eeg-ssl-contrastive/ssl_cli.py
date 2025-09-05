# EEG Self-Supervised Learning with Contrastive Learning
# This script trains a neural network to learn useful representations from unlabeled EEG data
# using contrastive learning, then evaluates the learned representations on a labeled task.

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
    """
    Apply data augmentation to a single EEG sample.
    
    Data augmentation creates slightly modified versions of the same data,
    which helps the model learn robust features that work across variations.
    
    Args:
        x: EEG data with shape (C, T) where C = channels, T = time points
        p_drop: Probability of dropping each channel (default 10%)
    
    Returns:
        Augmented EEG data with the same shape as input
    """
    # x: (C, T) - C channels (electrodes), T time points
    x = x.copy()  # Don't modify the original data
    
    # Jitter: Add small random noise to simulate natural signal variations
    x += 0.01 * np.random.randn(*x.shape)
    
    # Scaling: Randomly scale amplitude between 0.9x and 1.1x
    # This simulates variations in signal strength across recordings
    x *= (0.9 + 0.2 * np.random.rand())
    
    # Time masking: Zero out a random 10% segment of the time series
    # This forces the model to learn from partial information
    T = x.shape[1]
    t0 = np.random.randint(0, max(1, T - T // 10))  # Random start position
    x[:, t0 : t0 + T // 10] = 0  # Mask 10% of time points
    
    # Channel dropout: Randomly drop entire channels with probability p_drop
    # This simulates missing or faulty electrodes
    mask = np.random.rand(x.shape[0]) > p_drop
    x = x * mask[:, None]  # Apply mask to all time points of selected channels
    
    return x


class Encoder(nn.Module):
    """
    Neural network that converts raw EEG signals into compact feature vectors.
    
    This encoder uses convolutional layers to process the temporal patterns in EEG data
    and outputs a normalized 128-dimensional representation (embedding) for each input.
    """
    def __init__(self, in_ch):
        """
        Args:
            in_ch: Number of input channels (EEG electrodes)
        """
        super().__init__()
        # Convolutional neural network to process EEG time series
        self.net = nn.Sequential(
            # First conv layer: in_ch -> 64 features, kernel size 9, padding to preserve length
            nn.Conv1d(in_ch, 64, 9, padding=4), 
            nn.ReLU(),  # Non-linear activation
            nn.MaxPool1d(2),  # Downsample by factor of 2
            
            # Second conv layer: 64 -> 128 features
            nn.Conv1d(64, 128, 9, padding=4), 
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pool to single time point (global average pooling)
        )
        # Final projection layer to output 128-dimensional embeddings
        self.out = nn.Linear(128, 128)

    def forward(self, x):
        """
        Convert EEG input to normalized embedding vector.
        
        Args:
            x: Input EEG with shape (batch_size, channels, time_points)
        
        Returns:
            Normalized embeddings with shape (batch_size, 128)
        """
        z = self.net(x).squeeze(-1)  # Remove the time dimension (now just 1)
        # Normalize to unit length - this is important for contrastive learning
        # so that we measure similarity by angle, not magnitude
        return nn.functional.normalize(self.out(z), dim=1)


def nt_xent(z1, z2, tau=0.1):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
    
    This is the core of contrastive learning. The idea:
    - z1 and z2 are two augmented versions of the same batch of data
    - For each sample, its two augmented versions should be similar (positive pair)
    - But it should be different from all other samples (negative pairs)
    
    The model learns by pulling positive pairs together and pushing negative pairs apart.
    
    Args:
        z1: Embeddings of first augmented view, shape (N, 128)
        z2: Embeddings of second augmented view, shape (N, 128)
        tau: Temperature parameter (lower = harder negatives), default 0.1
    
    Returns:
        Contrastive loss value (scalar)
    """
    # Combine both views into a single batch of size 2N
    z = torch.cat([z1, z2], dim=0)  # Shape: (2N, 128)
    
    # Compute similarity matrix: how similar is each embedding to every other?
    sim = z @ z.T  # Shape: (2N, 2N), dot product = cosine similarity for normalized vectors
    
    N = z1.size(0)  # Original batch size
    
    # Create labels: for each sample i in z1, its positive pair is at position i+N in z2
    # and vice versa
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels + N, labels])  # [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
    
    # Remove self-similarities (diagonal) - we don't want to compare a sample with itself
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim[~mask].view(2 * N, -1)  # Shape: (2N, 2N-1)
    
    # Apply temperature scaling (makes the model focus on hard negatives)
    logits = sim / tau
    
    # Compute cross-entropy: model should predict which sample is the positive pair
    return nn.functional.cross_entropy(logits, labels)


def parse_args():
    """
    Parse command-line arguments for the training script.
    
    Returns:
        Parsed arguments with the following fields:
        - unlabeled_npz: Path to unlabeled EEG data for pretraining
        - labeled_npz: Path to labeled EEG data for evaluation
        - out_dir: Directory to save results
        - epochs: Number of training epochs (default: 10)
        - batch_size: Batch size for training (default: 128)
    """
    p = argparse.ArgumentParser(description="EEG self-supervised contrastive pretraining")
    p.add_argument("--unlabeled_npz", required=True, help="Path to unlabeled EEG data (.npz file)")
    p.add_argument("--labeled_npz", required=True, help="Path to labeled EEG data (.npz file)")
    p.add_argument("--out_dir", required=True, help="Output directory for results")
    p.add_argument("--epochs", type=int, default=10, help="Number of pretraining epochs")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    return p.parse_args()


def main():
    """
    Main training pipeline:
    1. Load unlabeled and labeled EEG data
    2. Pretrain encoder using contrastive learning on unlabeled data
    3. Evaluate learned representations using a linear classifier on labeled data
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory for results
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # U: Unlabeled data for self-supervised pretraining, shape (N, C, T)
    #    N = number of samples, C = channels (electrodes), T = time points
    U = np.load(args.unlabeled_npz)["X"]
    
    # L: Labeled data for evaluation
    L = np.load(args.labeled_npz)
    X_l, y = L["X_l"], L["y"].astype(int)  # Features and labels

    # Initialize the encoder network
    in_ch = U.shape[1]  # Number of EEG channels
    enc = Encoder(in_ch)
    
    # Adam optimizer with learning rate 0.001
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

    # ===== PHASE 1: Self-Supervised Pretraining =====
    # Train the encoder to learn useful representations from unlabeled data
    # using contrastive learning (no labels needed!)
    print("Starting self-supervised pretraining...")
    for epoch in range(args.epochs):
        # Shuffle the data each epoch
        idx = np.random.permutation(len(U))
        
        # Process data in batches
        for i in range(0, len(U), args.batch_size):
            # Get a batch of unlabeled samples
            batch = U[idx[i : i + args.batch_size]]
            
            # Create two different augmented versions of each sample
            # These are "positive pairs" - different views of the same data
            x1 = np.stack([augment(x) for x in batch])
            x2 = np.stack([augment(x) for x in batch])
            
            # Convert to PyTorch tensors
            x1 = torch.tensor(x1, dtype=torch.float32)
            x2 = torch.tensor(x2, dtype=torch.float32)
            
            # Standard PyTorch training loop
            opt.zero_grad()  # Reset gradients
            z1, z2 = enc(x1), enc(x2)  # Get embeddings for both views
            loss = nt_xent(z1, z2)  # Compute contrastive loss
            loss.backward()  # Compute gradients
            opt.step()  # Update weights
        
        print(f"epoch {epoch+1}/{args.epochs} completed")

    # ===== PHASE 2: Linear Probe Evaluation =====
    # Test how good the learned representations are by training a simple
    # linear classifier on top of the frozen encoder
    print("\nEvaluating with linear probe...")
    
    # Extract features from labeled data (no gradient computation needed)
    with torch.no_grad():
        Z = enc(torch.tensor(X_l, dtype=torch.float32)).numpy()
    
    # Split into train/test sets (80/20 split, stratified by class)
    X_tr, X_te, y_tr, y_te = train_test_split(Z, y, test_size=0.2, random_state=0, stratify=y)
    
    # Train a simple logistic regression classifier on the learned features
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    
    # Evaluate accuracy on test set
    acc = accuracy_score(y_te, clf.predict(X_te))
    
    # Save and print results
    (out / "metrics.txt").write_text(f"linear_probe_acc: {acc:.3f}\n")
    print(f"Linear probe accuracy: {acc:.3f}")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

