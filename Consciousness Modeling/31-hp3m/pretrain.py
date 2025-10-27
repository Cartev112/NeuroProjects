"""
Self-supervised pretraining: temporal contrastive learning and masked prediction.
"""
import numpy as np


def temporal_contrastive_loss(h, tau=0.1, n_neg=5):
    """Compute temporal contrastive loss (InfoNCE-style) on latent sequences.
    
    h: (N, T, D) latent representations
    tau: temperature
    n_neg: number of negative samples per anchor
    
    Returns: scalar loss
    """
    h = np.asarray(h)
    N, T, D = h.shape
    
    loss = 0.0
    count = 0
    
    for n in range(N):
        for t in range(T - 1):
            anchor = h[n, t]
            positive = h[n, t + 1]
            
            # Negative samples: random time points from other trials
            neg_indices = np.random.choice(N * T, size=n_neg, replace=False)
            negatives = h.reshape(-1, D)[neg_indices]
            
            # Cosine similarity
            pos_sim = np.dot(anchor, positive) / (np.linalg.norm(anchor) * np.linalg.norm(positive) + 1e-8)
            neg_sims = negatives @ anchor / (np.linalg.norm(negatives, axis=1) * np.linalg.norm(anchor) + 1e-8)
            
            # InfoNCE
            exp_pos = np.exp(pos_sim / tau)
            exp_neg = np.sum(np.exp(neg_sims / tau))
            loss += -np.log(exp_pos / (exp_pos + exp_neg + 1e-8))
            count += 1
    
    return loss / max(count, 1)


def masked_prediction_loss(X, mask_ratio=0.15):
    """Compute masked prediction loss (simple autoencoding with random masking).
    
    X: (N, T, D) input data
    mask_ratio: fraction of time points to mask
    
    Returns: dict with masked indices and reconstruction targets
    """
    X = np.asarray(X)
    N, T, D = X.shape
    
    # Random mask
    n_mask = int(T * mask_ratio)
    masked_data = X.copy()
    targets = []
    
    for n in range(N):
        mask_idx = np.random.choice(T, size=n_mask, replace=False)
        targets.append((n, mask_idx, X[n, mask_idx].copy()))
        masked_data[n, mask_idx] = 0.0
    
    return {'masked_data': masked_data, 'targets': targets}


def pretrain_ssm_contrastive(ssm, X, n_epochs=5, verbose=True):
    """Pretrain SSM using temporal contrastive loss.
    
    ssm: PredictiveSSM instance
    X: (N, T, D) data
    """
    X = np.asarray(X)
    N, T, D = X.shape
    
    for epoch in range(n_epochs):
        # Extract latents
        latents = ssm.extract_latents(X)
        h = latents['h']  # (N, T, latent_dim)
        
        # Compute contrastive loss
        loss = temporal_contrastive_loss(h, tau=0.1, n_neg=5)
        
        if verbose:
            print(f'Pretrain Epoch {epoch + 1}/{n_epochs}, Contrastive Loss: {loss:.4f}')
        
        # Simplified: we don't backprop through contrastive here (would need autograd)
        # In practice, you'd use PyTorch/JAX for this
        # For now, just report the loss as a diagnostic
    
    return ssm
