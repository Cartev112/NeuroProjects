"""
Variational State-Space Model for Predictive Processing with PE/Precision latents.
Simplified RNN-based SSM with encoder-decoder architecture.
"""
import numpy as np


class PredictiveSSM:
    """Simplified predictive coding SSM with latent PE and precision."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, n_layers: int = 2, lr: float = 0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.lr = lr
        
        # Encoder: input -> latent (prediction, precision, PE)
        self.W_enc = [self._init_weights(input_dim if i == 0 else latent_dim, latent_dim) for i in range(n_layers)]
        self.b_enc = [np.zeros(latent_dim) for _ in range(n_layers)]
        
        # Recurrent dynamics
        self.W_rec = [self._init_weights(latent_dim, latent_dim) for _ in range(n_layers)]
        
        # Decoder: latent -> reconstructed input
        self.W_dec = self._init_weights(latent_dim, input_dim)
        self.b_dec = np.zeros(input_dim)
        
        # Precision network (predicts inverse variance)
        self.W_prec = self._init_weights(latent_dim, 1)
        self.b_prec = np.zeros(1)
        
    def _init_weights(self, n_in, n_out):
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    def encode(self, x, h_prev=None):
        """Encode input to latent state with recurrent dynamics.
        x: (T, D) or (D,)
        h_prev: list of (latent_dim,) for each layer
        Returns: h (list of latent states), prediction, precision, PE
        """
        if x.ndim == 1:
            x = x[None, :]
        T, D = x.shape
        
        if h_prev is None:
            h_prev = [np.zeros(self.latent_dim) for _ in range(self.n_layers)]
        
        h_out = []
        for t in range(T):
            xt = x[t]
            h_new = []
            for i in range(self.n_layers):
                inp = xt if i == 0 else h_new[i - 1]
                pre_act = inp @ self.W_enc[i] + self.b_enc[i] + h_prev[i] @ self.W_rec[i]
                h_i = self._tanh(pre_act)
                h_new.append(h_i)
            h_prev = h_new
            h_out.append(h_new[-1])
        
        h_out = np.array(h_out)  # (T, latent_dim)
        # Prediction from latent
        pred = h_out @ self.W_dec + self.b_dec  # (T, D)
        # Precision (log scale)
        log_prec = h_out @ self.W_prec + self.b_prec  # (T, 1)
        precision = np.exp(np.clip(log_prec, -5, 5))
        # Prediction error
        pe = x - pred
        
        return h_prev, h_out, pred, precision, pe
    
    def fit(self, X, n_epochs=10, batch_size=32, verbose=True):
        """Fit SSM to data X (N, T, D) using simple SGD on reconstruction + precision loss."""
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[None, :, :]
        N, T, D = X.shape
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = X[indices[start:end]]
                
                # Forward pass
                batch_loss = 0.0
                for i in range(batch.shape[0]):
                    h_prev, h_out, pred, prec, pe = self.encode(batch[i])
                    # Negative log likelihood with precision weighting
                    nll = 0.5 * np.sum(prec.ravel() * (pe ** 2)) - 0.5 * np.sum(np.log(prec.ravel() + 1e-8))
                    # Regularization
                    reg = 0.001 * sum(np.sum(w ** 2) for w in self.W_enc + self.W_rec + [self.W_dec, self.W_prec])
                    loss = nll + reg
                    batch_loss += loss
                    
                    # Backward (simplified gradient descent on reconstruction error)
                    # For simplicity, we do a single gradient step per sample
                    grad_dec = -pe.T @ h_out / T
                    grad_prec = -(0.5 * (pe ** 2 - 1.0 / (prec + 1e-8))).T @ h_out / T
                    
                    self.W_dec -= self.lr * grad_dec.T
                    self.b_dec -= self.lr * np.mean(-pe, axis=0)
                    self.W_prec -= self.lr * grad_prec.T
                    self.b_prec -= self.lr * np.mean(-(0.5 * (pe ** 2 - 1.0 / (prec + 1e-8))), axis=0)
                
                epoch_loss += batch_loss
            
            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / N:.4f}')
    
    def extract_latents(self, X):
        """Extract latent states, predictions, precision, PE for each trial.
        X: (N, T, D)
        Returns: dict with h, pred, precision, pe each (N, T, ...)
        """
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[None, :, :]
        N, T, D = X.shape
        
        all_h = []
        all_pred = []
        all_prec = []
        all_pe = []
        
        for i in range(N):
            _, h, pred, prec, pe = self.encode(X[i])
            all_h.append(h)
            all_pred.append(pred)
            all_prec.append(prec)
            all_pe.append(pe)
        
        return {
            'h': np.array(all_h),          # (N, T, latent_dim)
            'pred': np.array(all_pred),    # (N, T, D)
            'precision': np.array(all_prec),  # (N, T, 1)
            'pe': np.array(all_pe),        # (N, T, D)
        }
