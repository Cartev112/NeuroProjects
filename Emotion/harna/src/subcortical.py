"""
Bottom-up subcortical pathway for fast, automatic emotional processing.

Implements amygdala-like threat detection and salience processing
with temporal precedence constraints (< 100ms).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SubcorticalResponse:
    """Response from subcortical pathway."""
    threat_level: float
    arousal: float
    salience: float
    processing_time_ms: float
    activation_pattern: np.ndarray
    
    def __repr__(self) -> str:
        return (
            f"SubcorticalResponse(\n"
            f"  Threat: {self.threat_level:.3f}\n"
            f"  Arousal: {self.arousal:.3f}\n"
            f"  Salience: {self.salience:.3f}\n"
            f"  Processing Time: {self.processing_time_ms:.1f}ms\n"
            f")"
        )


class ThreatDetector(nn.Module):
    """
    Amygdala-like threat detection network.
    
    Fast, automatic detection of threatening stimuli using
    coarse visual/auditory features.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.3
    ):
        """
        Initialize threat detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer: threat probability
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Temporal constraint: fast processing
        self.max_processing_time_ms = 100.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through threat detector.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Threat probability [batch_size, 1]
        """
        return self.network(x)
    
    def detect_threat(self, features: np.ndarray) -> float:
        """
        Detect threat level from stimulus features.
        
        Args:
            features: Stimulus features
            
        Returns:
            Threat level (0-1)
        """
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            threat = self.forward(x).item()
        
        return threat


class SalienceDetector(nn.Module):
    """
    Salience detection network.
    
    Identifies emotionally relevant stimuli that warrant attention.
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        """Initialize salience detector."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute salience."""
        return self.network(x)
    
    def compute_salience(self, features: np.ndarray) -> float:
        """Compute salience from features."""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            salience = self.forward(x).item()
        
        return salience


class ArousalModulator(nn.Module):
    """
    Arousal modulation network.
    
    Computes arousal level based on stimulus intensity and threat.
    """
    
    def __init__(self, input_dim: int = 512):
        """Initialize arousal modulator."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Arousal can be positive or negative
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute arousal."""
        return self.network(x)
    
    def compute_arousal(self, features: np.ndarray, threat: float) -> float:
        """
        Compute arousal level.
        
        Args:
            features: Stimulus features
            threat: Threat level
            
        Returns:
            Arousal level (-1 to 1)
        """
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            base_arousal = self.forward(x).item()
            
            # Threat amplifies arousal
            arousal = base_arousal + 0.5 * threat
            arousal = np.clip(arousal, -1, 1)
        
        return arousal


class SubcorticalPathway:
    """
    Complete subcortical processing pathway.
    
    Integrates threat detection, salience, and arousal modulation
    for fast, automatic emotional responses.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        device: str = 'cpu'
    ):
        """
        Initialize subcortical pathway.
        
        Args:
            input_dim: Input feature dimension
            device: Computing device
        """
        self.input_dim = input_dim
        self.device = device
        
        # Initialize components
        self.threat_detector = ThreatDetector(input_dim).to(device)
        self.salience_detector = SalienceDetector(input_dim).to(device)
        self.arousal_modulator = ArousalModulator(input_dim).to(device)
        
        # Processing time tracking
        self.processing_times = []
    
    def process(
        self,
        features: np.ndarray,
        context: Optional[Dict] = None
    ) -> SubcorticalResponse:
        """
        Process stimulus through subcortical pathway.
        
        Args:
            features: Stimulus features
            context: Optional context information
            
        Returns:
            SubcorticalResponse object
        """
        import time
        
        start_time = time.time()
        
        # Ensure features are correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Detect threat
        threat_level = self.threat_detector.detect_threat(features[0])
        
        # Compute salience
        salience = self.salience_detector.compute_salience(features[0])
        
        # Modulate arousal
        arousal = self.arousal_modulator.compute_arousal(features[0], threat_level)
        
        # Context modulation
        if context is not None:
            # Social context reduces threat perception
            if context.get('social', False):
                threat_level *= 0.8
            
            # Familiar context reduces arousal
            if context.get('familiar', False):
                arousal *= 0.7
        
        # Compute activation pattern (simplified)
        activation_pattern = np.array([threat_level, arousal, salience])
        
        # Processing time
        processing_time_ms = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time_ms)
        
        return SubcorticalResponse(
            threat_level=threat_level,
            arousal=arousal,
            salience=salience,
            processing_time_ms=processing_time_ms,
            activation_pattern=activation_pattern
        )
    
    def train_threat_detector(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True
    ):
        """
        Train threat detector on labeled data.
        
        Args:
            train_features: Training features [n_samples, input_dim]
            train_labels: Threat labels [n_samples] (0 or 1)
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress
        """
        # Convert to tensors
        X = torch.FloatTensor(train_features).to(self.device)
        y = torch.FloatTensor(train_labels).unsqueeze(1).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.threat_detector.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Training loop
        n_samples = len(X)
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            for i in range(n_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.threat_detector(X_batch)
                loss = criterion(predictions, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def evaluate_threat_detector(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate threat detector performance.
        
        Args:
            test_features: Test features
            test_labels: Test labels
            
        Returns:
            Dictionary with metrics
        """
        with torch.no_grad():
            X = torch.FloatTensor(test_features).to(self.device)
            y = test_labels
            
            predictions = self.threat_detector(X).cpu().numpy().flatten()
            
            # Binary predictions
            binary_preds = (predictions > 0.5).astype(int)
            
            # Metrics
            accuracy = np.mean(binary_preds == y)
            
            # True positives, false positives, etc.
            tp = np.sum((binary_preds == 1) & (y == 1))
            fp = np.sum((binary_preds == 1) & (y == 0))
            tn = np.sum((binary_preds == 0) & (y == 0))
            fn = np.sum((binary_preds == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
    
    def get_average_processing_time(self) -> float:
        """Get average processing time in milliseconds."""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times)
    
    def reset_processing_times(self):
        """Reset processing time tracking."""
        self.processing_times = []
    
    def save_models(self, filepath: str):
        """Save trained models."""
        torch.save({
            'threat_detector': self.threat_detector.state_dict(),
            'salience_detector': self.salience_detector.state_dict(),
            'arousal_modulator': self.arousal_modulator.state_dict()
        }, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.threat_detector.load_state_dict(checkpoint['threat_detector'])
        self.salience_detector.load_state_dict(checkpoint['salience_detector'])
        self.arousal_modulator.load_state_dict(checkpoint['arousal_modulator'])
        print(f"Models loaded from {filepath}")


def generate_synthetic_threat_data(
    n_samples: int = 1000,
    input_dim: int = 512,
    threat_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for threat detector training.
    
    Args:
        n_samples: Number of samples
        input_dim: Feature dimension
        threat_ratio: Ratio of threatening stimuli
        
    Returns:
        Tuple of (features, labels)
    """
    n_threat = int(n_samples * threat_ratio)
    n_safe = n_samples - n_threat
    
    # Threatening stimuli: higher variance, specific patterns
    threat_features = np.random.randn(n_threat, input_dim) * 2.0
    threat_features[:, :50] += 3.0  # Specific threat-related features
    
    # Safe stimuli: lower variance
    safe_features = np.random.randn(n_safe, input_dim) * 0.5
    
    # Combine
    features = np.vstack([threat_features, safe_features])
    labels = np.hstack([np.ones(n_threat), np.zeros(n_safe)])
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    features = features[perm]
    labels = labels[perm]
    
    return features, labels
