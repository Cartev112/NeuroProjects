"""Mirror neuron system for action-to-emotion mapping."""

import torch
import torch.nn as nn
import numpy as np


class MirrorNeuronSystem(nn.Module):
    """
    Mirror neuron system that maps observed actions to internal emotions.
    
    Simulates the neural mechanism for emotional resonance.
    """
    
    def __init__(self, input_dim: int = 512, emotion_dim: int = 64):
        """Initialize mirror neuron system."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, emotion_dim),
            nn.Tanh()
        )
    
    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """Map action features to emotional response."""
        return self.network(action_features)
    
    def simulate(self, action_features: np.ndarray) -> np.ndarray:
        """Simulate internal emotional response to observed action."""
        with torch.no_grad():
            x = torch.FloatTensor(action_features).unsqueeze(0)
            emotion = self.forward(x)
            return emotion.cpu().numpy()[0]
    
    def train(self, action_features: np.ndarray, emotion_labels: np.ndarray, 
              epochs: int = 50, lr: float = 0.001):
        """Train mirror neuron system."""
        X = torch.FloatTensor(action_features)
        y = torch.FloatTensor(emotion_labels)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
