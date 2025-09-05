"""
Predictive processing framework for interoceptive predictions.

Implements emotions as predictions about bodily states, with
prediction errors driving learning and emotional experience.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictiveState:
    """State of predictive processing system."""
    prediction: np.ndarray
    observation: np.ndarray
    prediction_error: np.ndarray
    precision: float
    confidence: float
    
    def __repr__(self) -> str:
        return (
            f"PredictiveState(\n"
            f"  Prediction Error Magnitude: {np.linalg.norm(self.prediction_error):.3f}\n"
            f"  Precision: {self.precision:.3f}\n"
            f"  Confidence: {self.confidence:.3f}\n"
            f")"
        )


class InteroceptivePredictor(nn.Module):
    """
    Forward model for predicting bodily states.
    
    Predicts physiological responses (heart rate, skin conductance, etc.)
    based on current context and emotional state.
    """
    
    def __init__(
        self,
        context_dim: int = 128,
        emotion_dim: int = 64,
        body_state_dim: int = 32
    ):
        """
        Initialize interoceptive predictor.
        
        Args:
            context_dim: Context representation dimension
            emotion_dim: Emotional state dimension
            body_state_dim: Bodily state dimension
        """
        super().__init__()
        
        self.body_state_dim = body_state_dim
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(context_dim + emotion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, body_state_dim)
        )
        
        # Precision network (estimates reliability of prediction)
        self.precision_network = nn.Sequential(
            nn.Linear(context_dim + emotion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Positive precision
        )
    
    def forward(
        self,
        context: torch.Tensor,
        emotion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bodily state and precision.
        
        Args:
            context: Context representation
            emotion: Emotional state
            
        Returns:
            Tuple of (prediction, precision)
        """
        combined = torch.cat([context, emotion], dim=1)
        
        prediction = self.predictor(combined)
        precision = self.precision_network(combined)
        
        return prediction, precision


class PredictionErrorComputer:
    """
    Computes prediction errors and updates beliefs.
    
    Implements precision-weighted prediction error minimization.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize prediction error computer.
        
        Args:
            learning_rate: Learning rate for belief updating
        """
        self.learning_rate = learning_rate
    
    def compute_error(
        self,
        prediction: np.ndarray,
        observation: np.ndarray,
        precision: float = 1.0
    ) -> np.ndarray:
        """
        Compute precision-weighted prediction error.
        
        Args:
            prediction: Predicted state
            observation: Observed state
            precision: Precision (inverse variance) of prediction
            
        Returns:
            Prediction error
        """
        error = observation - prediction
        weighted_error = precision * error
        
        return weighted_error
    
    def update_belief(
        self,
        current_belief: np.ndarray,
        prediction_error: np.ndarray,
        precision: float = 1.0
    ) -> np.ndarray:
        """
        Update belief based on prediction error.
        
        Args:
            current_belief: Current belief state
            prediction_error: Prediction error
            precision: Precision of prediction
            
        Returns:
            Updated belief
        """
        # Precision-weighted update
        update = self.learning_rate * precision * prediction_error
        new_belief = current_belief + update
        
        return new_belief


class ActiveInferenceController:
    """
    Active inference for action selection.
    
    Selects actions to minimize expected prediction error
    (free energy minimization).
    """
    
    def __init__(
        self,
        action_dim: int = 10,
        horizon: int = 5
    ):
        """
        Initialize active inference controller.
        
        Args:
            action_dim: Number of possible actions
            horizon: Planning horizon
        """
        self.action_dim = action_dim
        self.horizon = horizon
    
    def select_action(
        self,
        current_state: np.ndarray,
        predictor: InteroceptivePredictor,
        preferred_state: np.ndarray,
        context: torch.Tensor,
        emotion: torch.Tensor
    ) -> int:
        """
        Select action to minimize expected free energy.
        
        Args:
            current_state: Current bodily state
            predictor: Forward model
            preferred_state: Desired/homeostatic state
            context: Context representation
            emotion: Emotional state
            
        Returns:
            Selected action index
        """
        min_free_energy = float('inf')
        best_action = 0
        
        with torch.no_grad():
            for action in range(self.action_dim):
                # Simulate action effect (simplified)
                action_effect = self._simulate_action(action, current_state)
                
                # Predict resulting state
                predicted_state, precision = predictor(context, emotion)
                predicted_state = predicted_state.cpu().numpy()[0]
                
                # Compute expected free energy
                # F = prediction error + KL divergence from preferred state
                prediction_error = np.linalg.norm(predicted_state - action_effect)
                preference_error = np.linalg.norm(action_effect - preferred_state)
                
                free_energy = prediction_error + 0.5 * preference_error
                
                if free_energy < min_free_energy:
                    min_free_energy = free_energy
                    best_action = action
        
        return best_action
    
    def _simulate_action(
        self,
        action: int,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        Simulate effect of action on bodily state.
        
        Simplified simulation for demonstration.
        """
        # Random action effects (in real implementation, use learned model)
        action_effect = current_state + 0.1 * np.random.randn(len(current_state))
        
        return action_effect


class PredictiveProcessor:
    """
    Complete predictive processing system.
    
    Integrates prediction, error computation, and active inference
    for emotion generation and regulation.
    """
    
    def __init__(
        self,
        context_dim: int = 128,
        emotion_dim: int = 64,
        body_state_dim: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize predictive processor.
        
        Args:
            context_dim: Context dimension
            emotion_dim: Emotion dimension
            body_state_dim: Body state dimension
            device: Computing device
        """
        self.context_dim = context_dim
        self.emotion_dim = emotion_dim
        self.body_state_dim = body_state_dim
        self.device = device
        
        # Components
        self.predictor = InteroceptivePredictor(
            context_dim, emotion_dim, body_state_dim
        ).to(device)
        
        self.error_computer = PredictionErrorComputer(learning_rate=0.01)
        self.active_inference = ActiveInferenceController()
        
        # State tracking
        self.current_belief = np.zeros(body_state_dim)
        self.prediction_history = []
        self.error_history = []
    
    def process(
        self,
        context: np.ndarray,
        emotion_state: np.ndarray,
        bodily_observation: np.ndarray
    ) -> PredictiveState:
        """
        Process one time step of predictive processing.
        
        Args:
            context: Current context
            emotion_state: Current emotional state
            bodily_observation: Observed bodily state
            
        Returns:
            PredictiveState object
        """
        # Convert to tensors
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        emotion_tensor = torch.FloatTensor(emotion_state).unsqueeze(0).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            prediction, precision = self.predictor(context_tensor, emotion_tensor)
            prediction_np = prediction.cpu().numpy()[0]
            precision_val = precision.cpu().item()
        
        # Compute prediction error
        prediction_error = self.error_computer.compute_error(
            prediction_np,
            bodily_observation,
            precision_val
        )
        
        # Update belief
        self.current_belief = self.error_computer.update_belief(
            self.current_belief,
            prediction_error,
            precision_val
        )
        
        # Compute confidence (inverse of error magnitude)
        error_magnitude = np.linalg.norm(prediction_error)
        confidence = 1.0 / (1.0 + error_magnitude)
        
        # Store history
        self.prediction_history.append(prediction_np)
        self.error_history.append(prediction_error)
        
        return PredictiveState(
            prediction=prediction_np,
            observation=bodily_observation,
            prediction_error=prediction_error,
            precision=precision_val,
            confidence=confidence
        )
    
    def infer_emotion_from_interoception(
        self,
        bodily_state: np.ndarray,
        context: np.ndarray
    ) -> np.ndarray:
        """
        Infer emotional state from bodily sensations (constructionist view).
        
        Args:
            bodily_state: Current bodily state
            context: Contextual information
            
        Returns:
            Inferred emotional state
        """
        # Use inverse model to infer emotion that would predict this body state
        # Simplified: use gradient-based optimization
        
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        target_body = torch.FloatTensor(bodily_state).unsqueeze(0).to(self.device)
        
        # Initialize emotion estimate
        emotion_estimate = torch.randn(1, self.emotion_dim, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([emotion_estimate], lr=0.1)
        
        # Optimize to find emotion that predicts observed body state
        for _ in range(50):
            optimizer.zero_grad()
            
            predicted_body, _ = self.predictor(context_tensor, emotion_estimate)
            loss = F.mse_loss(predicted_body, target_body)
            
            loss.backward()
            optimizer.step()
        
        return emotion_estimate.detach().cpu().numpy()[0]
    
    def train_predictor(
        self,
        contexts: np.ndarray,
        emotions: np.ndarray,
        body_states: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True
    ):
        """
        Train interoceptive predictor on data.
        
        Args:
            contexts: Context representations [n_samples, context_dim]
            emotions: Emotional states [n_samples, emotion_dim]
            body_states: Bodily states [n_samples, body_state_dim]
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress
        """
        # Convert to tensors
        X_context = torch.FloatTensor(contexts).to(self.device)
        X_emotion = torch.FloatTensor(emotions).to(self.device)
        y = torch.FloatTensor(body_states).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        
        n_samples = len(X_context)
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle
            perm = torch.randperm(n_samples)
            X_context_shuffled = X_context[perm]
            X_emotion_shuffled = X_emotion[perm]
            y_shuffled = y[perm]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                context_batch = X_context_shuffled[start_idx:end_idx]
                emotion_batch = X_emotion_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions, precisions = self.predictor(context_batch, emotion_batch)
                
                # Loss: precision-weighted MSE
                errors = y_batch - predictions
                loss = torch.mean(precisions * errors ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_prediction_error_magnitude(self) -> float:
        """Get average prediction error magnitude."""
        if not self.error_history:
            return 0.0
        
        errors = np.array([np.linalg.norm(e) for e in self.error_history])
        return np.mean(errors)
    
    def reset_state(self):
        """Reset internal state."""
        self.current_belief = np.zeros(self.body_state_dim)
        self.prediction_history = []
        self.error_history = []


# Import for loss function
import torch.nn.functional as F
