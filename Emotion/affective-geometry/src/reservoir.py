"""
Reservoir computing / Echo State Network for emotional trajectory prediction.

Implements the "emotional weather system" predictor that forecasts
emotional trajectories hours in advance from real-time multimodal biosignals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class EmotionalPredictor:
    """
    Echo State Network for predicting emotional trajectories.
    
    Uses reservoir computing to forecast future emotional states from
    multimodal biosignal data (HRV, EDA, pupillometry, etc.).
    """
    
    def __init__(
        self,
        reservoir_size: int = 500,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        ridge_alpha: float = 1e-6,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Echo State Network.
        
        Args:
            reservoir_size: Number of reservoir neurons
            spectral_radius: Spectral radius of reservoir weight matrix
            input_scaling: Scaling factor for input weights
            leak_rate: Leak rate for leaky integrator neurons
            ridge_alpha: Regularization parameter for ridge regression
            random_seed: Random seed for reproducibility
        """
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Network weights (initialized in _initialize_weights)
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights
        self.W_out = None  # Output weights (trained)
        
        # Preprocessing
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # State
        self.reservoir_state = None
        self.is_trained = False
        
        # Dimensions (set during training)
        self.input_dim = None
        self.output_dim = None
    
    def _initialize_weights(self, input_dim: int, output_dim: int):
        """
        Initialize network weights.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input weights: random uniform [-input_scaling, input_scaling]
        self.W_in = np.random.uniform(
            -self.input_scaling,
            self.input_scaling,
            size=(self.reservoir_size, input_dim)
        )
        
        # Reservoir weights: sparse random matrix
        # Create sparse connectivity (10% connections)
        connectivity = 0.1
        W_res = np.random.randn(self.reservoir_size, self.reservoir_size)
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) > connectivity
        W_res[mask] = 0
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        current_spectral_radius = np.max(np.abs(eigenvalues))
        
        if current_spectral_radius > 0:
            self.W_res = W_res * (self.spectral_radius / current_spectral_radius)
        else:
            self.W_res = W_res
        
        # Initialize reservoir state
        self.reservoir_state = np.zeros(self.reservoir_size)
    
    def _update_reservoir(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Update reservoir state with new input.
        
        Args:
            input_vector: Input at current time step
            
        Returns:
            Updated reservoir state
        """
        # Leaky integrator neuron dynamics
        pre_activation = (
            np.dot(self.W_in, input_vector) +
            np.dot(self.W_res, self.reservoir_state)
        )
        
        # Apply activation function (tanh)
        activated = np.tanh(pre_activation)
        
        # Leaky integration
        self.reservoir_state = (
            (1 - self.leak_rate) * self.reservoir_state +
            self.leak_rate * activated
        )
        
        return self.reservoir_state
    
    def train(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        washout: int = 100,
        verbose: bool = True
    ):
        """
        Train the output weights using ridge regression.
        
        Args:
            input_data: Input time series of shape (n_timesteps, input_dim)
            target_data: Target outputs of shape (n_timesteps, output_dim)
            washout: Number of initial steps to discard
            verbose: Print training info
        """
        input_data = np.atleast_2d(input_data)
        target_data = np.atleast_2d(target_data)
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        if target_data.ndim == 1:
            target_data = target_data.reshape(-1, 1)
        
        n_timesteps = len(input_data)
        input_dim = input_data.shape[1]
        output_dim = target_data.shape[1]
        
        # Initialize weights if needed
        if self.W_in is None:
            self._initialize_weights(input_dim, output_dim)
        
        # Normalize data
        input_normalized = self.input_scaler.fit_transform(input_data)
        target_normalized = self.output_scaler.fit_transform(target_data)
        
        # Collect reservoir states
        reservoir_states = np.zeros((n_timesteps, self.reservoir_size))
        
        # Reset reservoir state
        self.reservoir_state = np.zeros(self.reservoir_size)
        
        # Run reservoir
        for t in range(n_timesteps):
            state = self._update_reservoir(input_normalized[t])
            reservoir_states[t] = state
        
        # Discard washout period
        reservoir_states_train = reservoir_states[washout:]
        target_train = target_normalized[washout:]
        
        # Train output weights using ridge regression
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        ridge.fit(reservoir_states_train, target_train)
        
        self.W_out = ridge.coef_
        self.W_bias = ridge.intercept_
        
        self.is_trained = True
        
        if verbose:
            # Compute training error
            predictions = ridge.predict(reservoir_states_train)
            mse = np.mean((predictions - target_train) ** 2)
            print(f"Training MSE: {mse:.6f}")
            print(f"Reservoir size: {self.reservoir_size}")
            print(f"Spectral radius: {self.spectral_radius}")
    
    def predict(
        self,
        initial_input: np.ndarray,
        horizon: int,
        autonomous: bool = False
    ) -> np.ndarray:
        """
        Predict future trajectory.
        
        Args:
            initial_input: Initial input state
            horizon: Number of steps to predict ahead
            autonomous: If True, feed predictions back as input
            
        Returns:
            Predicted trajectory of shape (horizon, output_dim)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        initial_input = np.atleast_1d(initial_input)
        if initial_input.ndim == 1:
            initial_input = initial_input.reshape(1, -1)
        
        # Normalize input
        current_input = self.input_scaler.transform(initial_input)[0]
        
        # Predictions
        predictions = np.zeros((horizon, self.output_dim))
        
        for t in range(horizon):
            # Update reservoir
            self._update_reservoir(current_input)
            
            # Compute output
            output = np.dot(self.W_out, self.reservoir_state) + self.W_bias
            
            # Denormalize
            output_denorm = self.output_scaler.inverse_transform(
                output.reshape(1, -1)
            )[0]
            
            predictions[t] = output_denorm
            
            # Update input for next step
            if autonomous:
                # Use prediction as next input
                current_input = self.input_scaler.transform(
                    output_denorm.reshape(1, -1)
                )[0]
            # Otherwise, would need to provide next input (not implemented here)
        
        return predictions
    
    def predict_with_input_sequence(
        self,
        input_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Predict outputs for a sequence of inputs.
        
        Args:
            input_sequence: Input sequence of shape (n_steps, input_dim)
            
        Returns:
            Predictions of shape (n_steps, output_dim)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        input_sequence = np.atleast_2d(input_sequence)
        if input_sequence.ndim == 1:
            input_sequence = input_sequence.reshape(-1, 1)
        
        n_steps = len(input_sequence)
        predictions = np.zeros((n_steps, self.output_dim))
        
        # Normalize inputs
        input_normalized = self.input_scaler.transform(input_sequence)
        
        for t in range(n_steps):
            # Update reservoir
            self._update_reservoir(input_normalized[t])
            
            # Compute output
            output = np.dot(self.W_out, self.reservoir_state) + self.W_bias
            
            # Denormalize
            predictions[t] = self.output_scaler.inverse_transform(
                output.reshape(1, -1)
            )[0]
        
        return predictions
    
    def reset_state(self):
        """Reset reservoir state to zero."""
        if self.reservoir_state is not None:
            self.reservoir_state = np.zeros(self.reservoir_size)
    
    def plot_forecast(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        dimension_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot predicted trajectory.
        
        Args:
            predictions: Predicted values
            ground_truth: True values (if available)
            dimension_names: Names for each dimension
            figsize: Figure size
        """
        n_dims = predictions.shape[1]
        
        if dimension_names is None:
            dimension_names = [f'Dimension {i}' for i in range(n_dims)]
        
        fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
        
        if n_dims == 1:
            axes = [axes]
        
        time_steps = np.arange(len(predictions))
        
        for i, ax in enumerate(axes):
            # Plot prediction
            ax.plot(time_steps, predictions[:, i], 'b-', 
                   linewidth=2, label='Prediction', alpha=0.8)
            
            # Plot ground truth if available
            if ground_truth is not None:
                ax.plot(time_steps, ground_truth[:, i], 'g--', 
                       linewidth=2, label='Ground Truth', alpha=0.8)
            
            ax.set_ylabel(dimension_names[i], fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            if i == 0:
                ax.set_title('Emotional Trajectory Forecast', 
                           fontsize=14, fontweight='bold')
        
        axes[-1].set_xlabel('Time Steps Ahead', fontsize=12)
        
        plt.tight_layout()
        return fig, axes
    
    def evaluate(
        self,
        test_input: np.ndarray,
        test_target: np.ndarray,
        washout: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_input: Test input sequence
            test_target: Test target sequence
            washout: Washout period
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Reset state
        self.reset_state()
        
        # Get predictions
        predictions = self.predict_with_input_sequence(test_input)
        
        # Discard washout
        predictions = predictions[washout:]
        test_target = test_target[washout:]
        
        # Compute metrics
        mse = np.mean((predictions - test_target) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - test_target))
        
        # R-squared
        ss_res = np.sum((test_target - predictions) ** 2)
        ss_tot = np.sum((test_target - np.mean(test_target, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class MultimodalEmotionalPredictor:
    """
    Extended predictor for multimodal biosignal data.
    
    Integrates multiple biosignal streams (HRV, EDA, pupillometry, EMG)
    for robust emotional prediction.
    """
    
    def __init__(
        self,
        modality_weights: Optional[Dict[str, float]] = None,
        **esn_kwargs
    ):
        """
        Initialize multimodal predictor.
        
        Args:
            modality_weights: Weights for each modality
            **esn_kwargs: Arguments for EmotionalPredictor
        """
        self.predictor = EmotionalPredictor(**esn_kwargs)
        self.modality_weights = modality_weights or {
            'hrv': 0.3,
            'eda': 0.3,
            'pupil': 0.2,
            'emg': 0.2
        }
    
    def prepare_multimodal_input(
        self,
        biosignals: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combine multiple biosignal modalities into single input.
        
        Args:
            biosignals: Dictionary mapping modality names to time series
            
        Returns:
            Combined input array
        """
        # Ensure all signals have same length
        lengths = [len(signal) for signal in biosignals.values()]
        min_length = min(lengths)
        
        # Concatenate and weight modalities
        combined = []
        for modality, signal in biosignals.items():
            weight = self.modality_weights.get(modality, 1.0)
            weighted_signal = signal[:min_length] * weight
            
            if weighted_signal.ndim == 1:
                weighted_signal = weighted_signal.reshape(-1, 1)
            
            combined.append(weighted_signal)
        
        return np.hstack(combined)
    
    def train(
        self,
        biosignals: Dict[str, np.ndarray],
        emotional_states: np.ndarray,
        **train_kwargs
    ):
        """
        Train on multimodal data.
        
        Args:
            biosignals: Dictionary of biosignal time series
            emotional_states: Target emotional states
            **train_kwargs: Additional training arguments
        """
        multimodal_input = self.prepare_multimodal_input(biosignals)
        self.predictor.train(multimodal_input, emotional_states, **train_kwargs)
    
    def predict(
        self,
        biosignals: Dict[str, np.ndarray],
        horizon: int,
        autonomous: bool = False
    ) -> np.ndarray:
        """
        Predict from multimodal biosignals.
        
        Args:
            biosignals: Current biosignal measurements
            horizon: Prediction horizon
            autonomous: Autonomous prediction mode
            
        Returns:
            Predicted emotional trajectory
        """
        multimodal_input = self.prepare_multimodal_input(biosignals)
        return self.predictor.predict(multimodal_input[0], horizon, autonomous)
    
    def plot_modality_importance(self, figsize: Tuple[int, int] = (10, 6)):
        """Plot importance weights for each modality."""
        fig, ax = plt.subplots(figsize=figsize)
        
        modalities = list(self.modality_weights.keys())
        weights = list(self.modality_weights.values())
        
        bars = ax.bar(modalities, weights, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title('Biosignal Modality Importance', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(weights) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig, ax
