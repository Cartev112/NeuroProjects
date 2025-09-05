"""
Hierarchical Predictive Coding Architecture with multiple levels.
Implements prediction, prediction error, and precision at each level.
"""
import numpy as np


class PredictiveLevel:
    """Single level in predictive hierarchy."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, timescale=1.0, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.timescale = timescale  # Temporal integration constant
        self.lr = lr
        
        # Prediction weights (top-down)
        self.W_pred = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b_pred = np.zeros(output_dim)
        
        # Error encoding weights (bottom-up)
        self.W_error = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_error = np.zeros(hidden_dim)
        
        # Precision (inverse variance)
        self.precision = np.ones(output_dim)
        
        # State
        self.hidden = np.zeros(hidden_dim)
        self.prediction = np.zeros(output_dim)
        self.error = np.zeros(output_dim)
    
    def predict(self, hidden_state=None):
        """Generate top-down prediction."""
        if hidden_state is None:
            hidden_state = self.hidden
        self.prediction = np.tanh(hidden_state @ self.W_pred + self.b_pred)
        return self.prediction
    
    def compute_error(self, observation, prediction=None):
        """Compute precision-weighted prediction error."""
        if prediction is None:
            prediction = self.prediction
        raw_error = observation - prediction
        self.error = self.precision * raw_error
        return self.error
    
    def update_hidden(self, error_from_below, prediction_from_above=None):
        """Update hidden state based on errors."""
        # Bottom-up error signal
        error_signal = error_from_below @ self.W_error + self.b_error
        
        # Top-down prediction (if available)
        if prediction_from_above is not None:
            pred_signal = prediction_from_above
        else:
            pred_signal = np.zeros_like(self.hidden)
        
        # Integrate with timescale
        delta = self.lr * (error_signal + pred_signal - self.hidden)
        self.hidden = self.hidden + delta / self.timescale
        
        return self.hidden
    
    def update_precision(self, error, adaptation_rate=0.01):
        """Adapt precision based on error magnitude (attention)."""
        # Precision increases where errors are small (reliable)
        # Precision decreases where errors are large (unreliable)
        error_var = error ** 2
        target_precision = 1.0 / (error_var + 0.1)
        self.precision = self.precision + adaptation_rate * (target_precision - self.precision)
        self.precision = np.clip(self.precision, 0.1, 10.0)
        return self.precision


class PredictiveHierarchy:
    """Multi-level predictive coding hierarchy."""
    
    def __init__(self, level_dims, timescales=None, lr=0.01):
        """
        level_dims: list of (input_dim, hidden_dim, output_dim) for each level
        timescales: list of temporal integration constants (slower at higher levels)
        """
        self.n_levels = len(level_dims)
        self.levels = []
        
        if timescales is None:
            # Exponentially increasing timescales
            timescales = [2.0 ** i for i in range(self.n_levels)]
        
        for i, (inp_dim, hid_dim, out_dim) in enumerate(level_dims):
            level = PredictiveLevel(inp_dim, hid_dim, out_dim, timescale=timescales[i], lr=lr)
            self.levels.append(level)
    
    def forward(self, observation, n_iterations=10):
        """Run predictive processing for n iterations.
        
        observation: sensory input at lowest level
        
        Returns: dict with predictions, errors, precisions at each level
        """
        # Initialize
        self.levels[0].prediction = observation
        
        for iteration in range(n_iterations):
            # Bottom-up pass: compute errors
            errors = []
            for i in range(self.n_levels):
                if i == 0:
                    # Sensory level
                    obs = observation
                else:
                    # Higher levels receive prediction from below
                    obs = self.levels[i - 1].hidden
                
                pred = self.levels[i].predict()
                error = self.levels[i].compute_error(obs, pred)
                errors.append(error)
            
            # Top-down pass: update hidden states
            for i in range(self.n_levels):
                error_below = errors[i] if i < len(errors) else np.zeros_like(self.levels[i].error)
                
                if i < self.n_levels - 1:
                    pred_above = self.levels[i + 1].prediction
                else:
                    pred_above = None
                
                self.levels[i].update_hidden(error_below, pred_above)
            
            # Update precisions
            for i, error in enumerate(errors):
                self.levels[i].update_precision(error)
        
        # Collect outputs
        return {
            'predictions': [level.prediction.copy() for level in self.levels],
            'errors': [level.error.copy() for level in self.levels],
            'precisions': [level.precision.copy() for level in self.levels],
            'hidden_states': [level.hidden.copy() for level in self.levels],
        }
    
    def get_total_error(self):
        """Compute total precision-weighted prediction error (free energy proxy)."""
        total = 0.0
        for level in self.levels:
            total += np.sum(level.precision * (level.error ** 2))
        return total
