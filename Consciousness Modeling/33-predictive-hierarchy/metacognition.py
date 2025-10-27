"""
Metacognitive module: self-modeling and confidence estimation.
"""
import numpy as np


class MetacognitiveModule:
    """Self-modeling module that monitors internal states."""
    
    def __init__(self, hierarchy, meta_dim=16):
        self.hierarchy = hierarchy
        self.meta_dim = meta_dim
        
        # Meta-level representation (monitors hierarchy state)
        n_levels = hierarchy.n_levels
        total_hidden = sum(level.hidden_dim for level in hierarchy.levels)
        
        self.W_meta = np.random.randn(total_hidden, meta_dim) * 0.1
        self.b_meta = np.zeros(meta_dim)
        
        # Confidence estimation
        self.W_conf = np.random.randn(meta_dim, 1) * 0.1
        self.b_conf = np.zeros(1)
        
        # Meta-state
        self.meta_state = np.zeros(meta_dim)
        self.confidence = 0.5
    
    def update(self):
        """Update meta-level representation from hierarchy states."""
        # Concatenate all hidden states
        all_hidden = np.concatenate([level.hidden for level in self.hierarchy.levels])
        
        # Meta-level encoding
        self.meta_state = np.tanh(all_hidden @ self.W_meta + self.b_meta)
        
        return self.meta_state
    
    def estimate_confidence(self):
        """Estimate confidence based on meta-state.
        
        Confidence correlates with:
        - Low prediction errors (high precision)
        - Stable hidden states
        - Consistent predictions
        """
        # Update meta-state
        self.update()
        
        # Confidence from meta-state
        conf_logit = self.meta_state @ self.W_conf + self.b_conf
        self.confidence = 1.0 / (1.0 + np.exp(-conf_logit[0]))
        
        # Also factor in precision (high precision -> high confidence)
        avg_precision = np.mean([np.mean(level.precision) for level in self.hierarchy.levels])
        self.confidence = 0.7 * self.confidence + 0.3 * (avg_precision / 10.0)
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
        
        return float(self.confidence)
    
    def predict_internal_state(self, steps_ahead=1):
        """Predict future internal state (self-modeling).
        
        Simplified: linear extrapolation of meta-state.
        """
        # Compute velocity
        # (In full implementation, would track history)
        velocity = 0.1 * self.meta_state  # Placeholder
        
        predicted_meta = self.meta_state + steps_ahead * velocity
        return predicted_meta
    
    def detect_surprise(self, threshold=2.0):
        """Detect surprising events based on prediction errors.
        
        Returns: bool indicating whether surprise exceeds threshold
        """
        # Total error across hierarchy
        total_error = self.hierarchy.get_total_error()
        
        # Normalize by number of levels
        avg_error = total_error / self.hierarchy.n_levels
        
        is_surprising = avg_error > threshold
        return is_surprising, float(avg_error)


class CounterfactualEngine:
    """Generate and evaluate counterfactual scenarios."""
    
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
    
    def simulate_counterfactual(self, observation, intervention, n_iterations=10):
        """Simulate 'what if' scenario with intervention.
        
        observation: current sensory input
        intervention: dict {level_idx: hidden_state_modification}
        
        Returns: counterfactual outputs
        """
        # Save current state
        saved_states = [level.hidden.copy() for level in self.hierarchy.levels]
        
        # Apply intervention
        for level_idx, modification in intervention.items():
            self.hierarchy.levels[level_idx].hidden = modification
        
        # Run forward pass
        outputs = self.hierarchy.forward(observation, n_iterations=n_iterations)
        
        # Restore original state
        for i, state in enumerate(saved_states):
            self.hierarchy.levels[i].hidden = state
        
        return outputs
    
    def compare_scenarios(self, observation, interventions_list):
        """Compare multiple counterfactual scenarios.
        
        interventions_list: list of intervention dicts
        
        Returns: list of (intervention, total_error) tuples
        """
        results = []
        
        for intervention in interventions_list:
            outputs = self.simulate_counterfactual(observation, intervention)
            total_error = sum(np.sum(err ** 2) for err in outputs['errors'])
            results.append((intervention, float(total_error)))
        
        # Sort by error (best scenarios first)
        results.sort(key=lambda x: x[1])
        
        return results
