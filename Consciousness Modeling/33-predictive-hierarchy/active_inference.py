"""
Active Inference: minimize prediction error through both perception and action.
"""
import numpy as np


class ActiveInferenceAgent:
    """Agent that minimizes free energy via perception and action."""
    
    def __init__(self, hierarchy, action_dim, action_lr=0.1):
        self.hierarchy = hierarchy
        self.action_dim = action_dim
        self.action_lr = action_lr
        
        # Action policy (maps top-level hidden state to action)
        top_hidden_dim = hierarchy.levels[-1].hidden_dim
        self.W_action = np.random.randn(top_hidden_dim, action_dim) * 0.1
        self.b_action = np.zeros(action_dim)
        
        # Current action
        self.action = np.zeros(action_dim)
    
    def select_action(self):
        """Select action based on top-level hidden state."""
        top_hidden = self.hierarchy.levels[-1].hidden
        logits = top_hidden @ self.W_action + self.b_action
        self.action = np.tanh(logits)
        return self.action
    
    def update_action_policy(self, reward):
        """Update action policy based on reward (simplified RL)."""
        top_hidden = self.hierarchy.levels[-1].hidden
        
        # Gradient: increase actions that led to reward
        grad_W = np.outer(top_hidden, reward * self.action)
        grad_b = reward * self.action
        
        self.W_action += self.action_lr * grad_W
        self.b_action += self.action_lr * grad_b
    
    def step(self, observation, n_iterations=10):
        """One step of active inference: perceive and act.
        
        observation: current sensory input
        
        Returns: action and hierarchy outputs
        """
        # Perception: update beliefs
        outputs = self.hierarchy.forward(observation, n_iterations=n_iterations)
        
        # Action: select action to minimize expected free energy
        action = self.select_action()
        
        return action, outputs
    
    def compute_expected_free_energy(self, observation, action):
        """Compute expected free energy for a given action (simplified).
        
        EFE = expected prediction error + expected information gain
        """
        # Simulate applying action (simplified: action modulates observation)
        obs_with_action = observation + 0.1 * action[:len(observation)]
        
        # Forward pass
        outputs = self.hierarchy.forward(obs_with_action, n_iterations=5)
        
        # Expected error
        exp_error = sum(np.sum(err ** 2) for err in outputs['errors'])
        
        # Expected information gain (entropy reduction, simplified)
        exp_info_gain = -sum(np.sum(np.log(prec + 1e-8)) for prec in outputs['precisions'])
        
        efe = exp_error - exp_info_gain
        return efe
