"""
Global Workspace: information becomes globally broadcast when prediction errors exceed thresholds.
Implements access consciousness as threshold-gated broadcasting.
"""
import numpy as np


class GlobalWorkspace:
    """Global workspace for conscious access via broadcasting."""
    
    def __init__(self, hierarchy, broadcast_threshold=1.5, decay_rate=0.1):
        self.hierarchy = hierarchy
        self.broadcast_threshold = broadcast_threshold
        self.decay_rate = decay_rate
        
        # Workspace state (globally accessible information)
        total_dim = sum(level.hidden_dim for level in hierarchy.levels)
        self.workspace = np.zeros(total_dim)
        
        # Broadcasting history
        self.broadcast_history = []
        
        # Access consciousness flag
        self.is_conscious = False
    
    def update(self):
        """Update workspace based on prediction errors.
        
        Information enters workspace when prediction errors exceed threshold.
        """
        # Collect all hidden states and errors
        all_hidden = []
        all_errors = []
        
        for level in self.hierarchy.levels:
            all_hidden.append(level.hidden)
            all_errors.append(level.error)
        
        all_hidden = np.concatenate(all_hidden)
        all_errors = np.concatenate(all_errors)
        
        # Gating: only broadcast if errors exceed threshold
        error_magnitude = np.abs(all_errors)
        gate = (error_magnitude > self.broadcast_threshold).astype(float)
        
        # Broadcast gated information
        broadcast_signal = gate * all_hidden
        
        # Update workspace with decay
        self.workspace = (1 - self.decay_rate) * self.workspace + self.decay_rate * broadcast_signal
        
        # Determine if conscious (any information broadcasted)
        self.is_conscious = np.sum(gate) > 0
        
        # Record
        self.broadcast_history.append({
            'workspace': self.workspace.copy(),
            'gate': gate.copy(),
            'is_conscious': self.is_conscious,
        })
        
        return self.workspace
    
    def get_conscious_content(self):
        """Return currently conscious (broadcasted) content."""
        return self.workspace
    
    def get_broadcast_strength(self):
        """Measure strength of current broadcast."""
        return float(np.linalg.norm(self.workspace))
    
    def reset(self):
        """Clear workspace."""
        self.workspace = np.zeros_like(self.workspace)
        self.broadcast_history = []
        self.is_conscious = False


class AttentionMechanism:
    """Attention as precision-weighting of prediction errors."""
    
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        
        # Attention weights per level
        self.attention_weights = [np.ones(level.output_dim) for level in hierarchy.levels]
    
    def focus_attention(self, level_idx, feature_indices, strength=2.0):
        """Focus attention on specific features at a level.
        
        level_idx: which level to attend to
        feature_indices: which features to enhance
        strength: multiplicative boost to precision
        """
        # Increase precision for attended features
        for idx in feature_indices:
            if idx < len(self.hierarchy.levels[level_idx].precision):
                self.hierarchy.levels[level_idx].precision[idx] *= strength
    
    def diffuse_attention(self):
        """Spread attention uniformly across all levels."""
        for level in self.hierarchy.levels:
            level.precision = np.ones_like(level.precision)
    
    def compute_attention_map(self):
        """Compute attention map based on current precisions."""
        attention_map = []
        for level in self.hierarchy.levels:
            # Normalize precision to [0, 1]
            norm_prec = level.precision / (np.max(level.precision) + 1e-8)
            attention_map.append(norm_prec)
        return attention_map
