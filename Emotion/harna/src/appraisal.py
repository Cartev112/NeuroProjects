"""
Top-down prefrontal appraisal networks.

Implements Scherer's Component Process Model with multi-dimensional
cognitive appraisal of emotional stimuli.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AppraisalDimension(Enum):
    """Scherer's appraisal dimensions."""
    NOVELTY = "novelty"
    INTRINSIC_PLEASANTNESS = "intrinsic_pleasantness"
    GOAL_RELEVANCE = "goal_relevance"
    COPING_POTENTIAL = "coping_potential"
    NORM_COMPATIBILITY = "norm_compatibility"


@dataclass
class AppraisalResult:
    """Result of appraisal process."""
    novelty: float  # How unexpected/familiar (-1 to 1)
    intrinsic_pleasantness: float  # Inherent pleasantness (-1 to 1)
    goal_relevance: float  # Relevance to goals (0 to 1)
    goal_conduciveness: float  # Helps or hinders goals (-1 to 1)
    coping_potential: float  # Ability to cope (0 to 1)
    norm_compatibility: float  # Aligns with norms (-1 to 1)
    
    # Derived emotional dimensions
    valence: float
    arousal: float
    dominance: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'novelty': self.novelty,
            'intrinsic_pleasantness': self.intrinsic_pleasantness,
            'goal_relevance': self.goal_relevance,
            'goal_conduciveness': self.goal_conduciveness,
            'coping_potential': self.coping_potential,
            'norm_compatibility': self.norm_compatibility,
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance
        }
    
    def __repr__(self) -> str:
        return (
            f"AppraisalResult(\n"
            f"  Novelty: {self.novelty:+.2f}\n"
            f"  Pleasantness: {self.intrinsic_pleasantness:+.2f}\n"
            f"  Goal Relevance: {self.goal_relevance:.2f}\n"
            f"  Goal Conduciveness: {self.goal_conduciveness:+.2f}\n"
            f"  Coping Potential: {self.coping_potential:.2f}\n"
            f"  Norm Compatibility: {self.norm_compatibility:+.2f}\n"
            f"  → Valence: {self.valence:+.2f}, Arousal: {self.arousal:+.2f}, Dominance: {self.dominance:+.2f}\n"
            f")"
        )


class NoveltyDetector(nn.Module):
    """
    Detects novelty/familiarity of stimulus.
    
    Compares current stimulus to memory representations.
    """
    
    def __init__(self, input_dim: int = 512, memory_size: int = 1000):
        """Initialize novelty detector."""
        super().__init__()
        
        self.input_dim = input_dim
        self.memory_size = memory_size
        
        # Memory buffer (simplified)
        self.register_buffer('memory', torch.zeros(memory_size, input_dim))
        self.memory_ptr = 0
        
        # Comparison network
        self.comparator = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 (very novel) to 1 (very familiar)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute novelty."""
        batch_size = x.size(0)
        
        # Compare to memory
        if self.memory_ptr > 0:
            # Get most similar memory
            similarities = torch.mm(x, self.memory[:self.memory_ptr].t())
            max_sim, max_idx = similarities.max(dim=1)
            
            # Get most similar memory item
            similar_memory = self.memory[max_idx]
            
            # Concatenate and compare
            combined = torch.cat([x, similar_memory], dim=1)
            novelty = self.comparator(combined)
        else:
            # No memory yet, everything is novel
            novelty = -torch.ones(batch_size, 1)
        
        return novelty
    
    def update_memory(self, x: torch.Tensor):
        """Add stimulus to memory."""
        if self.memory_ptr < self.memory_size:
            self.memory[self.memory_ptr] = x.detach()
            self.memory_ptr += 1
        else:
            # Replace random memory
            idx = np.random.randint(0, self.memory_size)
            self.memory[idx] = x.detach()


class GoalRelevanceEvaluator(nn.Module):
    """
    Evaluates relevance of stimulus to current goals.
    
    Takes stimulus features and goal representation as input.
    """
    
    def __init__(self, input_dim: int = 512, goal_dim: int = 64):
        """Initialize goal relevance evaluator."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # [relevance, conduciveness]
            nn.Tanh()
        )
    
    def forward(
        self,
        stimulus: torch.Tensor,
        goals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate goal relevance and conduciveness.
        
        Args:
            stimulus: Stimulus features
            goals: Goal representation
            
        Returns:
            Tuple of (relevance, conduciveness)
        """
        combined = torch.cat([stimulus, goals], dim=1)
        output = self.network(combined)
        
        relevance = (output[:, 0:1] + 1) / 2  # Scale to [0, 1]
        conduciveness = output[:, 1:2]  # Keep in [-1, 1]
        
        return relevance, conduciveness


class CopingPotentialEvaluator(nn.Module):
    """
    Evaluates ability to cope with situation.
    
    Considers stimulus demands and available resources.
    """
    
    def __init__(self, input_dim: int = 512):
        """Initialize coping evaluator."""
        super().__init__()
        
        # Demand assessment
        self.demand_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Demand level [0, 1]
        )
        
        # Resource assessment (context-dependent)
        self.resource_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Resource level [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute coping potential.
        
        Coping potential = resources / demands
        """
        demands = self.demand_network(x)
        resources = self.resource_network(x)
        
        # Avoid division by zero
        coping = resources / (demands + 1e-6)
        coping = torch.clamp(coping, 0, 1)
        
        return coping


class NormCompatibilityEvaluator(nn.Module):
    """
    Evaluates compatibility with social/moral norms.
    
    Considers cultural and personal values.
    """
    
    def __init__(self, input_dim: int = 512, norm_dim: int = 32):
        """Initialize norm evaluator."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + norm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Compatibility [-1, 1]
        )
    
    def forward(
        self,
        stimulus: torch.Tensor,
        norms: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate norm compatibility."""
        combined = torch.cat([stimulus, norms], dim=1)
        return self.network(combined)


class SchererAppraisal:
    """
    Complete Scherer Component Process Model implementation.
    
    Performs multi-dimensional appraisal and derives emotional
    dimensions (valence, arousal, dominance).
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        goal_dim: int = 64,
        norm_dim: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize Scherer appraisal system.
        
        Args:
            input_dim: Stimulus feature dimension
            goal_dim: Goal representation dimension
            norm_dim: Norm representation dimension
            device: Computing device
        """
        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.norm_dim = norm_dim
        self.device = device
        
        # Initialize appraisal components
        self.novelty_detector = NoveltyDetector(input_dim).to(device)
        self.goal_evaluator = GoalRelevanceEvaluator(input_dim, goal_dim).to(device)
        self.coping_evaluator = CopingPotentialEvaluator(input_dim).to(device)
        self.norm_evaluator = NormCompatibilityEvaluator(input_dim, norm_dim).to(device)
        
        # Pleasantness detector (simple network)
        self.pleasantness_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        ).to(device)
        
        # Default goals and norms
        self.default_goals = torch.randn(1, goal_dim).to(device)
        self.default_norms = torch.randn(1, norm_dim).to(device)
    
    def appraise(
        self,
        features: np.ndarray,
        goals: Optional[np.ndarray] = None,
        norms: Optional[np.ndarray] = None,
        context: Optional[Dict] = None
    ) -> AppraisalResult:
        """
        Perform complete appraisal of stimulus.
        
        Args:
            features: Stimulus features
            goals: Current goals (uses default if None)
            norms: Personal/cultural norms (uses default if None)
            context: Additional context
            
        Returns:
            AppraisalResult object
        """
        with torch.no_grad():
            # Convert to tensors
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            if goals is not None:
                goal_tensor = torch.FloatTensor(goals).unsqueeze(0).to(self.device)
            else:
                goal_tensor = self.default_goals
            
            if norms is not None:
                norm_tensor = torch.FloatTensor(norms).unsqueeze(0).to(self.device)
            else:
                norm_tensor = self.default_norms
            
            # Perform appraisals
            novelty = self.novelty_detector(x).item()
            pleasantness = self.pleasantness_detector(x).item()
            goal_relevance, goal_conduciveness = self.goal_evaluator(x, goal_tensor)
            goal_relevance = goal_relevance.item()
            goal_conduciveness = goal_conduciveness.item()
            coping_potential = self.coping_evaluator(x).item()
            norm_compatibility = self.norm_evaluator(x, norm_tensor).item()
            
            # Update memory
            self.novelty_detector.update_memory(x[0])
            
            # Derive emotional dimensions from appraisals
            valence = self._compute_valence(
                pleasantness, goal_conduciveness, norm_compatibility, coping_potential
            )
            
            arousal = self._compute_arousal(
                novelty, goal_relevance, coping_potential
            )
            
            dominance = self._compute_dominance(
                coping_potential, goal_conduciveness
            )
            
            return AppraisalResult(
                novelty=novelty,
                intrinsic_pleasantness=pleasantness,
                goal_relevance=goal_relevance,
                goal_conduciveness=goal_conduciveness,
                coping_potential=coping_potential,
                norm_compatibility=norm_compatibility,
                valence=valence,
                arousal=arousal,
                dominance=dominance
            )
    
    def _compute_valence(
        self,
        pleasantness: float,
        goal_conduciveness: float,
        norm_compatibility: float,
        coping_potential: float
    ) -> float:
        """
        Compute valence from appraisals.
        
        Valence is primarily determined by:
        - Intrinsic pleasantness
        - Goal conduciveness
        - Norm compatibility
        - Coping potential
        """
        valence = (
            0.3 * pleasantness +
            0.4 * goal_conduciveness +
            0.2 * norm_compatibility +
            0.1 * (2 * coping_potential - 1)  # Scale to [-1, 1]
        )
        
        return np.clip(valence, -1, 1)
    
    def _compute_arousal(
        self,
        novelty: float,
        goal_relevance: float,
        coping_potential: float
    ) -> float:
        """
        Compute arousal from appraisals.
        
        Arousal increases with:
        - Novelty
        - Goal relevance
        - Low coping potential (stress)
        """
        # Novelty contribution (novel = high arousal)
        novelty_arousal = -novelty  # Novel is -1, so negate
        
        # Goal relevance contribution
        relevance_arousal = goal_relevance
        
        # Coping stress contribution
        stress_arousal = 1 - coping_potential
        
        arousal = (
            0.3 * novelty_arousal +
            0.4 * relevance_arousal +
            0.3 * stress_arousal
        )
        
        # Scale to [-1, 1]
        arousal = 2 * arousal - 1
        
        return np.clip(arousal, -1, 1)
    
    def _compute_dominance(
        self,
        coping_potential: float,
        goal_conduciveness: float
    ) -> float:
        """
        Compute dominance/control from appraisals.
        
        Dominance reflects sense of control and power.
        """
        dominance = (
            0.7 * (2 * coping_potential - 1) +  # Scale to [-1, 1]
            0.3 * goal_conduciveness
        )
        
        return np.clip(dominance, -1, 1)
    
    def set_goals(self, goals: np.ndarray):
        """Set current goals."""
        self.default_goals = torch.FloatTensor(goals).unsqueeze(0).to(self.device)
    
    def set_norms(self, norms: np.ndarray):
        """Set personal/cultural norms."""
        self.default_norms = torch.FloatTensor(norms).unsqueeze(0).to(self.device)


class AppraisalNetwork:
    """
    General appraisal network wrapper.
    
    Provides interface for different appraisal theories.
    """
    
    def __init__(
        self,
        theory: str = 'scherer',
        input_dim: int = 512,
        device: str = 'cpu'
    ):
        """
        Initialize appraisal network.
        
        Args:
            theory: Appraisal theory ('scherer', 'lazarus', 'smith-ellsworth')
            input_dim: Input dimension
            device: Computing device
        """
        self.theory = theory
        self.input_dim = input_dim
        self.device = device
        
        if theory == 'scherer':
            self.appraisal_system = SchererAppraisal(input_dim, device=device)
        else:
            raise ValueError(f"Unknown appraisal theory: {theory}")
    
    def appraise(
        self,
        features: np.ndarray,
        **kwargs
    ) -> AppraisalResult:
        """Perform appraisal."""
        return self.appraisal_system.appraise(features, **kwargs)
    
    def set_goals(self, goals: np.ndarray):
        """Set goals."""
        self.appraisal_system.set_goals(goals)
    
    def set_norms(self, norms: np.ndarray):
        """Set norms."""
        self.appraisal_system.set_norms(norms)
