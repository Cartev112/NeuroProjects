"""
Emotion regulation modules.

Implements different regulation strategies: reappraisal, suppression,
distraction, and situation selection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass


class RegulationStrategy(Enum):
    """Emotion regulation strategies."""
    REAPPRAISAL = "reappraisal"
    SUPPRESSION = "suppression"
    DISTRACTION = "distraction"
    SITUATION_SELECTION = "situation_selection"
    ACCEPTANCE = "acceptance"


@dataclass
class RegulationResult:
    """Result of emotion regulation."""
    original_valence: float
    regulated_valence: float
    original_arousal: float
    regulated_arousal: float
    strategy_used: str
    effectiveness: float
    effort_cost: float
    
    def __repr__(self) -> str:
        return (
            f"RegulationResult(\n"
            f"  Strategy: {self.strategy_used}\n"
            f"  Valence: {self.original_valence:+.2f} → {self.regulated_valence:+.2f}\n"
            f"  Arousal: {self.original_arousal:+.2f} → {self.regulated_arousal:+.2f}\n"
            f"  Effectiveness: {self.effectiveness:.2%}\n"
            f"  Effort Cost: {self.effort_cost:.2f}\n"
            f")"
        )


class ReappraisalModule(nn.Module):
    """
    Cognitive reappraisal module.
    
    Reinterprets emotional stimulus to change its meaning
    and emotional impact.
    """
    
    def __init__(self, appraisal_dim: int = 128):
        """Initialize reappraisal module."""
        super().__init__()
        
        # Reappraisal network transforms appraisals
        self.reappraisal_network = nn.Sequential(
            nn.Linear(appraisal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, appraisal_dim),
            nn.Tanh()  # Bounded reappraisal
        )
    
    def forward(self, appraisals: torch.Tensor, intensity: float = 1.0) -> torch.Tensor:
        """
        Apply reappraisal to modify appraisals.
        
        Args:
            appraisals: Original appraisals
            intensity: Reappraisal intensity (0 to 1)
            
        Returns:
            Modified appraisals
        """
        reappraised = self.reappraisal_network(appraisals)
        
        # Blend original and reappraised based on intensity
        modified = (1 - intensity) * appraisals + intensity * reappraised
        
        return modified


class SuppressionModule:
    """
    Response suppression module.
    
    Inhibits emotional expression and physiological responses.
    """
    
    def __init__(self, suppression_strength: float = 0.7):
        """
        Initialize suppression module.
        
        Args:
            suppression_strength: Strength of suppression (0 to 1)
        """
        self.suppression_strength = suppression_strength
    
    def suppress(
        self,
        emotional_response: Dict[str, float],
        target_components: list = ['arousal', 'expression']
    ) -> Dict[str, float]:
        """
        Suppress emotional response components.
        
        Args:
            emotional_response: Original emotional response
            target_components: Components to suppress
            
        Returns:
            Suppressed emotional response
        """
        suppressed = emotional_response.copy()
        
        for component in target_components:
            if component in suppressed:
                # Reduce component magnitude
                suppressed[component] *= (1 - self.suppression_strength)
        
        return suppressed


class DistractionModule:
    """
    Attentional distraction module.
    
    Redirects attention away from emotional stimulus.
    """
    
    def __init__(self, distraction_effectiveness: float = 0.6):
        """
        Initialize distraction module.
        
        Args:
            distraction_effectiveness: Effectiveness of distraction (0 to 1)
        """
        self.distraction_effectiveness = distraction_effectiveness
    
    def distract(
        self,
        emotional_intensity: float,
        alternative_focus: Optional[float] = None
    ) -> float:
        """
        Apply distraction to reduce emotional intensity.
        
        Args:
            emotional_intensity: Original intensity
            alternative_focus: Intensity of alternative focus
            
        Returns:
            Reduced emotional intensity
        """
        if alternative_focus is None:
            alternative_focus = 0.5
        
        # Distraction reduces focus on emotional stimulus
        reduced_intensity = emotional_intensity * (1 - self.distraction_effectiveness)
        
        return reduced_intensity


class RegulationModule:
    """
    Complete emotion regulation system.
    
    Integrates multiple regulation strategies and selects
    appropriate strategy based on context.
    """
    
    def __init__(
        self,
        appraisal_dim: int = 128,
        device: str = 'cpu'
    ):
        """
        Initialize regulation module.
        
        Args:
            appraisal_dim: Appraisal dimension
            device: Computing device
        """
        self.appraisal_dim = appraisal_dim
        self.device = device
        
        # Initialize strategy modules
        self.reappraisal = ReappraisalModule(appraisal_dim).to(device)
        self.suppression = SuppressionModule()
        self.distraction = DistractionModule()
        
        # Strategy effectiveness tracking
        self.strategy_history = []
    
    def regulate(
        self,
        emotional_response: Dict[str, float],
        strategy: RegulationStrategy,
        intensity: float = 1.0,
        appraisals: Optional[np.ndarray] = None
    ) -> RegulationResult:
        """
        Apply emotion regulation strategy.
        
        Args:
            emotional_response: Original emotional response
            strategy: Regulation strategy to use
            intensity: Regulation intensity (0 to 1)
            appraisals: Appraisal values (for reappraisal)
            
        Returns:
            RegulationResult object
        """
        original_valence = emotional_response.get('valence', 0.0)
        original_arousal = emotional_response.get('arousal', 0.0)
        
        if strategy == RegulationStrategy.REAPPRAISAL:
            result = self._apply_reappraisal(
                emotional_response, intensity, appraisals
            )
        elif strategy == RegulationStrategy.SUPPRESSION:
            result = self._apply_suppression(emotional_response, intensity)
        elif strategy == RegulationStrategy.DISTRACTION:
            result = self._apply_distraction(emotional_response, intensity)
        elif strategy == RegulationStrategy.ACCEPTANCE:
            result = self._apply_acceptance(emotional_response)
        else:
            result = emotional_response.copy()
        
        regulated_valence = result.get('valence', original_valence)
        regulated_arousal = result.get('arousal', original_arousal)
        
        # Compute effectiveness
        valence_change = abs(regulated_valence - original_valence)
        arousal_change = abs(regulated_arousal - original_arousal)
        effectiveness = (valence_change + arousal_change) / 2
        
        # Effort cost (different strategies have different costs)
        effort_cost = self._compute_effort_cost(strategy, intensity)
        
        regulation_result = RegulationResult(
            original_valence=original_valence,
            regulated_valence=regulated_valence,
            original_arousal=original_arousal,
            regulated_arousal=regulated_arousal,
            strategy_used=strategy.value,
            effectiveness=effectiveness,
            effort_cost=effort_cost
        )
        
        self.strategy_history.append(regulation_result)
        
        return regulation_result
    
    def _apply_reappraisal(
        self,
        emotional_response: Dict[str, float],
        intensity: float,
        appraisals: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Apply cognitive reappraisal."""
        if appraisals is None:
            # Can't reappraise without appraisals
            return emotional_response
        
        with torch.no_grad():
            appraisal_tensor = torch.FloatTensor(appraisals).unsqueeze(0).to(self.device)
            modified_appraisals = self.reappraisal(appraisal_tensor, intensity)
            modified_appraisals = modified_appraisals.cpu().numpy()[0]
        
        # Recompute emotional response from modified appraisals
        # Simplified: adjust valence based on appraisal changes
        appraisal_change = np.mean(modified_appraisals - appraisals)
        
        regulated = emotional_response.copy()
        regulated['valence'] = emotional_response['valence'] + 0.5 * appraisal_change
        regulated['valence'] = np.clip(regulated['valence'], -1, 1)
        
        # Reappraisal also reduces arousal
        regulated['arousal'] = emotional_response['arousal'] * (1 - 0.3 * intensity)
        
        return regulated
    
    def _apply_suppression(
        self,
        emotional_response: Dict[str, float],
        intensity: float
    ) -> Dict[str, float]:
        """Apply response suppression."""
        self.suppression.suppression_strength = intensity
        
        # Suppression reduces arousal and expression but not valence
        regulated = emotional_response.copy()
        regulated['arousal'] *= (1 - intensity * 0.7)
        
        # Suppression paradox: can slightly increase negative valence
        if regulated['valence'] < 0:
            regulated['valence'] *= (1 + intensity * 0.1)
        
        return regulated
    
    def _apply_distraction(
        self,
        emotional_response: Dict[str, float],
        intensity: float
    ) -> Dict[str, float]:
        """Apply attentional distraction."""
        self.distraction.distraction_effectiveness = intensity
        
        # Distraction reduces both valence magnitude and arousal
        regulated = emotional_response.copy()
        
        valence_magnitude = abs(regulated['valence'])
        reduced_magnitude = self.distraction.distract(valence_magnitude)
        
        # Preserve sign
        regulated['valence'] = np.sign(regulated['valence']) * reduced_magnitude
        regulated['arousal'] *= (1 - intensity * 0.5)
        
        return regulated
    
    def _apply_acceptance(
        self,
        emotional_response: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply acceptance (minimal regulation)."""
        # Acceptance reduces arousal slightly without changing valence
        regulated = emotional_response.copy()
        regulated['arousal'] *= 0.9
        
        return regulated
    
    def _compute_effort_cost(
        self,
        strategy: RegulationStrategy,
        intensity: float
    ) -> float:
        """
        Compute cognitive effort cost of regulation.
        
        Different strategies have different effort requirements.
        """
        base_costs = {
            RegulationStrategy.REAPPRAISAL: 0.8,  # High cognitive effort
            RegulationStrategy.SUPPRESSION: 0.6,  # Moderate effort
            RegulationStrategy.DISTRACTION: 0.4,  # Lower effort
            RegulationStrategy.ACCEPTANCE: 0.1,   # Minimal effort
            RegulationStrategy.SITUATION_SELECTION: 0.3
        }
        
        base_cost = base_costs.get(strategy, 0.5)
        
        # Cost scales with intensity
        total_cost = base_cost * intensity
        
        return total_cost
    
    def select_optimal_strategy(
        self,
        emotional_response: Dict[str, float],
        context: Optional[Dict] = None
    ) -> RegulationStrategy:
        """
        Select optimal regulation strategy based on context.
        
        Args:
            emotional_response: Current emotional response
            context: Contextual information
            
        Returns:
            Recommended regulation strategy
        """
        valence = emotional_response.get('valence', 0.0)
        arousal = emotional_response.get('arousal', 0.0)
        
        # Decision rules based on emotion science
        
        # High arousal negative emotions: use reappraisal or distraction
        if valence < -0.5 and arousal > 0.5:
            # If time available, use reappraisal (more effective long-term)
            if context and context.get('time_available', True):
                return RegulationStrategy.REAPPRAISAL
            else:
                return RegulationStrategy.DISTRACTION
        
        # Moderate negative emotions: reappraisal
        elif valence < -0.3:
            return RegulationStrategy.REAPPRAISAL
        
        # Social context: avoid suppression (can impair social interaction)
        elif context and context.get('social', False):
            return RegulationStrategy.REAPPRAISAL
        
        # Low intensity emotions: acceptance
        elif abs(valence) < 0.3 and abs(arousal) < 0.3:
            return RegulationStrategy.ACCEPTANCE
        
        # Default: reappraisal (most effective overall)
        else:
            return RegulationStrategy.REAPPRAISAL
    
    def get_regulation_history(self) -> list:
        """Get history of regulation attempts."""
        return self.strategy_history
    
    def get_strategy_effectiveness(self) -> Dict[str, float]:
        """Compute average effectiveness of each strategy."""
        if not self.strategy_history:
            return {}
        
        strategy_effectiveness = {}
        
        for strategy in RegulationStrategy:
            strategy_results = [
                r for r in self.strategy_history 
                if r.strategy_used == strategy.value
            ]
            
            if strategy_results:
                avg_effectiveness = np.mean([r.effectiveness for r in strategy_results])
                strategy_effectiveness[strategy.value] = avg_effectiveness
        
        return strategy_effectiveness
