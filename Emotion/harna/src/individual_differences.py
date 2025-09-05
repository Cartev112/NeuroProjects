"""
Individual differences engine with personality trait modulation.

Implements how traits like neuroticism, alexithymia, and resilience
modulate emotional processing pathways.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass
from enum import Enum


class PersonalityTrait(Enum):
    """Personality traits that modulate emotion processing."""
    NEUROTICISM = "neuroticism"
    EXTRAVERSION = "extraversion"
    CONSCIENTIOUSNESS = "conscientiousness"
    AGREEABLENESS = "agreeableness"
    OPENNESS = "openness"
    ALEXITHYMIA = "alexithymia"
    RESILIENCE = "resilience"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"


@dataclass
class PersonalityTraits:
    """Container for personality trait values (0 to 1)."""
    neuroticism: float = 0.5
    extraversion: float = 0.5
    conscientiousness: float = 0.5
    agreeableness: float = 0.5
    openness: float = 0.5
    alexithymia: float = 0.2
    resilience: float = 0.7
    emotional_intelligence: float = 0.6
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'neuroticism': self.neuroticism,
            'extraversion': self.extraversion,
            'conscientiousness': self.conscientiousness,
            'agreeableness': self.agreeableness,
            'openness': self.openness,
            'alexithymia': self.alexithymia,
            'resilience': self.resilience,
            'emotional_intelligence': self.emotional_intelligence
        }


class IndividualDifferences:
    """
    Individual differences engine.
    
    Modulates emotional processing based on personality traits
    and emotional competencies.
    """
    
    def __init__(self, traits: PersonalityTraits = None):
        """
        Initialize individual differences engine.
        
        Args:
            traits: Personality trait values
        """
        self.traits = traits if traits is not None else PersonalityTraits()
    
    def modulate_threat_sensitivity(self, base_threat: float) -> float:
        """
        Modulate threat detection based on neuroticism.
        
        High neuroticism increases threat sensitivity.
        """
        modulation = 1.0 + 0.5 * self.traits.neuroticism
        return np.clip(base_threat * modulation, 0, 1)
    
    def modulate_arousal(self, base_arousal: float) -> float:
        """
        Modulate arousal based on extraversion and neuroticism.
        
        Extraverts seek higher arousal, neurotics have higher baseline.
        """
        # Neuroticism increases arousal
        neuroticism_effect = 0.3 * self.traits.neuroticism
        
        # Extraversion modulates optimal arousal level
        extraversion_effect = 0.2 * (self.traits.extraversion - 0.5)
        
        modulated = base_arousal + neuroticism_effect + extraversion_effect
        return np.clip(modulated, -1, 1)
    
    def modulate_valence(self, base_valence: float) -> float:
        """
        Modulate valence based on neuroticism and agreeableness.
        
        High neuroticism biases toward negative valence.
        """
        # Neuroticism creates negativity bias
        if base_valence < 0:
            modulation = 1.0 + 0.4 * self.traits.neuroticism
        else:
            modulation = 1.0 - 0.2 * self.traits.neuroticism
        
        # Agreeableness creates positivity bias
        agreeableness_effect = 0.1 * self.traits.agreeableness
        
        modulated = base_valence * modulation + agreeableness_effect
        return np.clip(modulated, -1, 1)
    
    def modulate_regulation_effectiveness(
        self,
        base_effectiveness: float,
        strategy: str
    ) -> float:
        """
        Modulate regulation effectiveness based on traits.
        
        Emotional intelligence and conscientiousness improve regulation.
        """
        # Emotional intelligence improves all strategies
        ei_boost = 0.3 * self.traits.emotional_intelligence
        
        # Conscientiousness improves effortful strategies (reappraisal)
        if strategy == 'reappraisal':
            conscientiousness_boost = 0.2 * self.traits.conscientiousness
        else:
            conscientiousness_boost = 0.0
        
        # Alexithymia impairs emotion-focused strategies
        alexithymia_penalty = -0.3 * self.traits.alexithymia
        
        modulated = base_effectiveness + ei_boost + conscientiousness_boost + alexithymia_penalty
        return np.clip(modulated, 0, 1)
    
    def modulate_coping_potential(self, base_coping: float) -> float:
        """
        Modulate coping potential based on resilience.
        
        Higher resilience increases perceived coping ability.
        """
        resilience_boost = 0.4 * self.traits.resilience
        modulated = base_coping + resilience_boost
        return np.clip(modulated, 0, 1)
    
    def modulate_appraisal_bias(self, appraisals: Dict[str, float]) -> Dict[str, float]:
        """
        Apply trait-based biases to appraisals.
        
        Args:
            appraisals: Original appraisal values
            
        Returns:
            Modulated appraisals
        """
        modulated = appraisals.copy()
        
        # Neuroticism biases threat appraisals
        if 'threat' in modulated:
            modulated['threat'] = self.modulate_threat_sensitivity(modulated['threat'])
        
        # Resilience biases coping potential
        if 'coping_potential' in modulated:
            modulated['coping_potential'] = self.modulate_coping_potential(
                modulated['coping_potential']
            )
        
        # Conscientiousness biases goal relevance
        if 'goal_relevance' in modulated:
            modulated['goal_relevance'] *= (1 + 0.2 * self.traits.conscientiousness)
            modulated['goal_relevance'] = np.clip(modulated['goal_relevance'], 0, 1)
        
        return modulated
    
    def get_pathway_weights(self) -> Dict[str, float]:
        """
        Get weights for bottom-up vs top-down pathways.
        
        Returns:
            Dictionary with pathway weights
        """
        # Neuroticism strengthens bottom-up (subcortical) pathway
        bottom_up_weight = 0.5 + 0.3 * self.traits.neuroticism
        
        # Conscientiousness and EI strengthen top-down (cortical) pathway
        top_down_weight = 0.5 + 0.2 * self.traits.conscientiousness + 0.2 * self.traits.emotional_intelligence
        
        # Normalize
        total = bottom_up_weight + top_down_weight
        bottom_up_weight /= total
        top_down_weight /= total
        
        return {
            'bottom_up': bottom_up_weight,
            'top_down': top_down_weight
        }
    
    def set_trait(self, trait: str, value: float):
        """Set a specific trait value."""
        if hasattr(self.traits, trait):
            setattr(self.traits, trait, np.clip(value, 0, 1))
        else:
            raise ValueError(f"Unknown trait: {trait}")
    
    def get_trait(self, trait: str) -> float:
        """Get a specific trait value."""
        if hasattr(self.traits, trait):
            return getattr(self.traits, trait)
        else:
            raise ValueError(f"Unknown trait: {trait}")
