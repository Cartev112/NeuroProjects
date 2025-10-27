"""
Main HARNA model integrating all components.

Hierarchical Appraisal-Regulation Neural Architecture for emotion processing.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from .subcortical import SubcorticalPathway
from .appraisal import AppraisalNetwork
from .predictive_processing import PredictiveProcessor
from .regulation import RegulationModule, RegulationStrategy
from .individual_differences import IndividualDifferences, PersonalityTraits
from .timescales import MultiTimescaleProcessor
from .utils import emotion_label_from_valence_arousal


@dataclass
class Stimulus:
    """Stimulus representation."""
    features: np.ndarray
    context: Optional[Dict] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class EmotionalResponse:
    """Complete emotional response from HARNA."""
    emotion_label: str
    valence: float
    arousal: float
    dominance: float
    
    # Component responses
    subcortical_threat: float
    subcortical_arousal: float
    appraisal_result: Dict
    prediction_error: float
    
    # Processing info
    processing_time_ms: float
    pathway_weights: Dict[str, float]
    
    def __repr__(self) -> str:
        return (
            f"EmotionalResponse(\n"
            f"  Emotion: {self.emotion_label}\n"
            f"  Valence: {self.valence:+.2f}\n"
            f"  Arousal: {self.arousal:+.2f}\n"
            f"  Dominance: {self.dominance:+.2f}\n"
            f"  Threat: {self.subcortical_threat:.2f}\n"
            f"  Processing Time: {self.processing_time_ms:.1f}ms\n"
            f")"
        )


class HARNAModel:
    """
    Complete HARNA model.
    
    Integrates bottom-up subcortical processing, top-down appraisal,
    predictive processing, emotion regulation, and individual differences.
    """
    
    def __init__(
        self,
        theory: str = 'scherer',
        individual_traits: Optional[Dict[str, float]] = None,
        input_dim: int = 512,
        device: str = 'cpu'
    ):
        """
        Initialize HARNA model.
        
        Args:
            theory: Emotion theory ('scherer', 'james_lange', etc.)
            individual_traits: Personality trait values
            input_dim: Input feature dimension
            device: Computing device
        """
        self.theory = theory
        self.input_dim = input_dim
        self.device = device
        
        # Initialize components
        self.subcortical = SubcorticalPathway(input_dim, device)
        self.appraisal = AppraisalNetwork(theory, input_dim, device)
        self.predictive = PredictiveProcessor(
            context_dim=128,
            emotion_dim=64,
            body_state_dim=32,
            device=device
        )
        self.regulation = RegulationModule(appraisal_dim=128, device=device)
        
        # Individual differences
        if individual_traits is not None:
            traits = PersonalityTraits(**individual_traits)
        else:
            traits = PersonalityTraits()
        self.individual_diff = IndividualDifferences(traits)
        
        # Multi-timescale processor
        self.timescale_processor = MultiTimescaleProcessor()
        
        # State tracking
        self.current_time = 0.0
    
    def process(
        self,
        stimulus: Stimulus,
        regulate: bool = False,
        regulation_strategy: Optional[RegulationStrategy] = None
    ) -> EmotionalResponse:
        """
        Process stimulus through complete HARNA architecture.
        
        Args:
            stimulus: Input stimulus
            regulate: Whether to apply emotion regulation
            regulation_strategy: Specific regulation strategy
            
        Returns:
            EmotionalResponse object
        """
        import time
        start_time = time.time()
        
        # 1. Bottom-up subcortical processing (fast pathway)
        subcortical_response = self.subcortical.process(
            stimulus.features,
            stimulus.context
        )
        
        # Apply individual differences to subcortical response
        threat = self.individual_diff.modulate_threat_sensitivity(
            subcortical_response.threat_level
        )
        arousal = self.individual_diff.modulate_arousal(
            subcortical_response.arousal
        )
        
        # Record millisecond-scale event
        self.timescale_processor.process_millisecond(self.current_time, threat)
        
        # 2. Top-down appraisal (slower pathway)
        appraisal_result = self.appraisal.appraise(
            stimulus.features,
            context=stimulus.context
        )
        
        # Apply individual differences to appraisals
        appraisal_dict = appraisal_result.to_dict()
        appraisal_dict = self.individual_diff.modulate_appraisal_bias(appraisal_dict)
        
        # Record second-scale event
        self.timescale_processor.process_second(self.current_time, appraisal_dict)
        
        # 3. Integrate pathways
        pathway_weights = self.individual_diff.get_pathway_weights()
        
        # Combine valence from both pathways
        subcortical_valence = -threat  # Threat implies negative valence
        appraisal_valence = appraisal_dict['valence']
        
        integrated_valence = (
            pathway_weights['bottom_up'] * subcortical_valence +
            pathway_weights['top_down'] * appraisal_valence
        )
        
        # Combine arousal
        integrated_arousal = (
            pathway_weights['bottom_up'] * arousal +
            pathway_weights['top_down'] * appraisal_dict['arousal']
        )
        
        # Dominance from appraisal
        dominance = appraisal_dict['dominance']
        
        # Apply individual differences
        integrated_valence = self.individual_diff.modulate_valence(integrated_valence)
        integrated_arousal = self.individual_diff.modulate_arousal(integrated_arousal)
        
        # 4. Predictive processing (optional)
        # Simplified: just track prediction error
        prediction_error = 0.1  # Placeholder
        
        # 5. Emotion regulation (if requested)
        if regulate:
            emotional_state = {
                'valence': integrated_valence,
                'arousal': integrated_arousal,
                'dominance': dominance
            }
            
            if regulation_strategy is None:
                regulation_strategy = self.regulation.select_optimal_strategy(
                    emotional_state,
                    stimulus.context
                )
            
            regulation_result = self.regulation.regulate(
                emotional_state,
                regulation_strategy,
                intensity=0.7
            )
            
            integrated_valence = regulation_result.regulated_valence
            integrated_arousal = regulation_result.regulated_arousal
        
        # 6. Generate emotion label
        emotion_label = emotion_label_from_valence_arousal(
            integrated_valence,
            integrated_arousal
        )
        
        # Processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update time
        self.current_time += processing_time_ms / 1000
        
        return EmotionalResponse(
            emotion_label=emotion_label,
            valence=integrated_valence,
            arousal=integrated_arousal,
            dominance=dominance,
            subcortical_threat=threat,
            subcortical_arousal=arousal,
            appraisal_result=appraisal_dict,
            prediction_error=prediction_error,
            processing_time_ms=processing_time_ms,
            pathway_weights=pathway_weights
        )
    
    def process_simple(self, features: np.ndarray) -> Dict:
        """Simplified processing for RL integration."""
        stimulus = Stimulus(features=features)
        response = self.process(stimulus)
        
        return {
            'valence': response.valence,
            'arousal': response.arousal,
            'novelty': 0.0  # Placeholder
        }
    
    def set_personality_trait(self, trait: str, value: float):
        """Set a personality trait value."""
        self.individual_diff.set_trait(trait, value)
    
    def get_personality_trait(self, trait: str) -> float:
        """Get a personality trait value."""
        return self.individual_diff.get_trait(trait)
    
    def reset(self):
        """Reset model state."""
        self.current_time = 0.0
        self.subcortical.reset_processing_times()
        self.predictive.reset_state()
