"""
Theory of Mind module for recursive mentalizing.

Implements nested belief representations and perspective taking.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from .agent import Agent, EmotionalState


@dataclass
class MentalState:
    """Mental state representation."""
    emotion: EmotionalState
    belief_about: Optional[Dict[str, 'MentalState']] = None
    confidence: float = 0.8
    depth: int = 0  # Recursion depth
    
    def __repr__(self) -> str:
        return f"MentalState(emotion={self.emotion.emotion_type.value}, depth={self.depth})"


class TheoryOfMindModule:
    """
    Theory of Mind reasoning module.
    
    Implements recursive mentalizing: "I think she thinks I feel..."
    """
    
    def __init__(self, max_recursion: int = 3):
        """
        Initialize ToM module.
        
        Args:
            max_recursion: Maximum recursion depth
        """
        self.max_recursion = max_recursion
    
    def infer_other_emotion(
        self,
        observer: Agent,
        target: Agent,
        observed_behavior: Optional[Dict] = None
    ) -> EmotionalState:
        """
        First-order ToM: Infer target's emotional state.
        
        Args:
            observer: Observing agent
            target: Target agent
            observed_behavior: Observable cues
            
        Returns:
            Inferred emotional state
        """
        # Use observer's perception
        perceived_emotion = observer.perceive_other_emotion(target)
        
        # Adjust based on observed behavior if provided
        if observed_behavior:
            # Simple adjustment based on cues
            if 'facial_expression' in observed_behavior:
                expression = observed_behavior['facial_expression']
                if expression == 'smile':
                    perceived_emotion.valence += 0.2
                elif expression == 'frown':
                    perceived_emotion.valence -= 0.2
            
            if 'voice_tone' in observed_behavior:
                tone = observed_behavior['voice_tone']
                if tone == 'tense':
                    perceived_emotion.arousal += 0.2
                elif tone == 'calm':
                    perceived_emotion.arousal -= 0.2
            
            # Clip
            perceived_emotion.valence = np.clip(perceived_emotion.valence, -1, 1)
            perceived_emotion.arousal = np.clip(perceived_emotion.arousal, -1, 1)
        
        return perceived_emotion
    
    def recursive_mentalizing(
        self,
        agent: Agent,
        target: Agent,
        depth: int = 2
    ) -> MentalState:
        """
        Recursive mentalizing to specified depth.
        
        Args:
            agent: Agent doing the reasoning
            target: Target agent
            depth: Recursion depth
            
        Returns:
            Nested mental state
        """
        if depth > self.max_recursion:
            depth = self.max_recursion
        
        return self._recursive_infer(agent, target, depth, 0)
    
    def _recursive_infer(
        self,
        agent: Agent,
        target: Agent,
        max_depth: int,
        current_depth: int
    ) -> MentalState:
        """Recursive inference helper."""
        # Base case: infer target's emotion
        inferred_emotion = self.infer_other_emotion(agent, target)
        
        mental_state = MentalState(
            emotion=inferred_emotion,
            depth=current_depth
        )
        
        # Recursive case: infer what target thinks about agent
        if current_depth < max_depth:
            # What does target think about agent?
            nested_belief = self._recursive_infer(
                target, agent,
                max_depth, current_depth + 1
            )
            
            mental_state.belief_about = {agent.agent_id: nested_belief}
        
        return mental_state
    
    def false_belief_reasoning(
        self,
        agent: Agent,
        target: Agent,
        true_state: Dict,
        target_belief: Dict
    ) -> bool:
        """
        Test false belief understanding.
        
        Args:
            agent: Agent doing reasoning
            target: Target agent
            true_state: True state of the world
            target_belief: What target believes
            
        Returns:
            Whether agent understands target has false belief
        """
        # Agent must recognize that target's belief differs from reality
        belief_difference = abs(
            true_state.get('valence', 0) - target_belief.get('valence', 0)
        )
        
        # If difference is significant, agent should recognize false belief
        return belief_difference > 0.3
    
    def perspective_taking(
        self,
        agent: Agent,
        target: Agent,
        situation: Dict
    ) -> EmotionalState:
        """
        Take target's perspective in situation.
        
        Args:
            agent: Agent taking perspective
            target: Target agent
            situation: Situational context
            
        Returns:
            Emotion from target's perspective
        """
        # Simulate how target would feel in situation
        # Adjust based on target's personality
        
        base_valence = situation.get('valence', 0.0)
        base_arousal = situation.get('arousal', 0.0)
        
        # Modulate by target's neuroticism
        if target.personality['neuroticism'] > 0.6:
            # More negative and aroused
            base_valence -= 0.2
            base_arousal += 0.2
        
        # Modulate by target's extraversion
        if target.personality['extraversion'] > 0.6:
            # More positive in social situations
            if situation.get('social', False):
                base_valence += 0.2
        
        perspective_emotion = EmotionalState(
            valence=np.clip(base_valence, -1, 1),
            arousal=np.clip(base_arousal, -1, 1)
        )
        
        return perspective_emotion
