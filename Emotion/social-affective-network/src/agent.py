"""
Agent class representing individuals in the social network.

Each agent has emotional states, personality traits, and social behaviors.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class EmotionType(Enum):
    """Basic emotion types."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class EmotionalState:
    """Emotional state representation."""
    valence: float  # -1 to 1
    arousal: float  # -1 to 1
    dominance: float = 0.0  # -1 to 1
    emotion_type: EmotionType = EmotionType.NEUTRAL
    intensity: float = 0.0  # 0 to 1
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'emotion_type': self.emotion_type.value,
            'intensity': self.intensity
        }
    
    def distance_to(self, other: 'EmotionalState') -> float:
        """Compute emotional distance to another state."""
        return np.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )


class Agent:
    """
    Individual agent in social network.
    
    Agents have emotional states, personality traits, and can
    interact with other agents through emotional contagion and empathy.
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_emotion: Optional[Dict] = None,
        personality: Optional[Dict] = None,
        cultural_background: str = 'western',
        attachment_style: str = 'secure'
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier
            initial_emotion: Initial emotional state
            personality: Personality trait values
            cultural_background: Cultural context
            attachment_style: Attachment style
        """
        self.agent_id = agent_id
        
        # Emotional state
        if initial_emotion is None:
            initial_emotion = {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        
        self.emotional_state = EmotionalState(
            valence=initial_emotion.get('valence', 0.0),
            arousal=initial_emotion.get('arousal', 0.0),
            dominance=initial_emotion.get('dominance', 0.0)
        )
        
        # Personality traits (Big Five)
        if personality is None:
            personality = {}
        
        self.personality = {
            'extraversion': personality.get('extraversion', 0.5),
            'neuroticism': personality.get('neuroticism', 0.5),
            'agreeableness': personality.get('agreeableness', 0.5),
            'conscientiousness': personality.get('conscientiousness', 0.5),
            'openness': personality.get('openness', 0.5)
        }
        
        # Social characteristics
        self.cultural_background = cultural_background
        self.attachment_style = attachment_style
        
        # Emotional susceptibility (how easily influenced)
        self.susceptibility = 0.5 + 0.3 * self.personality['neuroticism']
        
        # Emotional expressiveness
        self.expressiveness = 0.5 + 0.3 * self.personality['extraversion']
        
        # Empathy capacity
        self.empathy = 0.5 + 0.3 * self.personality['agreeableness']
        
        # History
        self.emotion_history: List[EmotionalState] = []
        self.interaction_history: List[Dict] = []
        
        # Beliefs about others (Theory of Mind)
        self.beliefs_about_others: Dict[str, EmotionalState] = {}
    
    def get_emotion(self) -> EmotionalState:
        """Get current emotional state."""
        return self.emotional_state
    
    def set_emotion(self, emotion: Dict):
        """Set emotional state."""
        self.emotional_state.valence = emotion.get('valence', self.emotional_state.valence)
        self.emotional_state.arousal = emotion.get('arousal', self.emotional_state.arousal)
        self.emotional_state.dominance = emotion.get('dominance', self.emotional_state.dominance)
        
        # Update emotion type based on valence-arousal
        self.emotional_state.emotion_type = self._classify_emotion()
        
        # Update intensity
        self.emotional_state.intensity = np.sqrt(
            self.emotional_state.valence ** 2 + self.emotional_state.arousal ** 2
        )
        
        # Record in history
        self.emotion_history.append(EmotionalState(
            valence=self.emotional_state.valence,
            arousal=self.emotional_state.arousal,
            dominance=self.emotional_state.dominance,
            emotion_type=self.emotional_state.emotion_type,
            intensity=self.emotional_state.intensity
        ))
    
    def update_emotion(self, delta: Dict, decay: float = 0.1):
        """
        Update emotional state with decay.
        
        Args:
            delta: Change in emotional dimensions
            decay: Decay rate toward neutral
        """
        # Apply changes
        new_valence = self.emotional_state.valence + delta.get('valence', 0)
        new_arousal = self.emotional_state.arousal + delta.get('arousal', 0)
        new_dominance = self.emotional_state.dominance + delta.get('dominance', 0)
        
        # Apply decay toward neutral
        new_valence *= (1 - decay)
        new_arousal *= (1 - decay)
        new_dominance *= (1 - decay)
        
        # Clip to valid range
        new_valence = np.clip(new_valence, -1, 1)
        new_arousal = np.clip(new_arousal, -1, 1)
        new_dominance = np.clip(new_dominance, -1, 1)
        
        self.set_emotion({
            'valence': new_valence,
            'arousal': new_arousal,
            'dominance': new_dominance
        })
    
    def _classify_emotion(self) -> EmotionType:
        """Classify emotion based on valence-arousal."""
        v = self.emotional_state.valence
        a = self.emotional_state.arousal
        
        if abs(v) < 0.2 and abs(a) < 0.2:
            return EmotionType.NEUTRAL
        elif v > 0.5 and a > 0.3:
            return EmotionType.JOY
        elif v < -0.5 and a < -0.2:
            return EmotionType.SADNESS
        elif v < -0.3 and a > 0.5:
            return EmotionType.ANGER
        elif v < -0.2 and a > 0.3:
            return EmotionType.FEAR
        elif v < -0.4:
            return EmotionType.DISGUST
        elif a > 0.6:
            return EmotionType.SURPRISE
        else:
            return EmotionType.NEUTRAL
    
    def perceive_other_emotion(
        self,
        other_agent: 'Agent',
        accuracy: float = 0.8
    ) -> EmotionalState:
        """
        Perceive another agent's emotion with some noise.
        
        Args:
            other_agent: Target agent
            accuracy: Perception accuracy (0 to 1)
            
        Returns:
            Perceived emotional state
        """
        true_emotion = other_agent.get_emotion()
        
        # Add perception noise
        noise_scale = 1 - accuracy
        perceived_valence = true_emotion.valence + np.random.normal(0, noise_scale * 0.2)
        perceived_arousal = true_emotion.arousal + np.random.normal(0, noise_scale * 0.2)
        perceived_dominance = true_emotion.dominance + np.random.normal(0, noise_scale * 0.2)
        
        # Clip
        perceived_valence = np.clip(perceived_valence, -1, 1)
        perceived_arousal = np.clip(perceived_arousal, -1, 1)
        perceived_dominance = np.clip(perceived_dominance, -1, 1)
        
        perceived = EmotionalState(
            valence=perceived_valence,
            arousal=perceived_arousal,
            dominance=perceived_dominance
        )
        
        # Store belief
        self.beliefs_about_others[other_agent.agent_id] = perceived
        
        return perceived
    
    def empathize_with(self, other_agent: 'Agent', strength: float = None) -> Dict:
        """
        Empathize with another agent's emotion.
        
        Args:
            other_agent: Target agent
            strength: Empathy strength (uses agent's empathy if None)
            
        Returns:
            Emotional change to apply
        """
        if strength is None:
            strength = self.empathy
        
        # Perceive other's emotion
        perceived_emotion = self.perceive_other_emotion(other_agent)
        
        # Compute emotional change (partial adoption of other's emotion)
        delta_valence = strength * (perceived_emotion.valence - self.emotional_state.valence)
        delta_arousal = strength * (perceived_emotion.arousal - self.emotional_state.arousal)
        
        # Modulate by susceptibility
        delta_valence *= self.susceptibility
        delta_arousal *= self.susceptibility
        
        return {
            'valence': delta_valence,
            'arousal': delta_arousal
        }
    
    def express_emotion(self) -> Dict:
        """
        Express current emotion (modulated by expressiveness).
        
        Returns:
            Observable emotional expression
        """
        expression_intensity = self.expressiveness * self.emotional_state.intensity
        
        return {
            'valence': self.emotional_state.valence * expression_intensity,
            'arousal': self.emotional_state.arousal * expression_intensity,
            'dominance': self.emotional_state.dominance * expression_intensity,
            'intensity': expression_intensity
        }
    
    def get_emotional_volatility(self, window: int = 10) -> float:
        """
        Compute emotional volatility over recent history.
        
        Args:
            window: Number of recent states to consider
            
        Returns:
            Volatility measure
        """
        if len(self.emotion_history) < 2:
            return 0.0
        
        recent = self.emotion_history[-window:]
        
        # Compute variance in valence and arousal
        valences = [e.valence for e in recent]
        arousals = [e.arousal for e in recent]
        
        volatility = np.std(valences) + np.std(arousals)
        
        return volatility
    
    def __repr__(self) -> str:
        return (
            f"Agent({self.agent_id}: "
            f"{self.emotional_state.emotion_type.value}, "
            f"v={self.emotional_state.valence:+.2f}, "
            f"a={self.emotional_state.arousal:+.2f})"
        )
