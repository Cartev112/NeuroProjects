"""
Competing emotion theory implementations.

Implements James-Lange, Cannon-Bard, Schachter-Singer, and
Constructionist theories for model comparison.
"""

import numpy as np
from enum import Enum
from typing import Dict
from dataclasses import dataclass


class TheoryType(Enum):
    """Emotion theory types."""
    JAMES_LANGE = "james_lange"
    CANNON_BARD = "cannon_bard"
    SCHACHTER_SINGER = "schachter_singer"
    CONSTRUCTIONIST = "constructionist"
    SCHERER = "scherer"


@dataclass
class EmotionTheoryResponse:
    """Response from emotion theory."""
    theory: str
    emotion_label: str
    valence: float
    arousal: float
    bodily_response: np.ndarray
    cognitive_label: str
    processing_pathway: str
    
    def to_dict(self) -> Dict:
        return {
            'theory': self.theory,
            'emotion_label': self.emotion_label,
            'valence': self.valence,
            'arousal': self.arousal,
            'cognitive_label': self.cognitive_label,
            'processing_pathway': self.processing_pathway
        }


class EmotionTheory:
    """
    Base class for emotion theories.
    
    Each theory implements a different causal model of emotion generation.
    """
    
    def __init__(self, theory_type: TheoryType):
        self.theory_type = theory_type
    
    def generate_emotion(
        self,
        stimulus: np.ndarray,
        context: Dict
    ) -> EmotionTheoryResponse:
        """Generate emotional response according to theory."""
        raise NotImplementedError


class JamesLangeTheory(EmotionTheory):
    """
    James-Lange Theory: Emotion follows from bodily response.
    
    Stimulus → Bodily Response → Emotion
    """
    
    def __init__(self):
        super().__init__(TheoryType.JAMES_LANGE)
    
    def generate_emotion(
        self,
        stimulus: np.ndarray,
        context: Dict
    ) -> EmotionTheoryResponse:
        # 1. Stimulus triggers bodily response
        bodily_response = self._compute_bodily_response(stimulus)
        
        # 2. Emotion is perception of bodily response
        valence, arousal = self._perceive_bodily_state(bodily_response)
        
        # 3. Label emotion based on bodily pattern
        emotion_label = self._label_from_body(bodily_response)
        
        return EmotionTheoryResponse(
            theory=self.theory_type.value,
            emotion_label=emotion_label,
            valence=valence,
            arousal=arousal,
            bodily_response=bodily_response,
            cognitive_label="",
            processing_pathway="stimulus→body→emotion"
        )
    
    def _compute_bodily_response(self, stimulus: np.ndarray) -> np.ndarray:
        """Compute bodily response to stimulus."""
        # Simplified: random bodily pattern
        return np.random.randn(10)
    
    def _perceive_bodily_state(self, bodily_response: np.ndarray) -> tuple:
        """Perceive emotion from bodily state."""
        valence = np.tanh(np.mean(bodily_response))
        arousal = np.abs(np.mean(bodily_response))
        return valence, arousal
    
    def _label_from_body(self, bodily_response: np.ndarray) -> str:
        """Label emotion from bodily pattern."""
        if np.mean(bodily_response) > 0.5:
            return "joy"
        elif np.mean(bodily_response) < -0.5:
            return "sadness"
        else:
            return "neutral"


class CannonBardTheory(EmotionTheory):
    """
    Cannon-Bard Theory: Emotion and bodily response occur simultaneously.
    
    Stimulus → (Emotion + Bodily Response)
    """
    
    def __init__(self):
        super().__init__(TheoryType.CANNON_BARD)
    
    def generate_emotion(
        self,
        stimulus: np.ndarray,
        context: Dict
    ) -> EmotionTheoryResponse:
        # Simultaneous processing
        valence, arousal = self._evaluate_stimulus(stimulus)
        bodily_response = self._trigger_bodily_response(valence, arousal)
        emotion_label = self._label_emotion(valence, arousal)
        
        return EmotionTheoryResponse(
            theory=self.theory_type.value,
            emotion_label=emotion_label,
            valence=valence,
            arousal=arousal,
            bodily_response=bodily_response,
            cognitive_label="",
            processing_pathway="stimulus→(emotion+body)"
        )
    
    def _evaluate_stimulus(self, stimulus: np.ndarray) -> tuple:
        """Direct evaluation of stimulus."""
        valence = np.tanh(np.mean(stimulus))
        arousal = np.abs(np.mean(stimulus))
        return valence, arousal
    
    def _trigger_bodily_response(self, valence: float, arousal: float) -> np.ndarray:
        """Trigger bodily response based on emotion."""
        return np.array([valence, arousal] * 5)
    
    def _label_emotion(self, valence: float, arousal: float) -> str:
        """Label emotion from valence and arousal."""
        if valence > 0.3 and arousal > 0.3:
            return "excitement"
        elif valence > 0.3:
            return "contentment"
        elif valence < -0.3 and arousal > 0.3:
            return "anger"
        elif valence < -0.3:
            return "sadness"
        else:
            return "neutral"


class SchachterSingerTheory(EmotionTheory):
    """
    Schachter-Singer Two-Factor Theory: Arousal + Cognitive Label.
    
    Stimulus → Arousal + Cognitive Label → Emotion
    """
    
    def __init__(self):
        super().__init__(TheoryType.SCHACHTER_SINGER)
    
    def generate_emotion(
        self,
        stimulus: np.ndarray,
        context: Dict
    ) -> EmotionTheoryResponse:
        # 1. Undifferentiated arousal
        arousal = self._compute_arousal(stimulus)
        
        # 2. Cognitive labeling based on context
        cognitive_label = self._apply_cognitive_label(context)
        
        # 3. Emotion from arousal + label
        valence, emotion_label = self._combine_arousal_and_label(arousal, cognitive_label)
        
        bodily_response = np.array([arousal] * 10)
        
        return EmotionTheoryResponse(
            theory=self.theory_type.value,
            emotion_label=emotion_label,
            valence=valence,
            arousal=arousal,
            bodily_response=bodily_response,
            cognitive_label=cognitive_label,
            processing_pathway="stimulus→arousal+label→emotion"
        )
    
    def _compute_arousal(self, stimulus: np.ndarray) -> float:
        """Compute undifferentiated arousal."""
        return np.abs(np.mean(stimulus))
    
    def _apply_cognitive_label(self, context: Dict) -> str:
        """Apply cognitive label based on context."""
        if context.get('threat', False):
            return "threatening"
        elif context.get('positive', False):
            return "rewarding"
        else:
            return "neutral"
    
    def _combine_arousal_and_label(self, arousal: float, label: str) -> tuple:
        """Combine arousal and label to form emotion."""
        if label == "threatening":
            return -0.7, "fear"
        elif label == "rewarding":
            return 0.7, "joy"
        else:
            return 0.0, "neutral"


class ConstructionistTheory(EmotionTheory):
    """
    Constructionist Theory (Barrett): Core Affect + Conceptualization.
    
    Core Affect + Conceptualization → Emotion
    """
    
    def __init__(self):
        super().__init__(TheoryType.CONSTRUCTIONIST)
    
    def generate_emotion(
        self,
        stimulus: np.ndarray,
        context: Dict
    ) -> EmotionTheoryResponse:
        # 1. Core affect (valence + arousal)
        valence, arousal = self._compute_core_affect(stimulus)
        
        # 2. Conceptualization based on context and concepts
        emotion_label = self._conceptualize(valence, arousal, context)
        
        # 3. Construct bodily response from concept
        bodily_response = self._construct_bodily_response(emotion_label)
        
        return EmotionTheoryResponse(
            theory=self.theory_type.value,
            emotion_label=emotion_label,
            valence=valence,
            arousal=arousal,
            bodily_response=bodily_response,
            cognitive_label=emotion_label,
            processing_pathway="core_affect+concept→emotion"
        )
    
    def _compute_core_affect(self, stimulus: np.ndarray) -> tuple:
        """Compute core affective dimensions."""
        valence = np.tanh(np.mean(stimulus))
        arousal = np.abs(np.mean(stimulus))
        return valence, arousal
    
    def _conceptualize(self, valence: float, arousal: float, context: Dict) -> str:
        """Conceptualize emotion from core affect and context."""
        # Context-dependent conceptualization
        if context.get('social', False):
            if valence > 0:
                return "love" if arousal > 0.5 else "contentment"
            else:
                return "embarrassment" if arousal > 0.5 else "loneliness"
        else:
            if valence > 0 and arousal > 0.5:
                return "excitement"
            elif valence > 0:
                return "calm"
            elif valence < 0 and arousal > 0.5:
                return "anxiety"
            else:
                return "sadness"
    
    def _construct_bodily_response(self, emotion_label: str) -> np.ndarray:
        """Construct bodily response from emotion concept."""
        # Different concepts predict different bodily patterns
        patterns = {
            "excitement": np.array([1, 1, 0.8, 0.9, 1, 0.7, 0.8, 0.9, 1, 0.8]),
            "anxiety": np.array([0.9, 0.8, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.7]),
            "sadness": np.array([-0.5, -0.6, -0.4, -0.5, -0.6, -0.3, -0.4, -0.5, -0.6, -0.4]),
        }
        return patterns.get(emotion_label, np.zeros(10))
