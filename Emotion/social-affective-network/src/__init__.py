"""
SARN: Social Affective Resonance Network

A multi-agent system for modeling emotional contagion, empathy,
and collective emotional dynamics in social networks.
"""

from .network import SocialNetwork
from .agent import Agent, EmotionalState
from .propagation import EmotionPropagator, GNNPropagator
from .mirror_neurons import MirrorNeuronSystem
from .emotion_recognition import BayesianEmotionRecognizer
from .theory_of_mind import TheoryOfMindModule, MentalState
from .cultural_norms import CulturalNorms
from .synchrony import SynchronyDetector
from .game_theory import EmotionalSignalingGame
from .attachment import AttachmentStyle, AttachmentModulator
from .epidemiology import EmotionalEpidemiology

__version__ = "0.1.0"

__all__ = [
    "SocialNetwork",
    "Agent",
    "EmotionalState",
    "EmotionPropagator",
    "GNNPropagator",
    "MirrorNeuronSystem",
    "BayesianEmotionRecognizer",
    "TheoryOfMindModule",
    "MentalState",
    "CulturalNorms",
    "SynchronyDetector",
    "EmotionalSignalingGame",
    "AttachmentStyle",
    "AttachmentModulator",
    "EmotionalEpidemiology",
]
