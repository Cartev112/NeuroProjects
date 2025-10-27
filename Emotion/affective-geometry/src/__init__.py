"""
Affective Geometry: Dynamical Systems Mapping of Emotional Landscapes

A comprehensive framework for modeling emotions as trajectories through
high-dimensional state spaces using nonlinear dynamical systems theory.
"""

from .state_space import EmotionalStateSpace
from .attractors import Attractor, AttractorType
from .recurrence import RecurrenceAnalysis
from .lyapunov import LyapunovAnalysis
from .bifurcation import BifurcationAnalysis
from .topology import TopologicalAnalysis
from .reservoir import EmotionalPredictor
from .intervention import InterventionOptimizer

__version__ = "0.1.0"

__all__ = [
    "EmotionalStateSpace",
    "Attractor",
    "AttractorType",
    "RecurrenceAnalysis",
    "LyapunovAnalysis",
    "BifurcationAnalysis",
    "TopologicalAnalysis",
    "EmotionalPredictor",
    "InterventionOptimizer",
]
