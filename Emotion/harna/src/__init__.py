"""
HARNA: Hierarchical Appraisal-Regulation Neural Architecture

A multi-layered computational model integrating cognitive appraisal theory
with affective neuroscience.
"""

from .model import HARNAModel, Stimulus, EmotionalResponse
from .subcortical import SubcorticalPathway, ThreatDetector
from .appraisal import AppraisalNetwork, SchererAppraisal
from .predictive_processing import PredictiveProcessor, InteroceptivePredictor
from .regulation import RegulationModule, RegulationStrategy
from .rl_agent import EmotionalRLAgent
from .individual_differences import IndividualDifferences, PersonalityTraits
from .timescales import MultiTimescaleProcessor, TimescaleAnalyzer
from .theories import EmotionTheory, TheoryType
from .model_comparison import ModelComparison, ComparisonMetrics

__version__ = "0.1.0"

__all__ = [
    "HARNAModel",
    "Stimulus",
    "EmotionalResponse",
    "SubcorticalPathway",
    "ThreatDetector",
    "AppraisalNetwork",
    "SchererAppraisal",
    "PredictiveProcessor",
    "InteroceptivePredictor",
    "RegulationModule",
    "RegulationStrategy",
    "EmotionalRLAgent",
    "IndividualDifferences",
    "PersonalityTraits",
    "MultiTimescaleProcessor",
    "TimescaleAnalyzer",
    "EmotionTheory",
    "TheoryType",
    "ModelComparison",
    "ComparisonMetrics",
]
