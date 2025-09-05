"""Attachment style modulation of emotional dynamics."""

from enum import Enum
from typing import Dict
import numpy as np


class AttachmentStyle(Enum):
    """Attachment style types."""
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"
    DISORGANIZED = "disorganized"


class AttachmentModulator:
    """
    Modulates emotional dynamics based on attachment style.
    
    Different attachment styles affect emotional reactivity,
    regulation, and interpersonal dynamics.
    """
    
    def __init__(self, attachment_style: AttachmentStyle):
        """Initialize with attachment style."""
        self.attachment_style = attachment_style
    
    def modulate_reactivity(self, emotional_stimulus: float) -> float:
        """
        Modulate emotional reactivity to stimulus.
        
        Args:
            emotional_stimulus: Stimulus intensity
            
        Returns:
            Modulated reactivity
        """
        if self.attachment_style == AttachmentStyle.SECURE:
            # Balanced reactivity
            return emotional_stimulus
        
        elif self.attachment_style == AttachmentStyle.ANXIOUS:
            # Heightened reactivity
            return emotional_stimulus * 1.5
        
        elif self.attachment_style == AttachmentStyle.AVOIDANT:
            # Dampened reactivity
            return emotional_stimulus * 0.6
        
        elif self.attachment_style == AttachmentStyle.DISORGANIZED:
            # Unpredictable reactivity
            return emotional_stimulus * np.random.uniform(0.5, 2.0)
        
        return emotional_stimulus
    
    def modulate_regulation(self, regulation_effort: float) -> float:
        """
        Modulate emotion regulation effectiveness.
        
        Args:
            regulation_effort: Base regulation effort
            
        Returns:
            Modulated effectiveness
        """
        if self.attachment_style == AttachmentStyle.SECURE:
            # Effective regulation
            return regulation_effort * 1.2
        
        elif self.attachment_style == AttachmentStyle.ANXIOUS:
            # Less effective regulation
            return regulation_effort * 0.7
        
        elif self.attachment_style == AttachmentStyle.AVOIDANT:
            # Suppression-based regulation
            return regulation_effort * 0.9
        
        elif self.attachment_style == AttachmentStyle.DISORGANIZED:
            # Inconsistent regulation
            return regulation_effort * np.random.uniform(0.5, 1.0)
        
        return regulation_effort
    
    def modulate_social_seeking(self, distress_level: float) -> float:
        """
        Modulate tendency to seek social support when distressed.
        
        Args:
            distress_level: Level of distress
            
        Returns:
            Social seeking tendency
        """
        if self.attachment_style == AttachmentStyle.SECURE:
            # Appropriate social seeking
            return distress_level * 0.8
        
        elif self.attachment_style == AttachmentStyle.ANXIOUS:
            # Excessive social seeking
            return distress_level * 1.5
        
        elif self.attachment_style == AttachmentStyle.AVOIDANT:
            # Minimal social seeking
            return distress_level * 0.3
        
        elif self.attachment_style == AttachmentStyle.DISORGANIZED:
            # Conflicted social seeking
            return distress_level * np.random.uniform(0.2, 1.2)
        
        return distress_level * 0.5
