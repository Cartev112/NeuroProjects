"""Cultural emotion norms and display rules."""

import numpy as np
from typing import Dict


class CulturalNorms:
    """
    Cultural emotion norms that modulate expression and interpretation.
    
    Encodes culture-specific display rules and emotion concepts.
    """
    
    def __init__(
        self,
        culture: str = 'western',
        individualism: float = 0.7,
        power_distance: float = 0.3,
        display_rules: Dict[str, str] = None
    ):
        """
        Initialize cultural norms.
        
        Args:
            culture: Cultural identifier
            individualism: Individualism vs collectivism (0 to 1)
            power_distance: Power distance index (0 to 1)
            display_rules: Emotion-specific display rules
        """
        self.culture = culture
        self.individualism = individualism
        self.power_distance = power_distance
        
        if display_rules is None:
            display_rules = self._default_display_rules()
        self.display_rules = display_rules
    
    def _default_display_rules(self) -> Dict[str, str]:
        """Default display rules based on culture."""
        if self.individualism > 0.6:
            # Western/individualistic
            return {
                'anger': 'moderate',
                'sadness': 'suppress',
                'joy': 'express',
                'fear': 'moderate'
            }
        else:
            # Eastern/collectivistic
            return {
                'anger': 'suppress',
                'sadness': 'moderate',
                'joy': 'moderate',
                'fear': 'suppress'
            }
    
    def modulate_expression(self, emotion: Dict) -> Dict:
        """
        Modulate emotional expression according to cultural norms.
        
        Args:
            emotion: Raw emotional state
            
        Returns:
            Culturally modulated expression
        """
        modulated = emotion.copy()
        
        # Apply individualism effect
        expression_factor = 0.5 + 0.5 * self.individualism
        
        modulated['intensity'] = emotion.get('intensity', 0.5) * expression_factor
        
        # Apply display rules
        if emotion.get('valence', 0) < -0.5:
            # Negative emotion
            if self.individualism < 0.5:
                # Suppress in collectivistic cultures
                modulated['intensity'] *= 0.7
        
        return modulated
    
    def interpret_expression(self, observed: Dict) -> Dict:
        """Interpret observed expression through cultural lens."""
        # Account for cultural display rules in interpretation
        interpretation = observed.copy()
        
        # If collectivistic culture, assume suppression
        if self.individualism < 0.5:
            interpretation['intensity'] = observed.get('intensity', 0.5) * 1.3
        
        return interpretation
