"""Game-theoretic emotional signaling."""

import numpy as np
from typing import Dict, Tuple


class EmotionalSignalingGame:
    """
    Game-theoretic framework for emotional displays as strategic signals.
    
    Models emotions as costly signals with payoff structures.
    """
    
    def __init__(self, signal_cost: float = 0.1, deception_penalty: float = 0.5):
        """
        Initialize signaling game.
        
        Args:
            signal_cost: Cost of displaying emotion
            deception_penalty: Penalty for deceptive signals
        """
        self.signal_cost = signal_cost
        self.deception_penalty = deception_penalty
    
    def compute_payoff(
        self,
        true_emotion: float,
        displayed_emotion: float,
        receiver_response: float
    ) -> Tuple[float, float]:
        """
        Compute payoffs for sender and receiver.
        
        Args:
            true_emotion: Sender's true emotional state
            displayed_emotion: Displayed emotional signal
            receiver_response: Receiver's response
            
        Returns:
            Tuple of (sender_payoff, receiver_payoff)
        """
        # Signal cost (proportional to intensity)
        cost = self.signal_cost * abs(displayed_emotion)
        
        # Deception cost (if display differs from true emotion)
        deception = abs(displayed_emotion - true_emotion)
        deception_cost = self.deception_penalty * deception
        
        # Benefit from receiver response
        benefit = receiver_response * displayed_emotion
        
        # Sender payoff
        sender_payoff = benefit - cost - deception_cost
        
        # Receiver payoff (benefit from accurate perception)
        accuracy = 1.0 - abs(displayed_emotion - true_emotion)
        receiver_payoff = accuracy * receiver_response
        
        return sender_payoff, receiver_payoff
    
    def find_nash_equilibrium(
        self,
        true_emotion: float,
        n_iterations: int = 100
    ) -> Dict:
        """
        Find Nash equilibrium for emotional display.
        
        Args:
            true_emotion: True emotional state
            n_iterations: Optimization iterations
            
        Returns:
            Equilibrium strategy
        """
        # Simplified: honest signaling is often equilibrium
        # when deception costs are high
        
        if self.deception_penalty > self.signal_cost:
            # Honest signaling equilibrium
            optimal_display = true_emotion
            strategy = 'honest'
        else:
            # May have incentive to exaggerate
            optimal_display = true_emotion * 1.2
            strategy = 'exaggerated'
        
        return {
            'optimal_display': optimal_display,
            'strategy': strategy,
            'expected_payoff': self.compute_payoff(true_emotion, optimal_display, 1.0)[0]
        }
