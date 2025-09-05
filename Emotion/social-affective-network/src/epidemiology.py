"""
Emotional epidemiology simulator.

Models emotion spread as epidemic-like processes on social networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Optional
from enum import Enum
from .network import SocialNetwork


class EmotionState(Enum):
    """Emotional infection state."""
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"


class EmotionalEpidemiology:
    """
    Emotional epidemiology simulator.
    
    Models emotion spread using SIR-like dynamics on social networks.
    """
    
    def __init__(
        self,
        network: SocialNetwork,
        emotion_type: str = 'anger',
        transmission_rate: float = 0.3,
        recovery_rate: float = 0.1,
        threshold: float = 0.5
    ):
        """
        Initialize epidemiology simulator.
        
        Args:
            network: Social network
            emotion_type: Type of emotion spreading
            transmission_rate: Probability of transmission per contact
            recovery_rate: Rate of recovery to neutral
            threshold: Intensity threshold for "infection"
        """
        self.network = network
        self.emotion_type = emotion_type
        self.transmission_rate = transmission_rate
        self.recovery_rate = recovery_rate
        self.threshold = threshold
        
        # Track states
        self.states: Dict[str, EmotionState] = {}
        for agent_id in network.agents.keys():
            self.states[agent_id] = EmotionState.SUSCEPTIBLE
        
        # History
        self.history: List[Dict] = []
    
    def seed_emotion(
        self,
        patient_zeros: List[str],
        intensity: float = 0.9
    ):
        """
        Seed initial emotional infection.
        
        Args:
            patient_zeros: List of initial infected agent IDs
            intensity: Initial emotion intensity
        """
        for agent_id in patient_zeros:
            agent = self.network.get_agent(agent_id)
            if agent:
                # Set strong emotion
                if self.emotion_type == 'anger':
                    agent.set_emotion({'valence': -0.7, 'arousal': intensity})
                elif self.emotion_type == 'joy':
                    agent.set_emotion({'valence': 0.8, 'arousal': intensity})
                elif self.emotion_type == 'fear':
                    agent.set_emotion({'valence': -0.6, 'arousal': intensity})
                elif self.emotion_type == 'sadness':
                    agent.set_emotion({'valence': -0.8, 'arousal': -0.3})
                
                self.states[agent_id] = EmotionState.INFECTED
    
    def step(self):
        """Execute one epidemic step."""
        new_infections = []
        new_recoveries = []
        
        # Process each agent
        for agent_id, state in self.states.items():
            agent = self.network.get_agent(agent_id)
            
            if state == EmotionState.INFECTED:
                # Try to infect neighbors
                neighbors = self.network.get_neighbors(agent_id)
                
                for neighbor in neighbors:
                    if self.states[neighbor.agent_id] == EmotionState.SUSCEPTIBLE:
                        # Transmission probability
                        if np.random.rand() < self.transmission_rate:
                            # Infect neighbor
                            neighbor.set_emotion({
                                'valence': agent.emotional_state.valence * 0.8,
                                'arousal': agent.emotional_state.arousal * 0.8
                            })
                            new_infections.append(neighbor.agent_id)
                
                # Try to recover
                if np.random.rand() < self.recovery_rate:
                    new_recoveries.append(agent_id)
            
            elif state == EmotionState.SUSCEPTIBLE:
                # Check if intensity crosses threshold
                if agent.emotional_state.intensity > self.threshold:
                    new_infections.append(agent_id)
        
        # Update states
        for agent_id in new_infections:
            self.states[agent_id] = EmotionState.INFECTED
        
        for agent_id in new_recoveries:
            self.states[agent_id] = EmotionState.RECOVERED
            # Reset to neutral
            agent = self.network.get_agent(agent_id)
            agent.set_emotion({'valence': 0.0, 'arousal': 0.0})
        
        # Record state
        self._record_state()
        self.network.step()
    
    def simulate(self, steps: int) -> List[Dict]:
        """
        Simulate epidemic for multiple steps.
        
        Args:
            steps: Number of steps
            
        Returns:
            History of epidemic states
        """
        for _ in range(steps):
            self.step()
        
        return self.history
    
    def _record_state(self):
        """Record current epidemic state."""
        counts = {
            'susceptible': 0,
            'infected': 0,
            'recovered': 0
        }
        
        for state in self.states.values():
            counts[state.value] += 1
        
        counts['time'] = self.network.time_step
        self.history.append(counts)
    
    def identify_influencers(self, top_k: int = 10) -> List[str]:
        """
        Identify most influential agents for emotion spread.
        
        Args:
            top_k: Number of top influencers
            
        Returns:
            List of agent IDs
        """
        # Use betweenness centrality as proxy for influence
        central_agents = self.network.identify_central_agents(top_k)
        return [agent_id for agent_id, _ in central_agents]
    
    def apply_intervention(
        self,
        target_nodes: List[str],
        intervention_type: str = 'regulation',
        effectiveness: float = 0.7
    ):
        """
        Apply intervention to target nodes.
        
        Args:
            target_nodes: List of agent IDs to target
            intervention_type: Type of intervention
            effectiveness: Intervention effectiveness (0 to 1)
        """
        for agent_id in target_nodes:
            agent = self.network.get_agent(agent_id)
            if agent:
                if intervention_type == 'regulation':
                    # Reduce emotional intensity
                    agent.emotional_state.valence *= (1 - effectiveness)
                    agent.emotional_state.arousal *= (1 - effectiveness)
                    
                    # Mark as recovered if below threshold
                    if agent.emotional_state.intensity < self.threshold:
                        self.states[agent_id] = EmotionState.RECOVERED
                
                elif intervention_type == 'isolation':
                    # Remove connections temporarily
                    self.states[agent_id] = EmotionState.RECOVERED
    
    def compute_r0(self) -> float:
        """
        Compute basic reproduction number R0.
        
        Returns:
            R0 value
        """
        # Average number of connections
        avg_degree = np.mean([self.network.graph.degree(node) 
                             for node in self.network.graph.nodes()])
        
        # R0 = transmission_rate * avg_contacts * infectious_period
        infectious_period = 1.0 / self.recovery_rate if self.recovery_rate > 0 else 10
        r0 = self.transmission_rate * avg_degree * infectious_period
        
        return r0
    
    def plot_epidemic_curve(self, figsize=(12, 6)):
        """Plot epidemic curve over time."""
        if not self.history:
            return
        
        times = [h['time'] for h in self.history]
        susceptible = [h['susceptible'] for h in self.history]
        infected = [h['infected'] for h in self.history]
        recovered = [h['recovered'] for h in self.history]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(times, susceptible, 'b-', linewidth=2, label='Susceptible')
        ax.plot(times, infected, 'r-', linewidth=2, label='Infected')
        ax.plot(times, recovered, 'g-', linewidth=2, label='Recovered')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Number of Agents', fontsize=12)
        ax.set_title(f'Emotional Epidemic Curve: {self.emotion_type.title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add R0 annotation
        r0 = self.compute_r0()
        ax.text(0.02, 0.98, f'R₀ = {r0:.2f}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig, ax
    
    def get_peak_info(self) -> Dict:
        """Get information about epidemic peak."""
        if not self.history:
            return {}
        
        infected_counts = [h['infected'] for h in self.history]
        peak_count = max(infected_counts)
        peak_time = infected_counts.index(peak_count)
        
        return {
            'peak_count': peak_count,
            'peak_time': peak_time,
            'peak_proportion': peak_count / len(self.network.agents)
        }
