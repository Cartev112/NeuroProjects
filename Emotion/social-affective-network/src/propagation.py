"""
Emotion propagation through social networks.

Implements contagion dynamics using graph neural networks
and message passing algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .network import SocialNetwork
from .agent import Agent


class EmotionPropagator:
    """
    Basic emotion propagation through social network.
    
    Implements simple contagion dynamics where emotions
    spread through network connections.
    """
    
    def __init__(
        self,
        network: SocialNetwork,
        decay_rate: float = 0.1,
        transmission_rate: float = 0.3
    ):
        """
        Initialize propagator.
        
        Args:
            network: Social network
            decay_rate: Rate of emotional decay
            transmission_rate: Rate of emotional transmission
        """
        self.network = network
        self.decay_rate = decay_rate
        self.transmission_rate = transmission_rate
        
        # Track spread statistics
        self.infected_agents: set = set()
        self.infection_times: Dict[str, int] = {}
    
    def step(self):
        """Execute one step of emotion propagation."""
        # Collect updates for all agents
        updates = {}
        
        for agent_id, agent in self.network.agents.items():
            # Get neighbors
            neighbors = self.network.get_neighbors(agent_id)
            
            if not neighbors:
                continue
            
            # Accumulate emotional influence from neighbors
            total_influence = {'valence': 0.0, 'arousal': 0.0}
            
            for neighbor in neighbors:
                # Get connection weight
                weight = self.network.get_connection_weight(agent_id, neighbor.agent_id)
                
                # Compute influence (weighted by connection and transmission rate)
                influence_strength = weight * self.transmission_rate
                
                # Empathize with neighbor
                empathy_delta = agent.empathize_with(neighbor, strength=influence_strength)
                
                total_influence['valence'] += empathy_delta['valence']
                total_influence['arousal'] += empathy_delta['arousal']
            
            # Average influence
            if neighbors:
                total_influence['valence'] /= len(neighbors)
                total_influence['arousal'] /= len(neighbors)
            
            updates[agent_id] = total_influence
        
        # Apply updates
        for agent_id, delta in updates.items():
            agent = self.network.get_agent(agent_id)
            agent.update_emotion(delta, decay=self.decay_rate)
            
            # Track infection
            if agent.emotional_state.intensity > 0.3:
                if agent_id not in self.infected_agents:
                    self.infected_agents.add(agent_id)
                    self.infection_times[agent_id] = self.network.time_step
        
        # Advance network time
        self.network.step()
    
    def simulate(self, n_steps: int) -> List[Dict]:
        """
        Simulate propagation for multiple steps.
        
        Args:
            n_steps: Number of steps
            
        Returns:
            History of network states
        """
        history = []
        
        for _ in range(n_steps):
            self.step()
            state = self.network.get_network_emotion_state()
            state['n_infected'] = len(self.infected_agents)
            history.append(state)
        
        return history
    
    def analyze_spread(self) -> Dict:
        """Analyze emotion spread statistics."""
        total_agents = len(self.network.agents)
        
        if total_agents == 0:
            return {}
        
        infection_rate = len(self.infected_agents) / total_agents
        
        # Find peak time
        if self.infection_times:
            infection_counts = {}
            for time in self.infection_times.values():
                infection_counts[time] = infection_counts.get(time, 0) + 1
            peak_time = max(infection_counts, key=infection_counts.get)
        else:
            peak_time = 0
        
        return {
            'infection_rate': infection_rate,
            'n_infected': len(self.infected_agents),
            'peak_time': peak_time,
            'total_agents': total_agents
        }
    
    def reset(self):
        """Reset propagation state."""
        self.infected_agents = set()
        self.infection_times = {}
        self.network.reset()


class GNNPropagator(nn.Module):
    """
    Graph Neural Network for emotion propagation.
    
    Uses message passing to propagate emotions through network.
    """
    
    def __init__(
        self,
        emotion_dim: int = 3,  # valence, arousal, dominance
        hidden_dim: int = 64,
        n_layers: int = 2
    ):
        """
        Initialize GNN propagator.
        
        Args:
            emotion_dim: Emotion state dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of message passing layers
        """
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        
        for i in range(n_layers):
            if i == 0:
                input_dim = emotion_dim
            else:
                input_dim = hidden_dim
            
            # Message function
            self.message_layers.append(nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            
            # Update function
            self.update_layers.append(nn.GRUCell(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, emotion_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            node_features: Node features [n_nodes, emotion_dim]
            edge_index: Edge indices [2, n_edges]
            edge_weights: Edge weights [n_edges]
            
        Returns:
            Updated node features [n_nodes, emotion_dim]
        """
        x = node_features
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        # Message passing layers
        for layer_idx in range(self.n_layers):
            # Aggregate messages
            messages = self._aggregate_messages(
                x if layer_idx == 0 else h,
                edge_index,
                edge_weights,
                layer_idx
            )
            
            # Update node states
            h = self.update_layers[layer_idx](messages, h)
        
        # Output
        output = self.output_layer(h)
        
        # Add residual connection
        output = output + node_features
        
        return output
    
    def _aggregate_messages(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor],
        layer_idx: int
    ) -> torch.Tensor:
        """Aggregate messages from neighbors."""
        n_nodes = node_features.size(0)
        aggregated = torch.zeros(n_nodes, self.hidden_dim, device=node_features.device)
        
        # For each edge
        for i in range(edge_index.size(1)):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            
            # Concatenate source and destination features
            combined = torch.cat([node_features[src], node_features[dst]], dim=0)
            
            # Compute message
            message = self.message_layers[layer_idx](combined)
            
            # Weight by edge weight
            if edge_weights is not None:
                message = message * edge_weights[i]
            
            # Aggregate to destination
            aggregated[dst] += message
        
        return aggregated
    
    def propagate_emotions(
        self,
        network: SocialNetwork,
        n_steps: int = 10
    ) -> List[Dict]:
        """
        Propagate emotions through network using GNN.
        
        Args:
            network: Social network
            n_steps: Number of propagation steps
            
        Returns:
            History of network states
        """
        history = []
        
        with torch.no_grad():
            for step in range(n_steps):
                # Extract node features
                node_features = self._extract_node_features(network)
                
                # Extract edge information
                edge_index, edge_weights = self._extract_edges(network)
                
                # Forward pass
                updated_features = self.forward(node_features, edge_index, edge_weights)
                
                # Update network
                self._update_network(network, updated_features)
                
                # Record state
                network.step()
                history.append(network.get_network_emotion_state())
        
        return history
    
    def _extract_node_features(self, network: SocialNetwork) -> torch.Tensor:
        """Extract emotion features from network."""
        features = []
        
        for agent_id in sorted(network.agents.keys()):
            agent = network.get_agent(agent_id)
            features.append([
                agent.emotional_state.valence,
                agent.emotional_state.arousal,
                agent.emotional_state.dominance
            ])
        
        return torch.FloatTensor(features)
    
    def _extract_edges(
        self,
        network: SocialNetwork
    ) -> tuple:
        """Extract edge information from network."""
        edge_list = []
        edge_weights = []
        
        # Create node ID to index mapping
        node_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(network.agents.keys()))}
        
        for edge in network.graph.edges():
            src_idx = node_to_idx[edge[0]]
            dst_idx = node_to_idx[edge[1]]
            
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])  # Undirected
            
            weight = network.get_connection_weight(edge[0], edge[1])
            edge_weights.append(weight)
            edge_weights.append(weight)
        
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t()
            edge_weights_tensor = torch.FloatTensor(edge_weights)
        else:
            edge_index = torch.LongTensor([[],[]])
            edge_weights_tensor = torch.FloatTensor([])
        
        return edge_index, edge_weights_tensor
    
    def _update_network(
        self,
        network: SocialNetwork,
        updated_features: torch.Tensor
    ):
        """Update network with new emotion features."""
        for idx, agent_id in enumerate(sorted(network.agents.keys())):
            agent = network.get_agent(agent_id)
            
            new_emotion = {
                'valence': updated_features[idx, 0].item(),
                'arousal': updated_features[idx, 1].item(),
                'dominance': updated_features[idx, 2].item()
            }
            
            agent.set_emotion(new_emotion)
