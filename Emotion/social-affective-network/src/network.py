"""
Social network graph structure.

Manages the network topology and agent interactions.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from .agent import Agent, EmotionalState


class SocialNetwork:
    """
    Social network graph with agents as nodes.
    
    Manages network topology, agent interactions, and
    emotional dynamics across the network.
    """
    
    def __init__(self, graph: nx.Graph = None):
        """
        Initialize social network.
        
        Args:
            graph: NetworkX graph (creates empty if None)
        """
        if graph is None:
            self.graph = nx.Graph()
        else:
            self.graph = graph.copy()
        
        # Agent storage
        self.agents: Dict[str, Agent] = {}
        
        # Network statistics
        self.time_step = 0
        self.emotion_history: List[Dict] = []
    
    def add_agent(self, agent: Agent):
        """Add agent to network."""
        self.agents[agent.agent_id] = agent
        
        # Ensure node exists in graph
        if agent.agent_id not in self.graph.nodes():
            self.graph.add_node(agent.agent_id)
    
    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def add_connection(
        self,
        agent1_id: str,
        agent2_id: str,
        weight: float = 1.0
    ):
        """
        Add social connection between agents.
        
        Args:
            agent1_id: First agent ID
            agent2_id: Second agent ID
            weight: Connection strength
        """
        self.graph.add_edge(agent1_id, agent2_id, weight=weight)
    
    def get_neighbors(self, agent_id: str) -> List[Agent]:
        """Get neighboring agents."""
        neighbor_ids = list(self.graph.neighbors(agent_id))
        return [self.agents[nid] for nid in neighbor_ids if nid in self.agents]
    
    def get_connection_weight(self, agent1_id: str, agent2_id: str) -> float:
        """Get connection weight between agents."""
        if self.graph.has_edge(agent1_id, agent2_id):
            return self.graph[agent1_id][agent2_id].get('weight', 1.0)
        return 0.0
    
    def get_network_emotion_state(self) -> Dict:
        """Get aggregate emotional state of network."""
        if not self.agents:
            return {'mean_valence': 0.0, 'mean_arousal': 0.0, 'std_valence': 0.0}
        
        valences = [agent.emotional_state.valence for agent in self.agents.values()]
        arousals = [agent.emotional_state.arousal for agent in self.agents.values()]
        
        return {
            'mean_valence': np.mean(valences),
            'mean_arousal': np.mean(arousals),
            'std_valence': np.std(valences),
            'std_arousal': np.std(arousals),
            'time_step': self.time_step
        }
    
    def record_state(self):
        """Record current network emotional state."""
        state = self.get_network_emotion_state()
        self.emotion_history.append(state)
    
    def compute_network_metrics(self) -> Dict:
        """Compute network topology metrics."""
        metrics = {}
        
        if len(self.graph.nodes()) > 0:
            metrics['n_nodes'] = self.graph.number_of_nodes()
            metrics['n_edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)
            
            if nx.is_connected(self.graph):
                metrics['avg_path_length'] = nx.average_shortest_path_length(self.graph)
                metrics['diameter'] = nx.diameter(self.graph)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
                metrics['diameter'] = nx.diameter(subgraph)
            
            metrics['clustering'] = nx.average_clustering(self.graph)
            
            # Centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            metrics['max_degree_centrality'] = max(degree_centrality.values())
            metrics['max_betweenness_centrality'] = max(betweenness_centrality.values())
        
        return metrics
    
    def identify_central_agents(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Identify most central agents in network.
        
        Args:
            top_k: Number of top agents to return
            
        Returns:
            List of (agent_id, centrality_score) tuples
        """
        # Use betweenness centrality as measure of influence
        centrality = nx.betweenness_centrality(self.graph)
        
        # Sort by centrality
        sorted_agents = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_agents[:top_k]
    
    def detect_communities(self) -> Dict[str, int]:
        """
        Detect communities in network.
        
        Returns:
            Dictionary mapping agent_id to community_id
        """
        try:
            from community import community_louvain
            communities = community_louvain.best_partition(self.graph)
            return communities
        except ImportError:
            # Fallback to greedy modularity
            communities_generator = nx.community.greedy_modularity_communities(self.graph)
            communities = {}
            for i, community in enumerate(communities_generator):
                for node in community:
                    communities[node] = i
            return communities
    
    def get_emotional_homophily(self) -> float:
        """
        Compute emotional homophily (similarity of connected agents).
        
        Returns:
            Homophily score (0 to 1)
        """
        if self.graph.number_of_edges() == 0:
            return 0.0
        
        similarities = []
        
        for edge in self.graph.edges():
            agent1 = self.agents.get(edge[0])
            agent2 = self.agents.get(edge[1])
            
            if agent1 and agent2:
                # Compute emotional similarity
                distance = agent1.emotional_state.distance_to(agent2.emotional_state)
                similarity = 1.0 / (1.0 + distance)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def step(self):
        """Advance time step."""
        self.time_step += 1
        self.record_state()
    
    def reset(self):
        """Reset network state."""
        self.time_step = 0
        self.emotion_history = []
        
        # Reset all agents to neutral
        for agent in self.agents.values():
            agent.set_emotion({'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0})
    
    def __len__(self) -> int:
        return len(self.agents)
    
    def __repr__(self) -> str:
        return (
            f"SocialNetwork("
            f"n_agents={len(self.agents)}, "
            f"n_connections={self.graph.number_of_edges()}, "
            f"time_step={self.time_step})"
        )


def create_network(
    network_type: str = 'small_world',
    n_nodes: int = 100,
    **kwargs
) -> SocialNetwork:
    """
    Create social network with specified topology.
    
    Args:
        network_type: Type of network ('small_world', 'scale_free', 'random', 'complete')
        n_nodes: Number of nodes
        **kwargs: Additional parameters for network generation
        
    Returns:
        SocialNetwork object
    """
    if network_type == 'small_world':
        k = kwargs.get('k', 6)
        p = kwargs.get('p', 0.3)
        graph = nx.watts_strogatz_graph(n_nodes, k, p)
        
    elif network_type == 'scale_free':
        m = kwargs.get('m', 3)
        graph = nx.barabasi_albert_graph(n_nodes, m)
        
    elif network_type == 'random':
        p = kwargs.get('p', 0.1)
        graph = nx.erdos_renyi_graph(n_nodes, p)
        
    elif network_type == 'complete':
        graph = nx.complete_graph(n_nodes)
        
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    network = SocialNetwork(graph)
    
    # Add agents
    for node in graph.nodes():
        agent = Agent(
            agent_id=str(node),
            initial_emotion={'valence': np.random.uniform(-0.2, 0.2),
                           'arousal': np.random.uniform(-0.2, 0.2)}
        )
        network.add_agent(agent)
    
    return network
