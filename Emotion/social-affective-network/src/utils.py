"""Utility functions for SARN."""

import numpy as np
import networkx as nx
from typing import Dict


def generate_random_emotion() -> Dict:
    """Generate random emotional state."""
    return {
        'valence': np.random.uniform(-1, 1),
        'arousal': np.random.uniform(-1, 1),
        'dominance': np.random.uniform(-1, 1)
    }


def emotion_distance(emotion1: Dict, emotion2: Dict) -> float:
    """Compute distance between two emotional states."""
    return np.sqrt(
        (emotion1.get('valence', 0) - emotion2.get('valence', 0)) ** 2 +
        (emotion1.get('arousal', 0) - emotion2.get('arousal', 0)) ** 2 +
        (emotion1.get('dominance', 0) - emotion2.get('dominance', 0)) ** 2
    )


def visualize_network_emotions(network, figsize=(12, 10)):
    """Visualize network with emotion-colored nodes."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get positions
    pos = nx.spring_layout(network.graph, seed=42)
    
    # Color nodes by valence
    node_colors = []
    for node_id in network.graph.nodes():
        agent = network.get_agent(node_id)
        if agent:
            valence = agent.emotional_state.valence
            # Map valence to color (red = negative, green = positive)
            if valence > 0:
                color = (0, valence, 0)  # Green
            else:
                color = (-valence, 0, 0)  # Red
            node_colors.append(color)
        else:
            node_colors.append((0.5, 0.5, 0.5))
    
    # Draw network
    nx.draw_networkx_nodes(network.graph, pos, node_color=node_colors, 
                          node_size=300, ax=ax)
    nx.draw_networkx_edges(network.graph, pos, alpha=0.3, ax=ax)
    
    ax.set_title('Social Network Emotional States', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=(0, 1, 0), label='Positive'),
        Rectangle((0, 0), 1, 1, fc=(1, 0, 0), label='Negative'),
        Rectangle((0, 0), 1, 1, fc=(0.5, 0.5, 0.5), label='Neutral')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax
