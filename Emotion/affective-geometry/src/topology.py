"""
Topological Data Analysis for persistent emotional structures.

Uses persistent homology and Mapper algorithm to identify
invariant emotional patterns across individuals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class TopologicalAnalysis:
    """
    Topological data analysis for emotional dynamics.
    
    Identifies persistent structures in emotional state space that
    are invariant across individuals and conditions.
    """
    
    def __init__(self):
        """Initialize topological analyzer."""
        self.persistence_diagrams = []
        self.barcodes = []
    
    def compute_persistent_homology(
        self,
        point_cloud: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None
    ) -> Dict:
        """
        Compute persistent homology of point cloud.
        
        Uses Ripser for efficient computation.
        
        Args:
            point_cloud: Array of shape (n_points, n_dimensions)
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge length for Rips complex
            
        Returns:
            Dictionary with persistence diagrams
        """
        try:
            from ripser import ripser
            from persim import plot_diagrams
            
            # Compute persistence
            result = ripser(
                point_cloud,
                maxdim=max_dimension,
                thresh=max_edge_length if max_edge_length else np.inf
            )
            
            self.persistence_diagrams = result['dgms']
            
            return {
                'diagrams': result['dgms'],
                'num_edges': result.get('num_edges', None),
                'cocycles': result.get('cocycles', None)
            }
            
        except ImportError:
            print("Warning: ripser not installed. Using simplified computation.")
            return self._compute_persistent_homology_simple(point_cloud, max_dimension)
    
    def _compute_persistent_homology_simple(
        self,
        point_cloud: np.ndarray,
        max_dimension: int = 1
    ) -> Dict:
        """
        Simplified persistent homology computation without ripser.
        
        Only computes H0 (connected components).
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Compute distance matrix
        distances = squareform(pdist(point_cloud))
        
        # Hierarchical clustering for H0
        linkage_matrix = linkage(distances, method='single')
        
        # Extract birth-death pairs
        n_points = len(point_cloud)
        births = np.zeros(n_points - 1)
        deaths = linkage_matrix[:, 2]
        
        # Create persistence diagram for H0
        h0_diagram = np.column_stack([births, deaths])
        
        self.persistence_diagrams = [h0_diagram]
        
        return {
            'diagrams': [h0_diagram],
            'num_edges': None,
            'cocycles': None
        }
    
    def plot_persistence_diagram(
        self,
        diagrams: Optional[List[np.ndarray]] = None,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot persistence diagrams.
        
        Args:
            diagrams: List of persistence diagrams (uses cached if None)
            figsize: Figure size
        """
        if diagrams is None:
            diagrams = self.persistence_diagrams
        
        if not diagrams:
            raise ValueError("No persistence diagrams available")
        
        try:
            from persim import plot_diagrams
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Persistence diagram
            plot_diagrams(diagrams, ax=axes[0])
            axes[0].set_title('Persistence Diagram', fontsize=12, fontweight='bold')
            
            # Barcode plot
            self._plot_barcode(diagrams, ax=axes[1])
            axes[1].set_title('Persistence Barcode', fontsize=12, fontweight='bold')
            
        except ImportError:
            # Fallback plotting
            fig, ax = plt.subplots(figsize=(8, 8))
            
            colors = ['red', 'blue', 'green', 'orange']
            labels = ['H0 (Components)', 'H1 (Loops)', 'H2 (Voids)', 'H3']
            
            for dim, diagram in enumerate(diagrams):
                if len(diagram) > 0:
                    # Filter out infinite death times
                    finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
                    
                    if len(finite_diagram) > 0:
                        ax.scatter(
                            finite_diagram[:, 0],
                            finite_diagram[:, 1],
                            c=colors[dim % len(colors)],
                            label=labels[dim],
                            alpha=0.6,
                            s=50
                        )
            
            # Diagonal line
            max_val = max([d[:, 1].max() for d in diagrams if len(d) > 0])
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            ax.set_xlabel('Birth', fontsize=11)
            ax.set_ylabel('Death', fontsize=11)
            ax.set_title('Persistence Diagram', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_barcode(self, diagrams: List[np.ndarray], ax):
        """Plot persistence barcode."""
        colors = ['red', 'blue', 'green', 'orange']
        y_pos = 0
        
        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                # Sort by birth time
                sorted_diagram = diagram[diagram[:, 0].argsort()]
                
                for birth, death in sorted_diagram:
                    if np.isfinite(death):
                        ax.plot(
                            [birth, death],
                            [y_pos, y_pos],
                            c=colors[dim % len(colors)],
                            linewidth=2,
                            alpha=0.7
                        )
                        y_pos += 1
        
        ax.set_xlabel('Filtration Value', fontsize=11)
        ax.set_ylabel('Features', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
    
    def compute_persistence_landscape(
        self,
        diagram: np.ndarray,
        num_landscapes: int = 5,
        resolution: int = 100
    ) -> np.ndarray:
        """
        Compute persistence landscape for vectorization.
        
        Args:
            diagram: Persistence diagram
            num_landscapes: Number of landscape functions
            resolution: Number of points to sample
            
        Returns:
            Landscape array of shape (num_landscapes, resolution)
        """
        # Filter finite points
        finite_diagram = diagram[np.isfinite(diagram).all(axis=1)]
        
        if len(finite_diagram) == 0:
            return np.zeros((num_landscapes, resolution))
        
        # Create grid
        min_val = finite_diagram[:, 0].min()
        max_val = finite_diagram[:, 1].max()
        grid = np.linspace(min_val, max_val, resolution)
        
        # Compute landscape functions
        landscapes = np.zeros((num_landscapes, resolution))
        
        for i, t in enumerate(grid):
            # Compute lambda values for this t
            lambda_values = []
            
            for birth, death in finite_diagram:
                if birth <= t <= death:
                    lambda_val = min(t - birth, death - t)
                    lambda_values.append(lambda_val)
            
            # Sort and take top k
            lambda_values = sorted(lambda_values, reverse=True)
            
            for k in range(min(num_landscapes, len(lambda_values))):
                landscapes[k, i] = lambda_values[k]
        
        return landscapes
    
    def compute_bottleneck_distance(
        self,
        diagram1: np.ndarray,
        diagram2: np.ndarray
    ) -> float:
        """
        Compute bottleneck distance between two persistence diagrams.
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            
        Returns:
            Bottleneck distance
        """
        try:
            from persim import bottleneck
            return bottleneck(diagram1, diagram2)
        except ImportError:
            # Simplified approximation
            return self._bottleneck_distance_simple(diagram1, diagram2)
    
    def _bottleneck_distance_simple(
        self,
        diagram1: np.ndarray,
        diagram2: np.ndarray
    ) -> float:
        """Simplified bottleneck distance approximation."""
        # Filter finite points
        d1 = diagram1[np.isfinite(diagram1).all(axis=1)]
        d2 = diagram2[np.isfinite(diagram2).all(axis=1)]
        
        if len(d1) == 0 or len(d2) == 0:
            return 0.0
        
        # Compute persistence for each point
        pers1 = d1[:, 1] - d1[:, 0]
        pers2 = d2[:, 1] - d2[:, 0]
        
        # Approximate as max difference in persistence
        return abs(pers1.max() - pers2.max())
    
    def mapper_algorithm(
        self,
        point_cloud: np.ndarray,
        filter_function: Optional[np.ndarray] = None,
        num_intervals: int = 10,
        overlap: float = 0.3,
        clustering_method: str = 'dbscan',
        eps: float = 0.5
    ) -> Dict:
        """
        Apply Mapper algorithm for topological visualization.
        
        Args:
            point_cloud: Data points
            filter_function: 1D filter values (uses PCA if None)
            num_intervals: Number of overlapping intervals
            overlap: Overlap percentage between intervals
            clustering_method: Clustering algorithm
            eps: DBSCAN epsilon parameter
            
        Returns:
            Dictionary with nodes, edges, and metadata
        """
        n_points = len(point_cloud)
        
        # Compute filter function if not provided
        if filter_function is None:
            pca = PCA(n_components=1)
            filter_function = pca.fit_transform(point_cloud).flatten()
        
        # Create overlapping intervals
        f_min, f_max = filter_function.min(), filter_function.max()
        interval_length = (f_max - f_min) / num_intervals
        overlap_length = interval_length * overlap
        
        nodes = []
        node_points = []
        
        # Process each interval
        for i in range(num_intervals):
            interval_start = f_min + i * interval_length - overlap_length
            interval_end = interval_start + interval_length + 2 * overlap_length
            
            # Get points in interval
            mask = (filter_function >= interval_start) & (filter_function <= interval_end)
            interval_points = point_cloud[mask]
            interval_indices = np.where(mask)[0]
            
            if len(interval_points) == 0:
                continue
            
            # Cluster points in interval
            if clustering_method == 'dbscan':
                clustering = DBSCAN(eps=eps, min_samples=2)
                labels = clustering.fit_predict(interval_points)
            else:
                labels = np.zeros(len(interval_points), dtype=int)
            
            # Create nodes for each cluster
            for label in set(labels):
                if label == -1:  # Noise
                    continue
                
                cluster_mask = labels == label
                cluster_indices = interval_indices[cluster_mask]
                
                nodes.append({
                    'id': len(nodes),
                    'interval': i,
                    'cluster': label,
                    'size': len(cluster_indices),
                    'center': interval_points[cluster_mask].mean(axis=0)
                })
                node_points.append(set(cluster_indices))
        
        # Find edges (nodes with shared points)
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                shared_points = node_points[i] & node_points[j]
                if len(shared_points) > 0:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': len(shared_points)
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_points': node_points,
            'filter_function': filter_function
        }
    
    def plot_mapper_graph(
        self,
        mapper_result: Dict,
        figsize: Tuple[int, int] = (12, 8),
        node_color_by: str = 'size'
    ):
        """
        Visualize Mapper graph.
        
        Args:
            mapper_result: Output from mapper_algorithm
            figsize: Figure size
            node_color_by: Color nodes by 'size' or 'interval'
        """
        try:
            import networkx as nx
            
            # Create graph
            G = nx.Graph()
            
            nodes = mapper_result['nodes']
            edges = mapper_result['edges']
            
            # Add nodes
            for node in nodes:
                G.add_node(node['id'], **node)
            
            # Add edges
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
            
            # Layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Determine node colors
            if node_color_by == 'size':
                node_colors = [node['size'] for node in nodes]
                cmap = 'viridis'
            else:
                node_colors = [node['interval'] for node in nodes]
                cmap = 'tab10'
            
            # Draw
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=[node['size'] * 50 for node in nodes],
                cmap=cmap,
                alpha=0.7,
                ax=ax
            )
            
            ax.set_title('Mapper Graph of Emotional Landscape', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            return fig, ax
            
        except ImportError:
            print("NetworkX not installed. Cannot plot Mapper graph.")
            return None, None


def compare_topological_features(
    trajectories1: List[np.ndarray],
    trajectories2: List[np.ndarray],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2"
) -> Dict:
    """
    Compare topological features between two groups.
    
    Args:
        trajectories1: List of trajectories for group 1
        trajectories2: List of trajectories for group 2
        group1_name: Name for group 1
        group2_name: Name for group 2
        
    Returns:
        Dictionary with comparison statistics
    """
    analyzer = TopologicalAnalysis()
    
    # Compute persistence for both groups
    group1_diagrams = []
    for traj in trajectories1:
        result = analyzer.compute_persistent_homology(traj, max_dimension=1)
        group1_diagrams.append(result['diagrams'])
    
    group2_diagrams = []
    for traj in trajectories2:
        result = analyzer.compute_persistent_homology(traj, max_dimension=1)
        group2_diagrams.append(result['diagrams'])
    
    # Compute average persistence
    def avg_persistence(diagrams_list):
        all_pers = []
        for diagrams in diagrams_list:
            for diagram in diagrams:
                finite = diagram[np.isfinite(diagram).all(axis=1)]
                if len(finite) > 0:
                    pers = finite[:, 1] - finite[:, 0]
                    all_pers.extend(pers)
        return np.mean(all_pers) if all_pers else 0.0
    
    g1_avg_pers = avg_persistence(group1_diagrams)
    g2_avg_pers = avg_persistence(group2_diagrams)
    
    print(f"\n{'='*60}")
    print(f"Topological Feature Comparison")
    print(f"{'='*60}\n")
    print(f"{group1_name} average persistence: {g1_avg_pers:.4f}")
    print(f"{group2_name} average persistence: {g2_avg_pers:.4f}")
    
    return {
        'group1_persistence': g1_avg_pers,
        'group2_persistence': g2_avg_pers,
        'group1_diagrams': group1_diagrams,
        'group2_diagrams': group2_diagrams
    }
