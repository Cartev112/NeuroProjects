"""Model comparison tools for testing competing emotion theories."""

import numpy as np
from typing import List, Dict
from .theories import TheoryType, JamesLangeTheory, CannonBardTheory, SchachterSingerTheory, ConstructionistTheory


class ComparisonMetrics:
    """Metrics for comparing emotion theories."""
    
    @staticmethod
    def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """Compute Bayesian Information Criterion."""
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    @staticmethod
    def compute_aic(log_likelihood: float, n_params: int) -> float:
        """Compute Akaike Information Criterion."""
        return -2 * log_likelihood + 2 * n_params
    
    @staticmethod
    def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error."""
        return np.mean((predictions - targets) ** 2)


class ModelComparison:
    """Compare different emotion theories on empirical data."""
    
    def __init__(self, theories: List[str] = None):
        """
        Initialize model comparison.
        
        Args:
            theories: List of theory names to compare
        """
        if theories is None:
            theories = ['james_lange', 'cannon_bard', 'schachter_singer', 'constructionist']
        
        self.theories = {}
        for theory_name in theories:
            if theory_name == 'james_lange':
                self.theories[theory_name] = JamesLangeTheory()
            elif theory_name == 'cannon_bard':
                self.theories[theory_name] = CannonBardTheory()
            elif theory_name == 'schachter_singer':
                self.theories[theory_name] = SchachterSingerTheory()
            elif theory_name == 'constructionist':
                self.theories[theory_name] = ConstructionistTheory()
        
        self.results = {}
    
    def fit_all_models(self, data: Dict):
        """Fit all theories to data."""
        for name, theory in self.theories.items():
            # Simplified fitting
            self.results[name] = {
                'fitted': True,
                'log_likelihood': np.random.randn(),  # Placeholder
                'n_params': 10
            }
    
    def compare_models(self, metric: str = 'bic') -> Dict:
        """Compare models using specified metric."""
        comparison = {}
        
        for name, result in self.results.items():
            if metric == 'bic':
                comparison[name] = ComparisonMetrics.compute_bic(
                    result['log_likelihood'],
                    result['n_params'],
                    100  # n_samples
                )
            elif metric == 'aic':
                comparison[name] = ComparisonMetrics.compute_aic(
                    result['log_likelihood'],
                    result['n_params']
                )
        
        return comparison
    
    def get_best_model(self) -> str:
        """Get best fitting model."""
        comparison = self.compare_models()
        return min(comparison, key=comparison.get)
