"""
Deep Learning Models for Recommendations
=========================================

Modern neural network-based recommendation models including:
- Neural Collaborative Filtering (NCF)
- Wide & Deep Learning
- Two-Tower Neural Networks
"""

from .ncf_model import NeuralCollaborativeFiltering
from .wide_and_deep import WideAndDeepModel
from .two_tower import TwoTowerModel

__all__ = [
    'NeuralCollaborativeFiltering',
    'WideAndDeepModel',
    'TwoTowerModel'
]
