"""
Machine Learning & Model Explainability Module.
"""

from .train import train_and_compare_models
from .evaluate import evaluate_binary_classifier
from .explain import ModelExplainer

__all__ = [
    "train_and_compare_models",
    "evaluate_binary_classifier",
    "ModelExplainer",
]
