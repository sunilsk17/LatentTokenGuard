"""
evaluation/__init__.py
Evaluation utilities for LatentTokenGuard.
"""

from evaluation.metrics import evaluate, EvaluationResult

__all__ = ["evaluate", "EvaluationResult"]
