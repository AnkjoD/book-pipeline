"""Compatibility for the original learning exercise's misspelled module name."""

from .evaluation import evaluate_query, recall_at_k, reciprocal_rank

__all__ = ["evaluate_query", "recall_at_k", "reciprocal_rank"]
