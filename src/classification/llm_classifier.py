"""
Backwards-compatible wrapper module forwarding to the enhanced RAIDClassifier.
"""
from .classifier import RAIDClassifier

# Direct alias for backward-compatibility
LLMClassifier = RAIDClassifier

__all__ = ["LLMClassifier", "RAIDClassifier"]
