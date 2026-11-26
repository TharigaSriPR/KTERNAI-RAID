"""
Classification package for RAID Identification Agent.
Provides category schemas, LLM providers (Gemini, Mock), and the RAIDClassifier engine.
"""

from .schemas import (
    RAIDCategory,
    RAIDExtractionResult,
    ClassificationStatus,
    ClassificationErrorCategory,
    EvidenceStatus,
    ClassificationOutcome
)
from .classifier import RAIDClassifier
from .llm_client import get_llm_provider
from .providers.base_provider import BaseLLMProvider
from .providers.gemini_provider import GeminiProvider
from .providers.mock_provider import MockLLMProvider

# Backward-compatibility alias
LLMClassifier = RAIDClassifier

__all__ = [
    "RAIDCategory",
    "RAIDExtractionResult",
    "ClassificationStatus",
    "ClassificationErrorCategory",
    "EvidenceStatus",
    "ClassificationOutcome",
    "RAIDClassifier",
    "LLMClassifier",
    "BaseLLMProvider",
    "GeminiProvider",
    "MockLLMProvider",
    "get_llm_provider"
]
