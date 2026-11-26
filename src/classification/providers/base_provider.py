"""
Abstract Base Provider interface for LLM backends.
Enables pluggable model providers (Google Gemini, OpenAI, Claude, Mock).
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM completion and structured generation.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Executes a prompt against the model backend and returns the raw string completion.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical name of the LLM provider."""
        pass
