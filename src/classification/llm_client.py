"""
LLM Client Factory.
Instantiates the configured provider (Gemini or Mock) based on application configuration.
"""

from typing import Optional
from config import LLM_PROVIDER
from .providers.base_provider import BaseLLMProvider
from .providers.gemini_provider import GeminiProvider
from .providers.mock_provider import MockLLMProvider
from .providers.openrouter_provider import OpenRouterProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseLLMProvider:
    """
    Factory function returning the configured BaseLLMProvider instance.
    Supports OpenRouter, Gemini, and Mock.
    """
    chosen = (provider_name or LLM_PROVIDER).lower().strip()

    if chosen == "openrouter":
        logger.debug("Instantiating OpenRouterProvider (model=%s)", model_name or "default")
        return OpenRouterProvider(api_key=api_key, model_name=model_name)
    elif chosen == "gemini":
        logger.debug("Instantiating GeminiProvider (model=%s)", model_name or "default")
        return GeminiProvider(api_key=api_key, model_name=model_name)
    elif chosen == "mock":
        logger.debug("Instantiating MockLLMProvider")
        return MockLLMProvider()
    else:
        logger.warning("Unrecognized LLM provider '%s'. Falling back to OpenRouterProvider", chosen)
        return OpenRouterProvider(api_key=api_key, model_name=model_name)
