"""
Google Gemini Provider implementation using the official google-genai SDK.
Handles:
- Authentication via GEMINI_API_KEY / GOOGLE_API_KEY environment variables
- Exponential backoff retry logic for transient rate-limits and timeouts
- Graceful error reporting without credential leakage
"""

import os
import time
from typing import Optional

from .base_provider import BaseLLMProvider
from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
except ImportError:
    genai = None
    types = None
    genai_errors = None


class GeminiProvider(BaseLLMProvider):
    """
    Provider adapter for Google Gemini models via google-genai SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = LLM_REQUEST_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
        retry_delay: float = LLM_RETRY_DELAY
    ):
        super().__init__(model_name or LLM_MODEL)
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "")).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key or not self.api_key.strip():
            logger.warning("GeminiProvider initialized without API key. Live calls will fail until GEMINI_API_KEY is provided.")

        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        """Initializes the google.genai.Client instance on demand."""
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is not installed. Please install it via: pip install google-genai"
            )
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Calls the Gemini API with retry logic and exponential backoff for transient errors.
        """
        client = self._get_client()

        config = types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for factual, deterministic extraction
            response_mime_type="application/json"
        )
        if system_instruction:
            config.system_instruction = system_instruction

        attempt = 0
        last_exception = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.debug("Dispatching request to Gemini (%s) - attempt %d/%d", self.model_name, attempt, self.max_retries)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                if response and response.text:
                    return response.text.strip()
                raise ValueError("Empty or null response text received from Gemini API.")

            except Exception as exc:
                last_exception = exc
                error_str = str(exc)

                # Check for permanent errors that should NOT be retried
                if any(k in error_str for k in ("API_KEY_INVALID", "401", "403", "PERMISSION_DENIED", "NOT_FOUND")):
                    logger.error("Permanent authentication/permission error from Gemini API: %s", error_str)
                    raise

                if attempt >= self.max_retries:
                    logger.error("Exceeded max retries (%d) for Gemini API: %s", self.max_retries, error_str)
                    break

                backoff = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error calling Gemini API on attempt %d: %s. Retrying in %.2f seconds...",
                    attempt, error_str, backoff
                )
                time.sleep(backoff)

        raise RuntimeError(f"Gemini API request failed after {self.max_retries} attempts: {last_exception}")
