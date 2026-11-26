"""
OpenRouter LLM Provider implementation for RAID Identification Agent POC.
Enables accessing OpenRouter models (such as google/gemma-4-26b-a4b-it:free) via standard HTTP REST API.
Handles:
- Authentication via OPENROUTER_API_KEY environment variable / .env
- Exponential backoff retry logic for transient rate-limits (429) and network timeouts
- Graceful error reporting without credential exposure
"""

import os
import time
from typing import Optional, Dict, Any
import requests

from .base_provider import BaseLLMProvider
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_API_URL,
    LLM_REQUEST_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenRouterRateLimitError(RuntimeError):
    """Raised when OpenRouter returns HTTP 429 to abort immediately without retrying."""
    pass


class OpenRouterProvider(BaseLLMProvider):
    """
    Provider adapter for OpenRouter API supporting Gemma and other models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = LLM_REQUEST_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
        retry_delay: float = LLM_RETRY_DELAY
    ):
        eff_model = model_name if model_name is not None else (OPENROUTER_MODEL or os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"))
        super().__init__(eff_model)
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")).strip()

        if api_url is not None:
            self.api_url = api_url.strip()
        else:
            self.api_url = (OPENROUTER_API_URL or os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")).strip()

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            logger.warning("OpenRouterProvider initialized without API key. Live calls will fail until OPENROUTER_API_KEY is configured.")

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def is_configured(self) -> bool:
        """Checks whether an API key has been configured without making live network calls."""
        return bool(self.api_key and len(self.api_key) >= 10)

    def _sanitize_error(self, err_msg: str) -> str:
        """Ensures API keys are never exposed in log output or exceptions."""
        import re
        sanitized = re.sub(r"sk-[0-9A-Za-z-_]{15,}", "[REDACTED_API_KEY]", str(err_msg))
        sanitized = re.sub(r"Bearer\s+[0-9A-Za-z-_.]+", "Bearer [REDACTED_TOKEN]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Dispatches chat completion request to OpenRouter with retry logic for transient errors.
        """
        if not self.is_configured():
            raise ValueError("OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in your .env file.")

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ktern.ai",
            "X-Title": "KTern.AI RAID Identification Agent"
        }

        attempt = 0
        last_exception = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.debug(
                    "Dispatching request to OpenRouter (%s) - attempt %d/%d",
                    self.model_name, attempt, self.max_retries
                )

                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

                # Check for permanent authentication/permission errors that should NOT be retried
                if resp.status_code in (401, 403):
                    safe_body = self._sanitize_error(resp.text[:300])
                    logger.error("Authentication failure from OpenRouter API (HTTP %d): %s", resp.status_code, safe_body)
                    raise PermissionError(f"OpenRouter authentication failure (HTTP {resp.status_code}): {safe_body}")

                if resp.status_code == 400:
                    safe_body = self._sanitize_error(resp.text[:300])
                    logger.error("Bad Request to OpenRouter API (HTTP 400): %s", safe_body)
                    raise ValueError(f"OpenRouter Bad Request (HTTP 400): {safe_body}")

                # Rate limit (HTTP 429): immediately abort without retrying against rate-limited endpoints
                if resp.status_code == 429:
                    safe_body = self._sanitize_error(resp.text[:300])
                    logger.warning("Upstream rate limit (HTTP 429) from OpenRouter API: %s", safe_body)
                    raise OpenRouterRateLimitError(f"OpenRouter rate limit exceeded (HTTP 429): {safe_body}")

                # Transient server errors: 500, 502, 503, 504
                if resp.status_code in (500, 502, 503, 504):
                    safe_body = self._sanitize_error(resp.text[:200])
                    raise requests.exceptions.RequestException(
                        f"Transient OpenRouter error HTTP {resp.status_code}: {safe_body}"
                    )

                resp.raise_for_status()

                data = resp.json()
                if not data or not isinstance(data, dict):
                    raise ValueError("Non-dictionary response received from OpenRouter API.")

                choices = data.get("choices")
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"OpenRouter response missing choices array: {data}")

                first_choice = choices[0]
                message = first_choice.get("message") or {}
                content = message.get("content")

                if content and str(content).strip():
                    return str(content).strip()

                raise ValueError("Empty or null content received in OpenRouter completion.")

            except (PermissionError, ValueError, OpenRouterRateLimitError) as permanent_err:
                # Permanent or non-retryable rate limit errors: do not retry
                raise permanent_err

            except Exception as exc:
                last_exception = exc
                safe_err_str = self._sanitize_error(str(exc))

                if attempt >= self.max_retries:
                    logger.error("Exceeded max retries (%d) for OpenRouter API: %s", self.max_retries, safe_err_str)
                    break

                backoff = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error calling OpenRouter on attempt %d: %s. Retrying in %.2f seconds...",
                    attempt, safe_err_str, backoff
                )
                time.sleep(backoff)

        raise RuntimeError(f"OpenRouter API request failed after {self.max_retries} attempts: {self._sanitize_error(str(last_exception))}")
