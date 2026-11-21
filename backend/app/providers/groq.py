"""Groq LLM provider built on the official Groq Python SDK (automatic fallback provider)."""

from __future__ import annotations

import os
from typing import Type

import groq
import httpx
from groq import Groq
from pydantic import BaseModel

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_TIMEOUT_SECONDS = 30.0


class GroqProviderError(Exception):
    """Base exception for Groq provider failures."""


class GroqConfigError(GroqProviderError):
    """Raised when Groq configuration (such as API key or settings) is missing or invalid."""


class GroqAuthError(GroqProviderError):
    """Raised when Groq rejects authentication credentials."""


class GroqRateLimitError(GroqProviderError):
    """Raised when Groq API rate limits are exceeded."""


class GroqTimeoutError(GroqProviderError):
    """Raised when a Groq API request exceeds the configured timeout."""


class GroqUnavailableError(GroqProviderError):
    """Raised when the Groq API is unavailable or returns a server/network error."""


class GroqInvalidResponseError(GroqProviderError):
    """Raised when Groq returns an empty or unreadable response."""


class GroqProvider:
    """Thin wrapper around the official Groq SDK for structured JSON generation."""

    provider_name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._api_key_override = api_key
        self._model_override = model
        self._timeout_override = timeout_seconds

    def _get_api_key(self) -> str:
        raw_key = (
            self._api_key_override
            if self._api_key_override is not None
            else os.getenv("GROQ_API_KEY", "")
        )
        api_key = raw_key.strip()
        if not api_key:
            raise GroqConfigError("GROQ_API_KEY is not configured.")
        return api_key

    def _get_model(self) -> str:
        raw_model = (
            self._model_override
            if self._model_override is not None
            else os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        )
        model = raw_model.strip()
        return model or DEFAULT_GROQ_MODEL

    def get_model(self) -> str:
        """Return the configured Groq model name."""
        return self._get_model()

    def _get_timeout_seconds(self) -> float:
        if self._timeout_override is not None:
            timeout_seconds = self._timeout_override
        else:
            raw_timeout = os.getenv(
                "GROQ_TIMEOUT_SECONDS", str(DEFAULT_GROQ_TIMEOUT_SECONDS)
            ).strip()
            try:
                timeout_seconds = (
                    float(raw_timeout) if raw_timeout else DEFAULT_GROQ_TIMEOUT_SECONDS
                )
            except ValueError as exc:
                raise GroqConfigError(
                    "GROQ_TIMEOUT_SECONDS must be a positive number."
                ) from exc

        if timeout_seconds <= 0:
            raise GroqConfigError("GROQ_TIMEOUT_SECONDS must be greater than zero.")
        return timeout_seconds

    def _create_client(self) -> Groq:
        api_key = self._get_api_key()
        timeout_seconds = self._get_timeout_seconds()
        return Groq(api_key=api_key, timeout=timeout_seconds)

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_schema: Type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Send a prompt to Groq with JSON object output mode and return the raw JSON text."""
        client = self._create_client()
        model = self._get_model()

        messages: list[dict[str, str]] = []
        if system_instruction:
            sys_content = system_instruction
            if "json" not in sys_content.lower():
                sys_content = f"{sys_content}\n\nReturn ONLY a valid JSON object."
            messages.append({"role": "system", "content": sys_content})
        else:
            messages.append({"role": "system", "content": "Return ONLY a valid JSON object."})

        messages.append({"role": "user", "content": prompt})

        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except (TimeoutError, httpx.TimeoutException, groq.APITimeoutError) as exc:
            raise GroqTimeoutError("Groq API request timed out.") from exc
        except groq.RateLimitError as exc:
            raise GroqRateLimitError("Groq API rate limit exceeded.") from exc
        except (groq.AuthenticationError, groq.PermissionDeniedError) as exc:
            raise GroqAuthError("Groq API authentication failed.") from exc
        except groq.APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code in (408, 504):
                raise GroqTimeoutError("Groq API request timed out.") from exc
            if status_code == 429:
                raise GroqRateLimitError("Groq API rate limit exceeded.") from exc
            if status_code in (401, 403):
                raise GroqAuthError("Groq API authentication failed.") from exc
            raise GroqUnavailableError(
                "Groq API request failed or service is unavailable."
            ) from exc
        except (groq.APIConnectionError, httpx.HTTPError, ConnectionError, OSError) as exc:
            raise GroqUnavailableError("Unable to connect to the Groq API.") from exc
        except GroqProviderError:
            raise
        except Exception as exc:
            raise GroqUnavailableError(
                "Unexpected error communicating with Groq API."
            ) from exc

        choices = getattr(completion, "choices", None)
        if not choices:
            raise GroqInvalidResponseError("Groq returned no completion choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise GroqInvalidResponseError("Groq returned an empty response.")

        return content.strip()
