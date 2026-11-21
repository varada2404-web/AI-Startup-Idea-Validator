"""Gemini LLM provider built on the official Google Gen AI Python SDK."""

from __future__ import annotations

import os
from typing import Type

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 30.0


class GeminiProviderError(Exception):
    """Base exception for Gemini provider failures."""


class GeminiConfigError(GeminiProviderError):
    """Raised when Gemini configuration (such as API key or settings) is missing or invalid."""


class GeminiTimeoutError(GeminiProviderError):
    """Raised when a Gemini API request exceeds the configured timeout."""


class GeminiUnavailableError(GeminiProviderError):
    """Raised when the Gemini API is unavailable or rejects a request."""


class GeminiRateLimitError(GeminiUnavailableError):
    """Raised when the Gemini API rate limit or quota is exceeded."""


class GeminiInvalidResponseError(GeminiProviderError):
    """Raised when Gemini returns an empty or unreadable response."""


class GeminiProvider:
    """Thin wrapper around the official Google Gen AI SDK for structured generation."""

    provider_name = "gemini"

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
            else os.getenv("GEMINI_API_KEY", "")
        )
        api_key = raw_key.strip()
        if not api_key:
            raise GeminiConfigError("GEMINI_API_KEY is not configured.")
        return api_key

    def _get_model(self) -> str:
        raw_model = (
            self._model_override
            if self._model_override is not None
            else os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        )
        model = raw_model.strip()
        return model or DEFAULT_GEMINI_MODEL

    def get_model(self) -> str:
        """Return the configured Gemini model name."""
        return self._get_model()

    def _get_timeout_ms(self) -> int:
        if self._timeout_override is not None:
            timeout_seconds = self._timeout_override
        else:
            raw_timeout = os.getenv("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
            try:
                timeout_seconds = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SECONDS
            except ValueError as exc:
                raise GeminiConfigError("GEMINI_TIMEOUT_SECONDS must be a positive number.") from exc

        if timeout_seconds <= 0:
            raise GeminiConfigError("GEMINI_TIMEOUT_SECONDS must be greater than zero.")
        return max(1, int(timeout_seconds * 1000))

    def _create_client(self) -> genai.Client:
        api_key = self._get_api_key()
        timeout_ms = self._get_timeout_ms()
        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_schema: Type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Send a prompt to Gemini and return the raw structured JSON text response."""
        client = self._create_client()
        model = self._get_model()

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise GeminiTimeoutError("Gemini API request timed out.") from exc
        except genai_errors.APIError as exc:
            message_lower = str(exc).lower()
            if exc.code in (408, 504) or "timeout" in message_lower or "deadline" in message_lower:
                raise GeminiTimeoutError("Gemini API request timed out.") from exc
            if (
                exc.code == 429
                or "rate limit" in message_lower
                or "resource_exhausted" in message_lower
                or "quota" in message_lower
            ):
                raise GeminiRateLimitError("Gemini API rate limit exceeded.") from exc
            raise GeminiUnavailableError("Gemini API request failed or service is unavailable.") from exc
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            raise GeminiUnavailableError("Unable to connect to the Gemini API.") from exc
        except GeminiProviderError:
            raise
        except Exception as exc:
            raise GeminiUnavailableError("Unexpected error communicating with Gemini API.") from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiInvalidResponseError("Gemini returned an empty response.")

        return text.strip()
