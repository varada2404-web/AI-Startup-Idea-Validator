"""Reusable LLM provider-selection and automatic fallback service (Gemini -> Groq)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from app.providers.gemini import (
    GeminiConfigError,
    GeminiInvalidResponseError,
    GeminiProvider,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from app.providers.groq import (
    GroqAuthError,
    GroqConfigError,
    GroqInvalidResponseError,
    GroqProvider,
    GroqProviderError,
    GroqRateLimitError,
    GroqTimeoutError,
    GroqUnavailableError,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


class StructuredLLMProvider(Protocol):
    """Minimal protocol implemented by GeminiProvider and GroqProvider."""

    provider_name: str

    def get_model(self) -> str: ...

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_schema: Type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> str: ...


class LLMExecutionMetadata(BaseModel):
    """Safe, secret-free metadata describing which LLM provider fulfilled a request."""

    provider: str = Field(..., description="Identifier of the provider that succeeded ('gemini' or 'groq').")
    model: str = Field(..., description="Model name used by the successful provider.")
    fallback_used: bool = Field(
        default=False,
        description="Whether automatic fallback to the secondary provider was triggered.",
    )
    attempted_providers: list[str] = Field(
        default_factory=list,
        description="Ordered list of providers attempted for this request.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Safe error classification explaining why fallback was triggered, if any.",
    )


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[SchemaT]):
    """Validated Pydantic schema output paired with safe execution metadata."""

    data: SchemaT
    metadata: LLMExecutionMetadata


class LLMRouterError(Exception):
    """Base exception for LLM provider routing and fallback failures."""


class LLMInvalidOutputError(LLMRouterError):
    """Raised when a model response is not valid JSON or not a JSON object."""

    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class LLMOutputValidationError(LLMRouterError):
    """Raised when parsed model JSON fails Pydantic schema validation."""

    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class LLMAllProvidersFailedError(LLMRouterError):
    """Raised when both primary (Gemini) and fallback (Groq) providers fail."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, str]],
        http_status: int = 503,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.http_status = http_status


def strip_markdown_fences(raw_text: str) -> str:
    """Remove surrounding markdown code fences if a model wraps JSON output in them."""
    cleaned = raw_text.strip()
    match = _CODE_FENCE_RE.match(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def parse_json_object(raw_text: str, *, schema_name: str = "schema") -> dict[str, Any]:
    """Parse raw model text into a JSON object dictionary."""
    cleaned = strip_markdown_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMInvalidOutputError(
            "Model response could not be parsed as valid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMInvalidOutputError(
            f"Model response JSON must be an object matching {schema_name}."
        )

    return parsed


def _classify_error(exc: Exception) -> str:
    """Return a safe, secret-free error type string for metadata and error reporting."""
    if isinstance(exc, (GeminiTimeoutError, GroqTimeoutError)):
        return "timeout"
    if isinstance(exc, (GeminiRateLimitError, GroqRateLimitError)):
        return "rate_limit"
    if isinstance(exc, (GeminiConfigError, GroqConfigError)):
        return "config_error"
    if isinstance(exc, GroqAuthError):
        return "auth_error"
    if isinstance(exc, (GeminiUnavailableError, GroqUnavailableError)):
        return "unavailable"
    if isinstance(exc, (GeminiInvalidResponseError, GroqInvalidResponseError, LLMInvalidOutputError)):
        return "invalid_output"
    if isinstance(exc, LLMOutputValidationError):
        return "validation_error"
    return "provider_error"


def _is_retryable_primary_error(exc: Exception, *, fallback_on_invalid_output: bool) -> bool:
    """Determine whether a failure from Gemini should trigger Groq fallback.

    Unrelated programming errors (TypeError, KeyError, AttributeError, ValueError, etc.)
    return False and propagate immediately without triggering fallback.
    """
    if isinstance(
        exc,
        (
            GeminiTimeoutError,
            GeminiRateLimitError,
            GeminiUnavailableError,
            GeminiConfigError,
        ),
    ):
        return True
    if fallback_on_invalid_output and isinstance(
        exc,
        (
            GeminiInvalidResponseError,
            LLMInvalidOutputError,
            LLMOutputValidationError,
        ),
    ):
        return True
    return False


def _determine_combined_http_status(attempts: list[dict[str, str]]) -> int:
    """Choose an appropriate HTTP status code when all providers fail."""
    error_types = [a["error_type"] for a in attempts]
    if all(et == "timeout" for et in error_types):
        return 504
    if any(et in {"invalid_output", "validation_error"} for et in error_types) and error_types[-1] in {
        "invalid_output",
        "validation_error",
    }:
        return 502
    return 503


class LLMFallbackService:
    """Executes structured LLM requests using Gemini first, with automatic Groq fallback."""

    def __init__(
        self,
        primary_provider: StructuredLLMProvider | None = None,
        fallback_provider: StructuredLLMProvider | None = None,
        *,
        enable_fallback: bool = True,
        fallback_on_invalid_output: bool = True,
    ) -> None:
        self._primary: StructuredLLMProvider = primary_provider or GeminiProvider()
        if enable_fallback:
            self._fallback: StructuredLLMProvider | None = (
                fallback_provider if fallback_provider is not None else GroqProvider()
            )
        else:
            self._fallback = None
        self._fallback_on_invalid_output = fallback_on_invalid_output

    def _invoke_and_validate(
        self,
        provider: StructuredLLMProvider,
        *,
        prompt: str,
        system_instruction: str | None,
        response_schema: Type[SchemaT],
        temperature: float,
    ) -> SchemaT:
        raw_response = provider.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=response_schema,
            temperature=temperature,
        )
        parsed = parse_json_object(raw_response, schema_name=response_schema.__name__)
        try:
            return response_schema.model_validate(parsed)
        except ValidationError as exc:
            raise LLMOutputValidationError(
                f"Model response failed {response_schema.__name__} schema validation."
            ) from exc

    def generate_structured(
        self,
        *,
        prompt: str,
        system_instruction: str | None,
        response_schema: Type[SchemaT],
        temperature: float = 0.2,
    ) -> StructuredGenerationResult[SchemaT]:
        """Generate and validate a Pydantic schema using Gemini primary and Groq fallback."""
        primary_name = getattr(self._primary, "provider_name", "gemini")
        attempts: list[dict[str, str]] = []

        try:
            validated = self._invoke_and_validate(
                self._primary,
                prompt=prompt,
                system_instruction=system_instruction,
                response_schema=response_schema,
                temperature=temperature,
            )
            model_name = (
                self._primary.get_model()
                if hasattr(self._primary, "get_model")
                else "unknown"
            )
            return StructuredGenerationResult(
                data=validated,
                metadata=LLMExecutionMetadata(
                    provider=primary_name,
                    model=model_name,
                    fallback_used=False,
                    attempted_providers=[primary_name],
                    fallback_reason=None,
                ),
            )
        except (GeminiProviderError, LLMInvalidOutputError, LLMOutputValidationError) as primary_exc:
            primary_error_type = _classify_error(primary_exc)
            attempts.append({"provider": primary_name, "error_type": primary_error_type})

            if self._fallback is None or not _is_retryable_primary_error(
                primary_exc,
                fallback_on_invalid_output=self._fallback_on_invalid_output,
            ):
                raise

            fallback_name = getattr(self._fallback, "provider_name", "groq")
            try:
                validated = self._invoke_and_validate(
                    self._fallback,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_schema=response_schema,
                    temperature=temperature,
                )
                fallback_model = (
                    self._fallback.get_model()
                    if hasattr(self._fallback, "get_model")
                    else "unknown"
                )
                return StructuredGenerationResult(
                    data=validated,
                    metadata=LLMExecutionMetadata(
                        provider=fallback_name,
                        model=fallback_model,
                        fallback_used=True,
                        attempted_providers=[primary_name, fallback_name],
                        fallback_reason=primary_error_type,
                    ),
                )
            except LLMOutputValidationError as fallback_val_exc:
                attempts.append({"provider": fallback_name, "error_type": "validation_error"})
                raise LLMOutputValidationError(
                    f"Fallback provider '{fallback_name}' output failed {response_schema.__name__} schema validation.",
                    attempts=attempts,
                ) from fallback_val_exc
            except LLMInvalidOutputError as fallback_json_exc:
                attempts.append({"provider": fallback_name, "error_type": "invalid_output"})
                raise LLMInvalidOutputError(
                    f"Fallback provider '{fallback_name}' returned unreadable JSON.",
                    attempts=attempts,
                ) from fallback_json_exc
            except (GroqProviderError, GeminiProviderError) as fallback_exc:
                fallback_error_type = _classify_error(fallback_exc)
                attempts.append({"provider": fallback_name, "error_type": fallback_error_type})
                raise LLMAllProvidersFailedError(
                    "All configured LLM providers failed to fulfill the request.",
                    attempts=attempts,
                    http_status=_determine_combined_http_status(attempts),
                ) from fallback_exc
