"""Asynchronous Gemini implementation for Agent 1 idea understanding."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import ConfigDict, ValidationError

from app.core.config import ApplicationConfig
from app.providers.llm.errors import ProviderFailure
from app.schemas.contracts import (
    IdeaBrief,
    ProviderError,
    ProviderRunProvenance,
    ValidationRequest,
)
from app.services.idea_understanding import extract_keywords

PROMPT_VERSION = "idea-brief-v1"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "idea_brief_v1.txt"


class GeminiIdeaBriefOutput(IdeaBrief):
    """Strict response contract for the SDK, with M1 validation inherited."""

    model_config = ConfigDict(extra="forbid")


class GeminiIdeaUnderstandingProvider:
    """Generate one validated IdeaBrief using Google's async Gen AI client.

    The SDK client is initialized lazily and can be injected in tests. The adapter never
    logs SDK exceptions, raw response bodies, prompts, or credentials.
    """

    def __init__(self, config: ApplicationConfig, client: Any | None = None) -> None:
        if config.mode != "live" or not config.gemini_api_key or not config.gemini_model:
            raise ValueError("Gemini provider requires LIVE mode, a key, and a model setting.")
        self._config = config
        self._client = client
        self._sync_client: Any | None = None
        self._prompt = PROMPT_PATH.read_text(encoding="utf-8")
        prompt_hash = hashlib.sha256(self._prompt.encode("utf-8")).hexdigest()
        self.provenance = ProviderRunProvenance(
            provider="gemini",
            model_identifier=config.gemini_model,
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash,
        )

    async def extract_idea_brief(self, request: ValidationRequest) -> IdeaBrief:
        try:
            client = self._get_client()
            async with asyncio.timeout(self._config.gemini_timeout_seconds):
                response = await client.models.generate_content(
                    model=self._config.gemini_model,
                    contents=self._input_content(request),
                    config=self._generation_config(
                        system_instruction=self._prompt,
                        response_mime_type="application/json",
                        response_schema=GeminiIdeaBriefOutput,
                        max_output_tokens=self._config.gemini_max_output_tokens,
                        temperature=0,
                    ),
                )
        except TimeoutError:
            raise self._failure("timeout", "Gemini request timed out. Try again later.", True)
        except ProviderFailure:
            raise
        except Exception as exc:
            raise self._classify_provider_exception(exc) from None

        if self._response_exceeds_limit(response):
            raise self._failure(
                "response_too_large",
                "Gemini response exceeded the configured size limit.",
            )
        if self._response_is_incomplete(response):
            raise self._failure(
                "incomplete_response",
                "Gemini output was truncated. Reduce the idea context or raise the output limit.",
            )
        brief = self._parse_response(response)
        self._validate_request_provenance(brief, request)
        # Search scope is controlled by the deterministic planner and original idea terms.
        brief.search_keywords = extract_keywords(request.idea)
        brief.interpretation_mode = "gemini_live"
        return brief

    async def aclose(self) -> None:
        """Close the SDK clients if this adapter created them."""

        if self._client is not None and self._sync_client is not None:
            try:
                await self._client.aclose()
            finally:
                self._sync_client.close()
                self._client = None
                self._sync_client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError:
            raise self._failure(
                "sdk_unavailable",
                "Gemini support is unavailable because its provider package is not installed.",
            ) from None

        key = self._config.gemini_api_key
        if key is None:  # Defensive: ApplicationConfig already checks this for LIVE mode.
            raise self._failure("configuration_error", "Gemini credentials are not configured.")
        try:
            self._sync_client = genai.Client(
                api_key=key.get_secret_value(),
                http_options=types.HttpOptions(
                    timeout=round(self._config.gemini_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._client = self._sync_client.aio
        except Exception:
            self._sync_client = None
            raise self._failure(
                "client_initialization_failed", "Gemini client could not be initialized."
            ) from None
        return self._client

    @staticmethod
    def _input_content(request: ValidationRequest) -> str:
        payload = {
            "idea": request.idea,
            "geography": request.geography,
            "target_customer": request.target_customer,
        }
        return "Founder submission data (untrusted text):\n" + json.dumps(
            payload, ensure_ascii=False
        )

    @staticmethod
    def _generation_config(**values: Any) -> types.GenerateContentConfig:
        fields = getattr(types.GenerateContentConfig, "model_fields", {})
        if "automatic_continuation" not in fields:
            raise GeminiIdeaUnderstandingProvider._failure(
                "sdk_incompatible",
                "Installed google-genai SDK lacks required automatic-continuation control. "
                "Install the project dependencies before running LIVE mode.",
            )
        values["automatic_continuation"] = False
        return types.GenerateContentConfig(**values)

    @staticmethod
    def _response_is_incomplete(response: Any) -> bool:
        for candidate in getattr(response, "candidates", None) or []:
            reason = getattr(candidate, "finish_reason", None)
            reason_value = getattr(reason, "value", reason)
            normalized = str(reason_value).upper()
            if normalized.endswith("MAX_TOKENS") or normalized.endswith("CONTINUATION"):
                return True
        return False

    def _response_exceeds_limit(self, response: Any) -> bool:
        limit = self._config.max_provider_response_bytes
        text = getattr(response, "text", None)
        if isinstance(text, str) and len(text.encode("utf-8")) > limit:
            return True
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            return False
        try:
            if hasattr(parsed, "model_dump_json"):
                size = len(parsed.model_dump_json().encode("utf-8"))
            else:
                size = len(json.dumps(parsed, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError):
            return False  # The schema parser will return a typed malformed-response error.
        return size > limit

    def _parse_response(self, response: Any) -> IdeaBrief:
        try:
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                return GeminiIdeaBriefOutput.model_validate(parsed)
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise self._failure(
                    "malformed_response", "Gemini returned no structured idea brief."
                )
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                raise self._failure(
                    "malformed_response", "Gemini returned malformed structured output."
                ) from None
            return GeminiIdeaBriefOutput.model_validate(payload)
        except ValidationError:
            raise self._failure(
                "schema_validation_failed",
                "Gemini output did not meet the required Agent 1 idea brief schema.",
            ) from None
        except ProviderFailure:
            raise
        except Exception:
            raise self._failure(
                "malformed_response", "Gemini returned an unreadable structured response."
            ) from None

    def _validate_request_provenance(
        self, brief: IdeaBrief, request: ValidationRequest
    ) -> None:
        if brief.original_text != request.idea:
            raise self._failure(
                "schema_validation_failed",
                "Gemini output did not preserve the submitted idea exactly.",
            )
        if request.target_customer:
            if (
                brief.target_customer.value != request.target_customer
                or brief.target_customer.origin != "user"
            ):
                raise self._failure(
                    "schema_validation_failed",
                    "Gemini output did not preserve the submitted target customer.",
                )
        elif brief.target_customer.origin not in {"inferred", "unknown"}:
            raise self._failure(
                "schema_validation_failed",
                "Gemini output used an invalid target-customer provenance label.",
            )
        if request.geography:
            if (
                brief.geography.value != request.geography
                or brief.geography.origin != "user"
            ):
                raise self._failure(
                    "schema_validation_failed",
                    "Gemini output did not preserve the submitted geography.",
                )
        elif brief.geography.value != "India" or brief.geography.origin != "default":
            raise self._failure(
                "schema_validation_failed",
                "Gemini output must label the default geography as India.",
            )

    @staticmethod
    def _classify_provider_exception(exc: Exception) -> ProviderFailure:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        status_text = str(status).upper()
        type_names = {cls.__name__.casefold() for cls in type(exc).__mro__}
        if status in {401, 403} or any(
            marker in status_text for marker in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        ):
            return GeminiIdeaUnderstandingProvider._failure(
                "authentication_failed", "Gemini authentication failed. Check server settings."
            )
        if status == 429 or "RESOURCE_EXHAUSTED" in status_text:
            return GeminiIdeaUnderstandingProvider._failure(
                "rate_limited", "Gemini rate limit reached. Try again later.", True
            )
        if any("timeout" in name for name in type_names):
            return GeminiIdeaUnderstandingProvider._failure(
                "timeout", "Gemini request timed out. Try again later.", True
            )
        if isinstance(exc, OSError) or any(
            marker in name
            for name in type_names
            for marker in ("connectionerror", "connecterror", "networkerror", "requesterror")
        ):
            return GeminiIdeaUnderstandingProvider._failure(
                "network_error", "Gemini could not be reached. Try again later.", True
            )
        return GeminiIdeaUnderstandingProvider._failure(
            "provider_error", "Gemini could not produce the idea brief."
        )

    @staticmethod
    def _failure(code: str, message: str, retryable: bool = False) -> ProviderFailure:
        return ProviderFailure(
            ProviderError(
                provider="gemini",
                stage="idea_understanding",
                code=code,
                message=message,
                retryable=retryable,
            )
        )
