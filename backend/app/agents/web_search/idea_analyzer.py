"""Idea Understanding analyzer for Agent 1 using Gemini primary with Groq fallback."""

from __future__ import annotations

from app.agents.web_search.prompts import (
    IDEA_UNDERSTANDING_SYSTEM_PROMPT,
    build_idea_understanding_prompt,
)
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.llm_router import (
    LLMExecutionMetadata,
    LLMFallbackService,
    LLMInvalidOutputError,
    LLMOutputValidationError,
)
from app.schemas.idea import IdeaBrief, StartupIdeaInput


class IdeaAnalysisError(Exception):
    """Base exception for Idea Understanding analysis failures."""


class InvalidModelOutputError(IdeaAnalysisError):
    """Raised when the model output is not valid JSON or not a JSON object."""

    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class IdeaValidationError(IdeaAnalysisError):
    """Raised when parsed model output fails IdeaBrief Pydantic validation."""

    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class IdeaAnalyzer:
    """Transforms a raw StartupIdeaInput into a validated IdeaBrief via LLMFallbackService."""

    def __init__(
        self,
        provider: GeminiProvider | None = None,
        *,
        fallback_provider: GroqProvider | None = None,
        llm_service: LLMFallbackService | None = None,
    ) -> None:
        if llm_service is not None:
            self._llm_service = llm_service
        elif provider is not None:
            self._llm_service = LLMFallbackService(
                primary_provider=provider,
                fallback_provider=fallback_provider,
                enable_fallback=fallback_provider is not None,
            )
        else:
            self._llm_service = LLMFallbackService(
                primary_provider=GeminiProvider(),
                fallback_provider=fallback_provider or GroqProvider(),
                enable_fallback=True,
            )
        self._last_execution_metadata: LLMExecutionMetadata | None = None

    @property
    def last_execution_metadata(self) -> LLMExecutionMetadata | None:
        """Safe execution metadata from the most recent analyze() call."""
        return self._last_execution_metadata

    def analyze_with_metadata(
        self, startup_input: StartupIdeaInput
    ) -> tuple[IdeaBrief, LLMExecutionMetadata]:
        """Analyze a startup idea submission and return (IdeaBrief, LLMExecutionMetadata).

        Returns metadata directly per call to avoid relying on shared mutable state.
        """
        if not isinstance(startup_input, StartupIdeaInput):
            raise TypeError("startup_input must be a StartupIdeaInput instance.")

        prompt = build_idea_understanding_prompt(startup_input)

        try:
            result = self._llm_service.generate_structured(
                prompt=prompt,
                system_instruction=IDEA_UNDERSTANDING_SYSTEM_PROMPT,
                response_schema=IdeaBrief,
            )
        except LLMInvalidOutputError as exc:
            raise InvalidModelOutputError(str(exc), attempts=exc.attempts) from exc
        except LLMOutputValidationError as exc:
            raise IdeaValidationError(str(exc), attempts=exc.attempts) from exc

        brief = result.data

        # Guarantee the founder's exact submitted geography is preserved verbatim.
        if brief.target_geography != startup_input.target_geography:
            brief = brief.model_copy(
                update={"target_geography": startup_input.target_geography}
            )

        return brief, result.metadata

    def analyze(self, startup_input: StartupIdeaInput) -> IdeaBrief:
        """Analyze a startup idea submission and return a validated IdeaBrief."""
        brief, metadata = self.analyze_with_metadata(startup_input)
        self._last_execution_metadata = metadata
        return brief


def analyze_startup_idea(
    startup_input: StartupIdeaInput,
    provider: GeminiProvider | None = None,
    *,
    fallback_provider: GroqProvider | None = None,
    llm_service: LLMFallbackService | None = None,
) -> IdeaBrief:
    """Convenience function to analyze a StartupIdeaInput with IdeaAnalyzer."""
    return IdeaAnalyzer(
        provider=provider,
        fallback_provider=fallback_provider,
        llm_service=llm_service,
    ).analyze(startup_input)
