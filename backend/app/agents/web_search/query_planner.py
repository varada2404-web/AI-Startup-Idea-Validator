"""Standalone Research Query Planner for Agent 1 using Gemini primary with Groq fallback."""

from __future__ import annotations

from app.agents.web_search.query_prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_query_planning_prompt,
)
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.llm_router import (
    LLMExecutionMetadata,
    LLMFallbackService,
    LLMInvalidOutputError,
    LLMOutputValidationError,
)
from app.schemas.idea import IdeaBrief
from app.schemas.research import ResearchQueryPlan


class QueryPlanningError(Exception):
    """Base exception for research query planning failures."""


class InvalidQueryPlanOutputError(QueryPlanningError):
    """Raised when the model output is not valid JSON or not a JSON object."""

    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class QueryPlanValidationError(QueryPlanningError):
    """Raised when parsed model output fails ResearchQueryPlan Pydantic validation."""

    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class QueryPlanner:
    """Generates a validated ResearchQueryPlan from an IdeaBrief via LLMFallbackService."""

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
        """Safe execution metadata from the most recent plan() call."""
        return self._last_execution_metadata

    def plan_with_metadata(
        self, idea_brief: IdeaBrief
    ) -> tuple[ResearchQueryPlan, LLMExecutionMetadata]:
        """Generate a ResearchQueryPlan and return (ResearchQueryPlan, LLMExecutionMetadata).

        Returns metadata directly per call to avoid relying on shared mutable state.
        """
        if not isinstance(idea_brief, IdeaBrief):
            raise TypeError("idea_brief must be an IdeaBrief instance.")

        prompt = build_query_planning_prompt(idea_brief)

        try:
            result = self._llm_service.generate_structured(
                prompt=prompt,
                system_instruction=QUERY_PLANNER_SYSTEM_PROMPT,
                response_schema=ResearchQueryPlan,
            )
        except LLMInvalidOutputError as exc:
            raise InvalidQueryPlanOutputError(str(exc), attempts=exc.attempts) from exc
        except LLMOutputValidationError as exc:
            raise QueryPlanValidationError(str(exc), attempts=exc.attempts) from exc

        plan = result.data

        # Guarantee the exact target geography from the IdeaBrief is preserved verbatim.
        if plan.target_geography != idea_brief.target_geography:
            plan = plan.model_copy(
                update={"target_geography": idea_brief.target_geography}
            )

        return plan, result.metadata

    def plan(self, idea_brief: IdeaBrief) -> ResearchQueryPlan:
        """Generate and validate a structured research query plan for the given IdeaBrief."""
        plan, metadata = self.plan_with_metadata(idea_brief)
        self._last_execution_metadata = metadata
        return plan


def plan_research_queries(
    idea_brief: IdeaBrief,
    provider: GeminiProvider | None = None,
    *,
    fallback_provider: GroqProvider | None = None,
    llm_service: LLMFallbackService | None = None,
) -> ResearchQueryPlan:
    """Convenience function to plan research queries from an IdeaBrief."""
    return QueryPlanner(
        provider=provider,
        fallback_provider=fallback_provider,
        llm_service=llm_service,
    ).plan(idea_brief)
