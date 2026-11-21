"""Unified Agent 1 orchestrator: IdeaAnalyzer -> QueryPlanner -> SearchExecutor -> EvidencePackager."""

from __future__ import annotations

from app.agents.web_search.evidence_packager import EvidencePackager
from app.agents.web_search.idea_analyzer import IdeaAnalyzer
from app.agents.web_search.query_planner import QueryPlanner
from app.agents.web_search.search_executor import SearchExecutor
from app.providers.llm_router import LLMExecutionMetadata
from app.schemas.agent1 import (
    Agent1ExecutionError,
    Agent1ExecutionMetadata,
    Agent1OverallStatus,
    Agent1RunResponse,
)
from app.schemas.idea import IdeaBrief, StartupIdeaInput
from app.schemas.research import ResearchQueryPlan, SearchQueryError


class Agent1OrchestrationError(Exception):
    """Base exception for Agent 1 orchestration failures."""


class Agent1SearchFailedError(Agent1OrchestrationError):
    """Raised when every planned Tavily query fails during Agent 1 execution."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[SearchQueryError],
        http_status: int = 503,
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.http_status = http_status


def _default_metadata() -> LLMExecutionMetadata:
    """Fallback metadata used only when a custom test double omits metadata."""
    return LLMExecutionMetadata(
        provider="unknown",
        model="unknown",
        fallback_used=False,
        attempted_providers=[],
        fallback_reason=None,
    )


class Agent1Runner:
    """Orchestrates the complete Agent 1 pipeline without storing per-request state."""

    def __init__(
        self,
        *,
        idea_analyzer: IdeaAnalyzer | None = None,
        query_planner: QueryPlanner | None = None,
        search_executor: SearchExecutor | None = None,
        evidence_packager: EvidencePackager | None = None,
    ) -> None:
        self._idea_analyzer = idea_analyzer or IdeaAnalyzer()
        self._query_planner = query_planner or QueryPlanner()
        self._search_executor = search_executor or SearchExecutor()
        self._evidence_packager = evidence_packager or EvidencePackager()

    def _run_idea_analysis(
        self, startup_input: StartupIdeaInput
    ) -> tuple[IdeaBrief, LLMExecutionMetadata]:
        if hasattr(self._idea_analyzer, "analyze_with_metadata"):
            return self._idea_analyzer.analyze_with_metadata(startup_input)
        brief = self._idea_analyzer.analyze(startup_input)
        meta = getattr(self._idea_analyzer, "last_execution_metadata", None) or _default_metadata()
        return brief, meta

    def _run_query_planning(
        self, idea_brief: IdeaBrief
    ) -> tuple[ResearchQueryPlan, LLMExecutionMetadata]:
        if hasattr(self._query_planner, "plan_with_metadata"):
            return self._query_planner.plan_with_metadata(idea_brief)
        plan = self._query_planner.plan(idea_brief)
        meta = getattr(self._query_planner, "last_execution_metadata", None) or _default_metadata()
        return plan, meta

    def run(self, startup_input: StartupIdeaInput) -> Agent1RunResponse:
        """Execute Steps 1-5 in strict order and return the validated Agent1RunResponse."""
        if not isinstance(startup_input, StartupIdeaInput):
            raise TypeError("startup_input must be a StartupIdeaInput instance.")

        # Step 1: Idea Understanding (Gemini primary, Groq automatic fallback)
        idea_brief, idea_meta = self._run_idea_analysis(startup_input)

        # Step 2: Research Query Planning (Gemini primary, Groq automatic fallback)
        research_plan, plan_meta = self._run_query_planning(idea_brief)

        # Step 3: Tavily Web Search Execution (sequential, max 3 results/query, deduplicated)
        search_report = self._search_executor.execute(research_plan)

        # Total search failure: every planned query failed -> raise structured error
        if search_report.total_queries > 0 and search_report.successful_queries == 0:
            all_timeouts = bool(search_report.errors) and all(
                err.error_type == "timeout" for err in search_report.errors
            )
            raise Agent1SearchFailedError(
                "All planned web search queries failed to complete.",
                errors=search_report.errors,
                http_status=504 if all_timeouts else 503,
            )

        # Step 4: Deterministic Evidence Packaging (no LLM calls)
        evidence_package = self._evidence_packager.package(search_report)

        # Step 5: Combine all outputs and independent per-step metadata
        execution_errors = [
            Agent1ExecutionError(
                stage="web_search",
                error_type=err.error_type,
                message=err.message,
                query=err.query,
                category=err.category,
            )
            for err in search_report.errors
        ]
        execution_errors.extend(
            Agent1ExecutionError(
                stage="evidence_packaging",
                error_type=err.error_type,
                message=err.message,
            )
            for err in evidence_package.packaging_errors
        )

        if search_report.failed_queries > 0:
            overall_status = Agent1OverallStatus.PARTIAL_SUCCESS
        elif search_report.unique_sources_count == 0:
            overall_status = Agent1OverallStatus.NO_SOURCES
        else:
            overall_status = Agent1OverallStatus.SUCCESS

        execution_metadata = Agent1ExecutionMetadata(
            idea_analysis_provider=idea_meta.provider,
            idea_analysis_model=idea_meta.model,
            idea_analysis_fallback_used=idea_meta.fallback_used,
            query_planning_provider=plan_meta.provider,
            query_planning_model=plan_meta.model,
            query_planning_fallback_used=plan_meta.fallback_used,
            total_queries=search_report.total_queries,
            successful_queries=search_report.successful_queries,
            failed_queries=search_report.failed_queries,
            total_sources_retrieved=search_report.total_results_retrieved,
            unique_sources_count=search_report.unique_sources_count,
            packaged_unique_sources=evidence_package.unique_sources,
            duplicate_sources_merged=evidence_package.duplicate_sources,
            uncovered_categories=evidence_package.uncovered_categories,
            average_metadata_completeness=evidence_package.metadata_quality_summary.average_completeness,
            overall_status=overall_status,
            errors=execution_errors,
        )

        return Agent1RunResponse(
            idea_brief=idea_brief,
            research_plan=research_plan,
            search_report=search_report,
            evidence_package=evidence_package,
            execution_metadata=execution_metadata,
        )


def run_agent1(
    startup_input: StartupIdeaInput,
    *,
    idea_analyzer: IdeaAnalyzer | None = None,
    query_planner: QueryPlanner | None = None,
    search_executor: SearchExecutor | None = None,
    evidence_packager: EvidencePackager | None = None,
) -> Agent1RunResponse:
    """Convenience function to run the complete Agent 1 pipeline."""
    return Agent1Runner(
        idea_analyzer=idea_analyzer,
        query_planner=query_planner,
        search_executor=search_executor,
        evidence_packager=evidence_packager,
    ).run(startup_input)
