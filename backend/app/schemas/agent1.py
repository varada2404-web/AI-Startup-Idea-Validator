"""Pydantic v2 response schemas for unified Agent 1 orchestration."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.evidence import EvidencePackage
from app.schemas.idea import IdeaBrief
from app.schemas.research import (
    ResearchCategory,
    ResearchQueryPlan,
    SearchExecutionReport,
)


class Agent1OverallStatus(str, Enum):
    """Overall execution status of a unified Agent 1 pipeline run."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_SOURCES = "no_sources"


class Agent1ExecutionError(BaseModel):
    """Safe, structured execution error entry surfaced in Agent 1 metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    stage: str = Field(
        ...,
        min_length=1,
        description="Pipeline stage where the error occurred (e.g. 'web_search').",
    )
    error_type: str = Field(
        ...,
        min_length=1,
        description="Machine-readable error classification without secrets.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Sanitized human-readable error message without secrets or stack traces.",
    )
    query: str | None = Field(
        default=None,
        description="Optional search query associated with a search-stage error.",
    )
    category: ResearchCategory | None = Field(
        default=None,
        description="Optional research category associated with a search-stage error.",
    )


class Agent1ExecutionMetadata(BaseModel):
    """Per-request provider, search coverage, and safe error metadata for Agent 1."""

    model_config = ConfigDict(str_strip_whitespace=True)

    idea_analysis_provider: str = Field(
        ...,
        min_length=1,
        description="LLM provider that succeeded for Step 1 (Idea Analysis).",
    )
    idea_analysis_model: str = Field(
        ...,
        min_length=1,
        description="Model identifier used for Step 1 (Idea Analysis).",
    )
    idea_analysis_fallback_used: bool = Field(
        ...,
        description="Whether automatic fallback to Groq was used during Step 1.",
    )
    query_planning_provider: str = Field(
        ...,
        min_length=1,
        description="LLM provider that succeeded for Step 2 (Research Query Planning).",
    )
    query_planning_model: str = Field(
        ...,
        min_length=1,
        description="Model identifier used for Step 2 (Research Query Planning).",
    )
    query_planning_fallback_used: bool = Field(
        ...,
        description="Whether automatic fallback to Groq was used during Step 2.",
    )
    total_queries: int = Field(
        ...,
        ge=0,
        description="Total planned queries executed in Step 3.",
    )
    successful_queries: int = Field(
        ...,
        ge=0,
        description="Number of planned queries that completed successfully.",
    )
    failed_queries: int = Field(
        ...,
        ge=0,
        description="Number of planned queries that failed during Step 3.",
    )
    total_sources_retrieved: int = Field(
        ...,
        ge=0,
        description="Total raw web search results retrieved across all queries before deduplication.",
    )
    unique_sources_count: int = Field(
        ...,
        ge=0,
        description="Total unique, filtered, deduplicated sources returned.",
    )
    packaged_unique_sources: int = Field(
        default=0,
        ge=0,
        description="Unique deduplicated sources in the Stage 5 EvidencePackage.",
    )
    duplicate_sources_merged: int = Field(
        default=0,
        ge=0,
        description="Duplicate source records merged during Stage 5 Evidence Packaging.",
    )
    uncovered_categories: list[ResearchCategory] = Field(
        default_factory=list,
        description="Research categories with zero packaged evidence.",
    )
    average_metadata_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean metadata completeness across packaged sources (metadata presence only, not accuracy).",
    )
    overall_status: Agent1OverallStatus = Field(
        ...,
        description="Overall Agent 1 run status ('success', 'partial_success', or 'no_sources').",
    )
    errors: list[Agent1ExecutionError] = Field(
        default_factory=list,
        description="Safe, structured execution errors captured during partial failures.",
    )

    @model_validator(mode="after")
    def validate_query_counts(self) -> Agent1ExecutionMetadata:
        """Keep query counters internally consistent."""
        if self.successful_queries + self.failed_queries != self.total_queries:
            raise ValueError(
                "successful_queries + failed_queries must equal total_queries."
            )
        return self


class Agent1RunResponse(BaseModel):
    """Complete Agent 1 (Web Search & Idea Understanding) output returned by POST /api/v1/agent1/run."""

    idea_brief: IdeaBrief = Field(
        ...,
        description="Validated IdeaBrief extracted from the submitted startup idea.",
    )
    research_plan: ResearchQueryPlan = Field(
        ...,
        description="Validated multi-category ResearchQueryPlan generated from the IdeaBrief.",
    )
    search_report: SearchExecutionReport = Field(
        ...,
        description="Deduplicated Tavily web search results with provenance and per-query errors.",
    )
    evidence_package: EvidencePackage = Field(
        ...,
        description="Structured, deduplicated, category-grouped evidence package from Stage 5.",
    )
    execution_metadata: Agent1ExecutionMetadata = Field(
        ...,
        description="Safe provider selection, fallback, search summary, and status metadata.",
    )
