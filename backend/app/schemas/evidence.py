"""Pydantic v2 schemas for Stage 5: Deterministic Evidence Packaging."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.research import ResearchCategory

TRACKED_METADATA_FIELDS: tuple[str, ...] = (
    "title",
    "domain",
    "content_snippet",
    "published_date",
    "relevance_score",
)


class CategoryCoverageStatus(str, Enum):
    """Volume-based evidence coverage status for a research category.

    Reflects only the count of retrieved items, never source credibility or factual accuracy.
    """

    NONE = "none"
    LIMITED = "limited"
    MODERATE = "moderate"
    BROAD = "broad"


class PackagingError(BaseModel):
    """Safe, structured record describing an invalid source record skipped during packaging."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str | None = Field(
        default=None,
        description="Source URL if available on the malformed record.",
    )
    error_type: str = Field(
        ...,
        min_length=1,
        description="Machine-readable packaging error category.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Sanitized human-readable explanation without secrets or stack traces.",
    )


class PackagedEvidence(BaseModel):
    """Single deduplicated, normalized web evidence record with full provenance.

    Note: Retrieved content snippets are untrusted external excerpts; relevance_score
    reflects search-engine relevance (not credibility), and metadata_completeness
    reflects metadata presence (not factual accuracy).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    evidence_id: str = Field(
        ...,
        min_length=1,
        description="Deterministic identifier derived from the canonicalized URL.",
    )
    title: str | None = Field(
        default=None,
        description="Original page title if provided; never fabricated.",
    )
    url: str = Field(
        ...,
        min_length=1,
        description="Original source URL preserved verbatim for provenance.",
    )
    domain: str | None = Field(
        default=None,
        description="Normalized lowercase domain without leading 'www.' if parseable.",
    )
    content_snippet: str | None = Field(
        default=None,
        description="Untrusted text snippet from the source page if provided; never fabricated.",
    )
    research_category: ResearchCategory = Field(
        ...,
        description="Primary research category under which this source was first discovered.",
    )
    originating_queries: list[str] = Field(
        default_factory=list,
        description="All search queries that discovered this canonical URL.",
    )
    matched_categories: list[ResearchCategory] = Field(
        default_factory=list,
        description="All research categories associated with this canonical URL.",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        description="Original search-engine relevance score (not a credibility rating).",
    )
    published_date: str | None = Field(
        default=None,
        description="Original publication date if reported by the search provider; never fabricated.",
    )
    metadata_completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic fraction (0.0 to 1.0) of tracked metadata fields present "
            "(title, domain, content_snippet, published_date, relevance_score). "
            "Measures metadata presence only, not factual accuracy."
        ),
    )
    missing_metadata: list[str] = Field(
        default_factory=list,
        description="Tracked metadata fields absent from this packaged evidence item.",
    )


class EvidenceCategoryGroup(BaseModel):
    """Packaged evidence grouped under a single ResearchCategory."""

    category: ResearchCategory = Field(
        ...,
        description="Research category represented by this group.",
    )
    evidence_items: list[PackagedEvidence] = Field(
        default_factory=list,
        description="Deduplicated evidence records associated with this category.",
    )
    total_evidence: int = Field(
        ...,
        ge=0,
        description="Number of packaged evidence items in this category.",
    )
    unique_domains: list[str] = Field(
        default_factory=list,
        description="Sorted unique domains represented in this category.",
    )
    coverage_status: CategoryCoverageStatus = Field(
        ...,
        description="Volume-based coverage level ('none', 'limited', 'moderate', 'broad').",
    )

    @model_validator(mode="after")
    def validate_group_counts(self) -> EvidenceCategoryGroup:
        """Ensure total_evidence matches the number of items in evidence_items."""
        if self.total_evidence != len(self.evidence_items):
            raise ValueError("total_evidence must equal len(evidence_items).")
        return self


class MetadataQualitySummary(BaseModel):
    """Aggregate metadata-presence statistics across all unique packaged sources.

    Measures metadata completeness only — never factual accuracy or trustworthiness.
    """

    average_completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Arithmetic mean of metadata_completeness across unique packaged sources.",
    )
    complete_metadata_count: int = Field(
        ...,
        ge=0,
        description="Number of unique sources with all tracked metadata fields present (1.0).",
    )
    partial_metadata_count: int = Field(
        ...,
        ge=0,
        description="Number of unique sources with some, but not all, tracked metadata fields present.",
    )
    minimal_metadata_count: int = Field(
        ...,
        ge=0,
        description="Number of unique sources with zero tracked metadata fields present (0.0).",
    )
    missing_field_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of unique sources missing each tracked metadata field.",
    )


class EvidencePackage(BaseModel):
    """Deterministic, category-grouped evidence package derived from SearchExecutionReport."""

    total_sources_received: int = Field(
        ...,
        ge=0,
        description="Total source records inspected from the input SearchExecutionReport.",
    )
    unique_sources: int = Field(
        ...,
        ge=0,
        description="Count of unique deduplicated sources after canonical URL merging.",
    )
    duplicate_sources: int = Field(
        ...,
        ge=0,
        description="Count of duplicate source records merged by canonical URL.",
    )
    total_categories: int = Field(
        ...,
        ge=0,
        description="Total number of fixed research categories represented in categories.",
    )
    categories: list[EvidenceCategoryGroup] = Field(
        default_factory=list,
        description="Evidence grouped under all five fixed research categories.",
    )
    metadata_quality_summary: MetadataQualitySummary = Field(
        ...,
        description="Aggregate metadata-presence statistics across unique packaged sources.",
    )
    uncovered_categories: list[ResearchCategory] = Field(
        default_factory=list,
        description="Research categories that currently have zero packaged evidence.",
    )
    packaging_errors: list[PackagingError] = Field(
        default_factory=list,
        description="Safe structured errors for any malformed source records skipped during packaging.",
    )

    @model_validator(mode="after")
    def validate_package_invariants(self) -> EvidencePackage:
        """Verify internal consistency of category counts and source accounting."""
        if self.total_categories != len(self.categories):
            raise ValueError("total_categories must equal len(categories).")
        if self.unique_sources + self.duplicate_sources + len(self.packaging_errors) != self.total_sources_received:
            raise ValueError(
                "unique_sources + duplicate_sources + len(packaging_errors) "
                "must equal total_sources_received."
            )
        return self
