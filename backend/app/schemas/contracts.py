"""Pydantic contracts for the M1 demo API and run state."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ApplicationMode = Literal["demo", "live"]
SourceType = Literal["mock", "tavily"]
ExtractionStatus = Literal["not_attempted", "succeeded", "failed"]
ProviderName = Literal["gemini", "tavily"]
ProviderStage = Literal["idea_understanding", "search", "extraction"]
RunQuality = Literal["complete", "degraded"]

ResearchCategory = Literal[
    "customer_pain",
    "market",
    "direct_competitors",
    "alternatives",
    "pricing",
    "recent_developments",
    "negative_evidence",
]


class ValidationRequest(BaseModel):
    idea: str = Field(min_length=20, max_length=1000)
    geography: str | None = Field(default=None, max_length=100)
    target_customer: str | None = Field(default=None, max_length=200)

    @field_validator("idea", mode="before")
    @classmethod
    def remove_control_characters(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return "".join(char for char in value if char.isprintable() or char in "\n\t").strip()

    @field_validator("geography", "target_customer")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value else value


class Provenance(BaseModel):
    value: str | None = None
    origin: Literal["user", "default", "inferred", "unknown"] = "unknown"


class Hypothesis(BaseModel):
    id: str
    kind: Literal["demand", "willingness_to_pay", "competition", "feasibility", "timing"]
    statement: str
    critical: bool = False
    status: Literal["unknown", "supported", "mixed", "contradicted"] = "unknown"


class IdeaBrief(BaseModel):
    original_text: str
    problem: str
    solution: str
    target_customer: Provenance
    geography: Provenance
    industry: str
    business_model: Literal[
        "subscription", "marketplace", "saas", "ads", "hardware", "other", "unclear"
    ] = "unclear"
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    search_keywords: list[str] = Field(default_factory=list)
    interpretation_mode: Literal["deterministic_demo", "gemini_live"] = "deterministic_demo"

    @model_validator(mode="after")
    def hypotheses_remain_unassessed_in_m1(self) -> "IdeaBrief":
        expected = {"demand", "willingness_to_pay", "competition", "feasibility", "timing"}
        actual = {hypothesis.kind for hypothesis in self.hypotheses}
        if len(self.hypotheses) != 5 or actual != expected:
            raise ValueError("M1 idea briefs must contain the five defined hypothesis types.")
        if any(hypothesis.status != "unknown" for hypothesis in self.hypotheses):
            raise ValueError("M1 hypotheses must remain unknown and unassessed.")
        if sum(hypothesis.critical for hypothesis in self.hypotheses) != 2:
            raise ValueError("M1 idea briefs must mark exactly two hypotheses as critical.")
        return self


class QueryPlanItem(BaseModel):
    category: ResearchCategory
    query: str
    round: Literal[1, 2] = 1


class ProviderError(BaseModel):
    """Sanitized provider failure suitable for persistence and API responses."""

    provider: ProviderName
    stage: ProviderStage
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)
    retryable: bool = False


class ProviderRunProvenance(BaseModel):
    provider: Literal["mock", "gemini"]
    model_identifier: str
    prompt_version: str
    prompt_hash: str | None = None


class RunManifest(BaseModel):
    idea_understanding: ProviderRunProvenance | None = None


class Source(BaseModel):
    id: str
    url: str
    domain: str
    title: str
    query: str
    queries: list[str] = Field(default_factory=list)
    categories: list[ResearchCategory]
    source_type: SourceType = "mock"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cluster_id: str
    text: str
    simulated: bool = True
    snippet: str | None = None
    extracted_text: str | None = None
    extraction_status: ExtractionStatus = "not_attempted"
    extraction_error: ProviderError | None = None
    canonical_url: str | None = None
    provider_score: float | None = None
    published_at: datetime | None = None
    provider_metadata: dict[str, str | int | float | bool | None] | None = None

    @model_validator(mode="after")
    def source_mode_matches_provenance(self) -> "Source":
        if (self.source_type == "mock") != self.simulated:
            raise ValueError("Mock sources must be simulated; Tavily sources must be live.")
        if self.extraction_status == "succeeded" and not self.extracted_text:
            raise ValueError("Successful extraction requires extracted_text.")
        if self.extraction_status == "failed" and self.extracted_text:
            raise ValueError("Failed extraction cannot include extracted_text.")
        if self.extraction_error is not None and self.extraction_status != "failed":
            raise ValueError("An extraction error requires failed extraction status.")
        if self.extraction_error is not None and self.extraction_error.provider != "tavily":
            raise ValueError("Extraction errors must identify Tavily as the provider.")
        return self


class CategoryCoverage(BaseModel):
    category: ResearchCategory
    status: Literal["covered", "partial", "missing"]
    source_count: int = 0
    independent_clusters: int = 0


class ProgressItem(BaseModel):
    stage: str
    label: str
    status: Literal["pending", "running", "completed", "failed", "interrupted"] = "pending"
    detail: str | None = None


class ValidationRun(BaseModel):
    run_id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["queued", "running", "completed", "failed", "interrupted"] = "queued"
    request: ValidationRequest
    idea_brief: IdeaBrief | None = None
    queries: list[QueryPlanItem] = Field(default_factory=list)
    coverage: list[CategoryCoverage] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    progress: list[ProgressItem] = Field(default_factory=list)
    raw_result_count: int = 0
    deduplicated_count: int = 0
    mode: ApplicationMode = "demo"
    demo_mode: bool = True
    quality: RunQuality = "complete"
    provider_errors: list[ProviderError] = Field(default_factory=list)
    manifest: RunManifest = Field(default_factory=RunManifest)
    simulation_notice: str | None = (
        "DEMO MODE: results are deterministic synthetic placeholders. "
        "No live search or market research was performed."
    )
    error: str | None = None

    @model_validator(mode="after")
    def mode_fields_are_consistent(self) -> "ValidationRun":
        if self.demo_mode != (self.mode == "demo"):
            raise ValueError("demo_mode must match the run mode.")
        if self.mode == "live" and self.simulation_notice is not None:
            raise ValueError("Live runs cannot carry the demo simulation notice.")
        if self.idea_brief is not None:
            expected_interpretation = "deterministic_demo" if self.mode == "demo" else "gemini_live"
            if self.idea_brief.interpretation_mode != expected_interpretation:
                raise ValueError("Idea brief interpretation mode must match the run mode.")
        expected_source_type = "mock" if self.mode == "demo" else "tavily"
        if any(source.source_type != expected_source_type for source in self.sources):
            raise ValueError("Source types must match the run mode.")
        terminal_stage_status = {
            "completed": "completed",
            "failed": "failed",
            "interrupted": "interrupted",
        }.get(self.status)
        if terminal_stage_status:
            for stage in self.progress:
                if stage.status == "running":
                    stage.status = terminal_stage_status
        return self


class ValidationAccepted(BaseModel):
    run_id: str
    status: Literal["queued"]
    mode: ApplicationMode = "demo"
    demo_mode: bool = True

    @model_validator(mode="after")
    def mode_fields_are_consistent(self) -> "ValidationAccepted":
        if self.demo_mode != (self.mode == "demo"):
            raise ValueError("demo_mode must match the application mode.")
        return self


class ApiError(BaseModel):
    error: dict[str, str]
