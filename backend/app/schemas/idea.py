"""Pydantic schemas for startup idea submission and structured idea briefs."""

from pydantic import BaseModel, ConfigDict, Field


class StartupIdeaInput(BaseModel):
    """Input payload for submitting a startup idea for validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    idea: str = Field(
        ...,
        min_length=20,
        description="Detailed description of the startup idea to validate.",
        examples=[
            "An AI-powered mobile app that helps small retail shop owners automate inventory tracking using smartphone photos."
        ],
    )
    target_geography: str = Field(
        ...,
        min_length=2,
        description="Target country, region, or market geography for the startup.",
        examples=["India", "United States", "Europe"],
    )


class StartupIdeaResponse(BaseModel):
    """Response returned when a startup idea submission is accepted."""

    message: str = Field(
        ...,
        description="Status message confirming receipt of the startup idea.",
    )
    idea: str = Field(
        ...,
        description="The submitted startup idea.",
    )
    target_geography: str = Field(
        ...,
        description="The submitted target geography.",
    )


class IdeaBrief(BaseModel):
    """Structured brief extracted from a raw startup idea description."""

    model_config = ConfigDict(str_strip_whitespace=True)

    problem: str = Field(
        ...,
        min_length=1,
        description="Core problem the startup aims to solve.",
    )
    solution: str = Field(
        ...,
        min_length=1,
        description="Proposed product or service addressing the problem.",
    )
    target_customers: list[str] = Field(
        ...,
        description="Primary customer segments or user personas.",
    )
    industry: str = Field(
        ...,
        min_length=1,
        description="Primary industry or market vertical.",
    )
    business_model: str | None = Field(
        default=None,
        description="Optional description of how the startup plans to generate revenue.",
    )
    target_geography: str = Field(
        ...,
        min_length=2,
        description="Target geographic market for the startup.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Key details absent from the initial idea submission.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions made while structuring the idea brief.",
    )
