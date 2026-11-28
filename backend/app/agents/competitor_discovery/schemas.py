"""
Pydantic schemas for Competitor Discovery and Comparison Agent (Milestone 2).
"""

from pydantic import BaseModel, Field


class CompetitorProfile(BaseModel):
    """A single identified competitor with comparison data."""

    name: str
    url: str
    domain: str = ""
    description: str = ""
    source_urls: list[str] = Field(default_factory=list)
    offerings: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class CompetitorReport(BaseModel):
    """Full output of the Competitor Discovery and Comparison Agent."""

    idea: str
    competitors: list[CompetitorProfile] = Field(default_factory=list)
    gap_analysis: list[str] = Field(default_factory=list)
    summary: str = ""
    search_queries_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
