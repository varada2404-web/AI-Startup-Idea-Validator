"""
Automated tests for Competitor Discovery and Comparison Agent (Milestone 2)
---------------------------------------------------------------------------
Run with:  pytest backend/tests/test_competitor_agent.py -v
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.competitor_discovery import CompetitorDiscoveryAgent
from app.agents.competitor_discovery.schemas import CompetitorReport
from app.providers.search.base import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    url: str = "https://example.com",
    domain: str = "example.com",
    title: str = "Example Tool",
    text: str = "A tool for HR teams",
    category: str = "direct_competitors",
) -> SearchResult:
    return SearchResult(
        url=url,
        domain=domain,
        title=title,
        query="test query",
        category=category,  # type: ignore[arg-type]
        text=text,
        retrieved_at=datetime.now(UTC),
        simulated=False,
    )


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.search = AsyncMock(return_value=[
        _make_result(
            url="https://workday.com/hiring",
            domain="workday.com",
            title="Workday - AI Hiring Solution",
            text="Workday offers AI-powered resume screening with ATS integration.",
            category="direct_competitors",
        ),
        _make_result(
            url="https://greenhouse.io/product",
            domain="greenhouse.io",
            title="Greenhouse | Recruiting Software",
            text="Greenhouse provides structured hiring with limited AI features.",
            category="direct_competitors",
        ),
        _make_result(
            url="https://review.io/complaints",
            domain="review.io",
            title="User Reviews",
            text="Users say it is expensive and difficult to use with no integration available.",
            category="negative_evidence",
        ),
    ])
    return provider


@pytest.fixture
def agent(mock_provider):
    return CompetitorDiscoveryAgent(search_provider=mock_provider)


# ---------------------------------------------------------------------------
# CompetitorDiscoveryAgent — query generation
# ---------------------------------------------------------------------------

class TestQueryGeneration:
    def test_no_hardcoded_year_in_templates(self):
        """Query templates must not contain hardcoded years."""
        from app.agents.competitor_discovery.agent import COMPETITOR_QUERY_TEMPLATES
        for template, _ in COMPETITOR_QUERY_TEMPLATES:
            assert "2024" not in template, f"Hardcoded year in: '{template}'"
            assert "2023" not in template, f"Hardcoded year in: '{template}'"

    def test_queries_contain_idea(self):
        """All generated queries must contain the startup idea."""
        from app.agents.competitor_discovery.agent import _build_queries
        idea = "AI resume screener"
        queries = _build_queries(idea)
        for query, _ in queries:
            assert idea in query, f"Idea missing from query: '{query}'"

    def test_generates_correct_number_of_queries(self):
        """Should generate one query per template."""
        from app.agents.competitor_discovery.agent import (
            COMPETITOR_QUERY_TEMPLATES,
            _build_queries,
        )
        queries = _build_queries("test idea")
        assert len(queries) == len(COMPETITOR_QUERY_TEMPLATES)

    def test_dynamic_year_in_queries(self):
        """The year in queries must match the current year."""
        from datetime import datetime

        from app.agents.competitor_discovery.agent import _build_queries
        current_year = str(datetime.now().year)
        queries = _build_queries("some startup idea")
        year_queries = [q for q, _ in queries if current_year in q]
        assert len(year_queries) >= 1, "At least one query should include the current year"


# ---------------------------------------------------------------------------
# CompetitorDiscoveryAgent — run output
# ---------------------------------------------------------------------------

class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_competitor_report(self, agent):
        """run() must return a CompetitorReport instance."""
        report = await agent.run("AI resume screener for HR")
        assert isinstance(report, CompetitorReport)

    @pytest.mark.asyncio
    async def test_report_contains_idea(self, agent):
        """Report must echo the input idea."""
        idea = "AI resume screener for HR"
        report = await agent.run(idea)
        assert report.idea == idea

    @pytest.mark.asyncio
    async def test_report_has_competitors(self, agent):
        """Report must include identified competitors."""
        report = await agent.run("AI resume screener for HR")
        assert len(report.competitors) > 0

    @pytest.mark.asyncio
    async def test_competitors_have_required_fields(self, agent):
        """Every CompetitorProfile must have name and url."""
        report = await agent.run("AI resume screener for HR")
        for competitor in report.competitors:
            assert competitor.name
            assert competitor.url

    @pytest.mark.asyncio
    async def test_report_has_search_queries(self, agent):
        """Report must list the queries that were used."""
        report = await agent.run("AI resume screener for HR")
        assert len(report.search_queries_used) > 0

    @pytest.mark.asyncio
    async def test_report_has_summary(self, agent):
        """Report must include a non-empty summary."""
        report = await agent.run("AI resume screener for HR")
        assert report.summary

    @pytest.mark.asyncio
    async def test_gap_analysis_populated_from_negative_evidence(self, agent):
        """Gaps should be identified from negative_evidence category results."""
        report = await agent.run("AI resume screener for HR")
        assert len(report.gap_analysis) > 0

    @pytest.mark.asyncio
    async def test_domain_deduplication(self, mock_provider):
        """Results from the same domain should produce one competitor profile."""
        mock_provider.search = AsyncMock(return_value=[
            _make_result(url="https://workday.com/page1", domain="workday.com", title="Workday A"),
            _make_result(url="https://workday.com/page2", domain="workday.com", title="Workday B"),
        ])
        agent = CompetitorDiscoveryAgent(search_provider=mock_provider)
        report = await agent.run("test idea")
        workday_profiles = [c for c in report.competitors if "workday" in (c.url or "")]
        assert len(workday_profiles) == 1
        # Both URLs should be in source_urls
        assert len(workday_profiles[0].source_urls) == 2


# ---------------------------------------------------------------------------
# CompetitorDiscoveryAgent — error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_partial_search_failure_captured_in_errors(self, mock_provider):
        """Partial search failures should be captured in report.errors."""
        call_count = 0

        async def side_effect(query, category, round_number):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated search failure")
            return []

        mock_provider.search = side_effect
        agent = CompetitorDiscoveryAgent(search_provider=mock_provider)
        report = await agent.run("test idea")
        assert len(report.errors) == 1

    @pytest.mark.asyncio
    async def test_all_searches_fail_returns_empty_report(self, mock_provider):
        """All searches failing should return an empty but valid report."""
        mock_provider.search = AsyncMock(side_effect=RuntimeError("network error"))
        agent = CompetitorDiscoveryAgent(search_provider=mock_provider)
        report = await agent.run("test idea")
        assert isinstance(report, CompetitorReport)
        assert report.competitors == []
        assert len(report.errors) > 0

    @pytest.mark.asyncio
    async def test_empty_results_handled_gracefully(self, mock_provider):
        """Empty search results should return an empty but valid report."""
        mock_provider.search = AsyncMock(return_value=[])
        agent = CompetitorDiscoveryAgent(search_provider=mock_provider)
        report = await agent.run("test idea")
        assert isinstance(report, CompetitorReport)
        assert report.competitors == []
        assert "No direct competitors found" in report.summary
