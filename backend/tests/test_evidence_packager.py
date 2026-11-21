"""Mocked tests for Stage 5: Deterministic Evidence Packaging & Agent 1 Integration."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agents.web_search.agent1_runner import Agent1Runner
from app.agents.web_search.evidence_packager import EvidencePackager, package_evidence
from app.agents.web_search.idea_analyzer import IdeaAnalyzer
from app.agents.web_search.query_planner import QueryPlanner
from app.agents.web_search.search_executor import SearchExecutor
from app.main import (
    app,
    get_gemini_provider,
    get_groq_provider,
    get_tavily_provider,
)
from app.providers.llm_router import LLMFallbackService
from app.providers.tavily import TavilyProvider, TavilySearchItem
from app.schemas.agent1 import Agent1RunResponse
from app.schemas.evidence import (
    CategoryCoverageStatus,
    EvidencePackage,
)
from app.schemas.idea import IdeaBrief, StartupIdeaInput
from app.schemas.research import (
    ResearchCategory,
    ResearchQueryPlan,
    SearchExecutionReport,
    SourceRecord,
)

client = TestClient(app)


def _build_report(sources: list[SourceRecord]) -> SearchExecutionReport:
    """Helper constructing a valid SearchExecutionReport for packaging tests."""
    return SearchExecutionReport(
        total_queries=max(1, len(sources)),
        successful_queries=max(1, len(sources)),
        failed_queries=0,
        total_results_retrieved=len(sources),
        unique_sources_count=len(sources),
        sources=sources,
        errors=[],
    )


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. NORMAL EVIDENCE PACKAGING & SAFEGUARDS
# ---------------------------------------------------------------------------


def test_normal_evidence_packaging_and_safeguards():
    sources = [
        SourceRecord(
            title="India FoodTech Market Report 2025",
            url="https://www.marketinsights.in/reports/foodtech-2025",
            content="Indian restaurant software spending is projected to expand across Tier-1 and Tier-2 cities.",
            query="India restaurant food waste software market size",
            category=ResearchCategory.MARKET_RESEARCH,
            matched_queries=["India restaurant food waste software market size"],
            matched_categories=[ResearchCategory.MARKET_RESEARCH],
            relevance_score=0.87,
            published_date="2025-02-10",
            domain="marketinsights.in",
        ),
        SourceRecord(
            title="Restaurant Inventory SaaS Pricing in India",
            url="https://saasbenchmarks.co.in/restaurant-pricing",
            content="Monthly subscriptions for independent Indian restaurants typically range from INR 2,000 to INR 5,000.",
            query="restaurant inventory SaaS pricing INR India",
            category=ResearchCategory.PRICING_AND_BUSINESS_MODELS,
            matched_queries=["restaurant inventory SaaS pricing INR India"],
            matched_categories=[ResearchCategory.PRICING_AND_BUSINESS_MODELS],
            relevance_score=0.79,
            published_date=None,
            domain="saasbenchmarks.co.in",
        ),
    ]

    pkg = package_evidence(_build_report(sources))

    assert isinstance(pkg, EvidencePackage)
    assert pkg.total_sources_received == 2
    assert pkg.unique_sources == 2
    assert pkg.duplicate_sources == 0
    assert pkg.total_categories == 5
    assert pkg.packaging_errors == []

    groups_by_cat = {g.category: g for g in pkg.categories}
    market_group = groups_by_cat[ResearchCategory.MARKET_RESEARCH]
    assert market_group.total_evidence == 1
    assert market_group.unique_domains == ["marketinsights.in"]
    assert market_group.coverage_status == CategoryCoverageStatus.LIMITED

    item = market_group.evidence_items[0]
    assert item.evidence_id.startswith("ev_")
    assert item.url == "https://www.marketinsights.in/reports/foodtech-2025"
    assert item.domain == "marketinsights.in"
    assert item.relevance_score == pytest.approx(0.87)
    assert item.published_date == "2025-02-10"
    assert item.metadata_completeness == pytest.approx(1.0)
    assert item.missing_metadata == []

    # Pricing source lacks published_date -> 4/5 = 0.80, never fabricated
    pricing_item = groups_by_cat[ResearchCategory.PRICING_AND_BUSINESS_MODELS].evidence_items[0]
    assert pricing_item.published_date is None
    assert pricing_item.metadata_completeness == pytest.approx(0.8)
    assert pricing_item.missing_metadata == ["published_date"]

    # Summary checks
    assert pkg.metadata_quality_summary.complete_metadata_count == 1
    assert pkg.metadata_quality_summary.partial_metadata_count == 1
    assert pkg.metadata_quality_summary.minimal_metadata_count == 0
    assert pkg.metadata_quality_summary.average_completeness == pytest.approx(0.9)
    assert pkg.metadata_quality_summary.missing_field_counts["published_date"] == 1

    # Safeguard check: no credibility/trustworthiness/verified labels added
    serialized = json.dumps(pkg.model_dump(mode="json")).lower()
    for forbidden in ("credibility_score", "trustworthy", "verified_fact"):
        assert forbidden not in serialized


# ---------------------------------------------------------------------------
# 2-4. DUPLICATE URL DETECTION, ORIGINATING QUERIES & CATEGORY ASSOCIATIONS
# ---------------------------------------------------------------------------


def test_duplicate_url_merging_preserves_originating_queries_and_categories():
    src_first = SourceRecord(
        title="Winnow & Local Competitors in Indian Kitchen Analytics",
        url="https://www.foodopsindia.com/articles/waste-analytics/",
        content="Short initial snippet about food waste analytics in India.",
        query="India restaurant food waste market size",
        category=ResearchCategory.MARKET_RESEARCH,
        matched_queries=["India restaurant food waste market size"],
        matched_categories=[ResearchCategory.MARKET_RESEARCH],
        relevance_score=0.68,
        published_date=None,
        domain="foodopsindia.com",
    )
    src_duplicate = SourceRecord(
        title="Winnow & Local Competitors in Indian Kitchen Analytics",
        url="http://foodopsindia.com/articles/waste-analytics?utm_source=tavily&utm_campaign=test#summary",
        content=(
            "Expanded snippet discussing Winnow, Leanpath, and local POS vendors competing "
            "for mid-sized Indian restaurant food waste analytics deployments."
        ),
        query="restaurant food waste tracking competitors India",
        category=ResearchCategory.COMPETITOR_RESEARCH,
        matched_queries=["restaurant food waste tracking competitors India"],
        matched_categories=[ResearchCategory.COMPETITOR_RESEARCH],
        relevance_score=0.91,
        published_date="2025-03-18",
        domain="www.foodopsindia.com",
    )

    pkg = EvidencePackager().package(_build_report([src_first, src_duplicate]))

    assert pkg.total_sources_received == 2
    assert pkg.unique_sources == 1
    assert pkg.duplicate_sources == 1

    groups_by_cat = {g.category: g for g in pkg.categories}
    market_items = groups_by_cat[ResearchCategory.MARKET_RESEARCH].evidence_items
    competitor_items = groups_by_cat[ResearchCategory.COMPETITOR_RESEARCH].evidence_items

    # Same deduplicated item appears under BOTH matched categories with identical evidence_id
    assert len(market_items) == 1
    assert len(competitor_items) == 1
    assert market_items[0].evidence_id == competitor_items[0].evidence_id

    merged = market_items[0]
    assert merged.url == "https://www.foodopsindia.com/articles/waste-analytics/"
    assert merged.originating_queries == [
        "India restaurant food waste market size",
        "restaurant food waste tracking competitors India",
    ]
    assert merged.matched_categories == [
        ResearchCategory.MARKET_RESEARCH,
        ResearchCategory.COMPETITOR_RESEARCH,
    ]
    assert merged.relevance_score == pytest.approx(0.91)
    assert merged.published_date == "2025-03-18"
    assert "Winnow" in (merged.content_snippet or "")
    assert merged.metadata_completeness == pytest.approx(1.0)
    assert merged.missing_metadata == []


# ---------------------------------------------------------------------------
# 5. MISSING METADATA (NEVER DISCARDS OR FABRICATES)
# ---------------------------------------------------------------------------


def test_missing_metadata_preserves_source_and_computes_deterministic_completeness():
    # Construct an incomplete SourceRecord via model_construct to simulate missing title/snippet
    incomplete_source = SourceRecord.model_construct(
        title="",
        url="https://reports.hospitality.in/sparse-page",
        content="",
        query="India restaurant kitchen waste challenges",
        category=ResearchCategory.CUSTOMER_PAIN_POINTS,
        matched_queries=["India restaurant kitchen waste challenges"],
        matched_categories=[ResearchCategory.CUSTOMER_PAIN_POINTS],
        relevance_score=None,
        published_date=None,
        domain=None,
    )
    report = SearchExecutionReport.model_construct(
        total_queries=1,
        successful_queries=1,
        failed_queries=0,
        total_results_retrieved=1,
        unique_sources_count=1,
        sources=[incomplete_source],
        errors=[],
    )

    pkg = EvidencePackager().package(report)

    assert pkg.unique_sources == 1
    assert pkg.packaging_errors == []

    pain_group = next(g for g in pkg.categories if g.category == ResearchCategory.CUSTOMER_PAIN_POINTS)
    assert pain_group.total_evidence == 1
    item = pain_group.evidence_items[0]

    # Domain normalized from URL; missing title/snippet/published_date/relevance_score kept as None
    assert item.domain == "reports.hospitality.in"
    assert item.title is None
    assert item.content_snippet is None
    assert item.published_date is None
    assert item.relevance_score is None
    assert item.missing_metadata == ["title", "content_snippet", "published_date", "relevance_score"]
    assert item.metadata_completeness == pytest.approx(0.2)
    assert pkg.metadata_quality_summary.partial_metadata_count == 1
    assert pkg.metadata_quality_summary.missing_field_counts == {
        "title": 1,
        "domain": 0,
        "content_snippet": 1,
        "published_date": 1,
        "relevance_score": 1,
    }


# ---------------------------------------------------------------------------
# 6 & 7. EMPTY SEARCH RESULTS & UNCOVERED CATEGORIES
# ---------------------------------------------------------------------------


def test_empty_search_results_marks_all_five_categories_uncovered():
    empty_report = SearchExecutionReport(
        total_queries=10,
        successful_queries=10,
        failed_queries=0,
        total_results_retrieved=0,
        unique_sources_count=0,
        sources=[],
        errors=[],
    )

    pkg = EvidencePackager().package(empty_report)

    assert pkg.total_sources_received == 0
    assert pkg.unique_sources == 0
    assert pkg.duplicate_sources == 0
    assert pkg.total_categories == 5
    assert pkg.uncovered_categories == list(ResearchCategory)
    assert all(g.coverage_status == CategoryCoverageStatus.NONE for g in pkg.categories)
    assert all(g.total_evidence == 0 for g in pkg.categories)
    assert pkg.metadata_quality_summary.average_completeness == 0.0
    assert pkg.packaging_errors == []


def test_uncovered_categories_lists_exact_missing_categories():
    src = SourceRecord(
        title="FSSAI Sustainability Mandate 2025",
        url="https://fssai-news.example.in/sustainability-2025",
        content="New hospitality guidelines encourage Indian restaurants to monitor surplus food.",
        query="FSSAI food waste trends India",
        category=ResearchCategory.INDUSTRY_TRENDS,
        matched_queries=["FSSAI food waste trends India"],
        matched_categories=[ResearchCategory.INDUSTRY_TRENDS],
        relevance_score=0.75,
        domain="fssai-news.example.in",
    )

    pkg = EvidencePackager().package(_build_report([src]))

    assert pkg.uncovered_categories == [
        ResearchCategory.MARKET_RESEARCH,
        ResearchCategory.COMPETITOR_RESEARCH,
        ResearchCategory.CUSTOMER_PAIN_POINTS,
        ResearchCategory.PRICING_AND_BUSINESS_MODELS,
    ]


# ---------------------------------------------------------------------------
# 8. INVALID OR MALFORMED SOURCE RECORDS
# ---------------------------------------------------------------------------


def test_invalid_source_records_are_captured_in_packaging_errors_without_crashing():
    valid_source = SourceRecord(
        title="Valid Restaurant Report",
        url="https://valid.example.in/report",
        content="Valid market overview snippet for Indian restaurants.",
        query="India restaurant market",
        category=ResearchCategory.MARKET_RESEARCH,
        matched_queries=["India restaurant market"],
        matched_categories=[ResearchCategory.MARKET_RESEARCH],
        relevance_score=0.80,
        domain="valid.example.in",
    )
    bad_url_source = SourceRecord.model_construct(
        title="Bad Scheme",
        url="javascript:alert(1)",
        content="Malicious URL",
        query="India restaurant market",
        category=ResearchCategory.MARKET_RESEARCH,
        matched_queries=[],
        matched_categories=[],
        relevance_score=0.5,
        published_date=None,
        domain=None,
    )
    bad_category_source = SourceRecord.model_construct(
        title="Unknown Category",
        url="https://example.in/unknown-cat",
        content="Snippet with invalid category",
        query="India restaurant market",
        category="not_a_real_category",
        matched_queries=[],
        matched_categories=[],
        relevance_score=0.5,
        published_date=None,
        domain="example.in",
    )
    report = SearchExecutionReport.model_construct(
        total_queries=1,
        successful_queries=1,
        failed_queries=0,
        total_results_retrieved=4,
        unique_sources_count=4,
        sources=[valid_source, bad_url_source, bad_category_source, "corrupt-entry"],  # type: ignore[list-item]
        errors=[],
    )

    pkg = EvidencePackager().package(report)

    assert pkg.total_sources_received == 4
    assert pkg.unique_sources == 1
    assert pkg.duplicate_sources == 0
    assert len(pkg.packaging_errors) == 3
    assert {e.error_type for e in pkg.packaging_errors} == {
        "invalid_url",
        "invalid_category",
        "invalid_source_record",
    }

    with pytest.raises(TypeError):
        EvidencePackager().package({"not": "a report"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. DETERMINISTIC OUTPUT
# ---------------------------------------------------------------------------


def test_evidence_packaging_is_strictly_deterministic():
    s1 = SourceRecord(
        title="Source A",
        url="https://alpha.example.in/a",
        content="Content for source A covering restaurant operations in India.",
        query="q1",
        category=ResearchCategory.MARKET_RESEARCH,
        matched_queries=["q1"],
        matched_categories=[ResearchCategory.MARKET_RESEARCH],
        relevance_score=0.72,
        domain="alpha.example.in",
    )
    s2 = SourceRecord(
        title="Source B",
        url="https://beta.example.in/b",
        content="Content for source B covering restaurant operations in India.",
        query="q2",
        category=ResearchCategory.MARKET_RESEARCH,
        matched_queries=["q2"],
        matched_categories=[ResearchCategory.MARKET_RESEARCH],
        relevance_score=0.89,
        published_date="2025-01-20",
        domain="beta.example.in",
    )

    pkg_order_1 = EvidencePackager().package(_build_report([s1, s2]))
    pkg_order_2 = EvidencePackager().package(_build_report([s2, s1]))

    assert pkg_order_1.model_dump(mode="json") == pkg_order_2.model_dump(mode="json")


# ---------------------------------------------------------------------------
# 10. STANDALONE ENDPOINT: POST /api/v1/ideas/package-evidence
# ---------------------------------------------------------------------------


def test_standalone_package_evidence_endpoint():
    payload = _build_report(
        [
            SourceRecord(
                title="India Cloud Kitchen Waste Study",
                url="https://kitchenstudy.in/waste-2025",
                content="Study examining daily food waste patterns across Indian cloud kitchens.",
                query="cloud kitchen waste India",
                category=ResearchCategory.CUSTOMER_PAIN_POINTS,
                matched_queries=["cloud kitchen waste India"],
                matched_categories=[ResearchCategory.CUSTOMER_PAIN_POINTS],
                relevance_score=0.84,
                published_date="2025-04-02",
                domain="kitchenstudy.in",
            )
        ]
    ).model_dump(mode="json")

    resp = client.post("/api/v1/ideas/package-evidence", json=payload)
    assert resp.status_code == 200

    pkg = EvidencePackage.model_validate(resp.json())
    assert pkg.unique_sources == 1
    assert pkg.metadata_quality_summary.average_completeness == pytest.approx(1.0)

    # Invalid payload rejected with HTTP 422
    bad_resp = client.post("/api/v1/ideas/package-evidence", json={"total_queries": -1})
    assert bad_resp.status_code == 422


# ---------------------------------------------------------------------------
# 11. AGENT 1 ORCHESTRATION INTEGRATION (STAGE 5 INCLUDED IN /api/v1/agent1/run)
# ---------------------------------------------------------------------------


def test_agent1_pipeline_executes_stage_5_evidence_packaging_in_order():
    from tests.test_agent1_runner import (
        FOOD_WASTE_INPUT,
        VALID_BRIEF_DICT,
        VALID_PLAN_DICT,
        _StubLLMProvider,
        _StubTavilyProvider,
    )

    call_log: list[str] = []

    class _TracingPackager(EvidencePackager):
        def package(self, search_report: SearchExecutionReport) -> EvidencePackage:
            call_log.append("evidence_packager:package")
            return super().package(search_report)

    gemini_stub = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_PLAN_DICT),
        },
        call_log=call_log,
    )
    groq_stub = _StubLLMProvider(provider_name="groq", model_name="openai/gpt-oss-20b")
    tavily_stub = _StubTavilyProvider(call_log=call_log)

    service = LLMFallbackService(primary_provider=gemini_stub, fallback_provider=groq_stub)
    runner = Agent1Runner(
        idea_analyzer=IdeaAnalyzer(llm_service=service),
        query_planner=QueryPlanner(llm_service=service),
        search_executor=SearchExecutor(provider=tavily_stub),
        evidence_packager=_TracingPackager(),
    )

    result = runner.run(StartupIdeaInput(**FOOD_WASTE_INPUT))

    assert isinstance(result, Agent1RunResponse)
    assert call_log[0] == "gemini:IdeaBrief"
    assert call_log[1] == "gemini:ResearchQueryPlan"
    assert call_log[2:12] == ["tavily:search"] * 10
    assert call_log[12] == "evidence_packager:package"

    # Existing fields preserved alongside new evidence_package
    assert result.search_report.unique_sources_count == 10
    assert result.evidence_package.unique_sources == 10
    assert result.evidence_package.uncovered_categories == []
    assert result.execution_metadata.packaged_unique_sources == 10
    assert result.execution_metadata.duplicate_sources_merged == 0
    assert result.execution_metadata.uncovered_categories == []
    assert result.execution_metadata.average_metadata_completeness == pytest.approx(0.8)

    # Also verify via HTTP endpoint POST /api/v1/agent1/run
    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub
    app.dependency_overrides[get_tavily_provider] = lambda: tavily_stub

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 200
    parsed = Agent1RunResponse.model_validate(resp.json())
    assert parsed.evidence_package.unique_sources == 10
    assert parsed.evidence_package.total_categories == 5
