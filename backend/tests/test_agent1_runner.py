"""Tests for Stage 6: Unified Agent 1 Orchestration (Agent1Runner & POST /api/v1/agent1/run)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.agents.web_search.agent1_runner import Agent1Runner
from app.agents.web_search.idea_analyzer import IdeaAnalyzer
from app.agents.web_search.query_planner import QueryPlanner
from app.agents.web_search.search_executor import SearchExecutor
from app.main import (
    app,
    get_agent1_runner,
    get_gemini_provider,
    get_groq_provider,
    get_tavily_provider,
)
from app.providers.gemini import GeminiTimeoutError, GeminiUnavailableError
from app.providers.groq import GroqUnavailableError
from app.providers.llm_router import LLMFallbackService
from app.providers.tavily import (
    TavilyAuthError,
    TavilyConfigError,
    TavilyProvider,
    TavilyRateLimitError,
    TavilySearchItem,
    TavilyTimeoutError,
)
from app.schemas.agent1 import Agent1OverallStatus, Agent1RunResponse
from app.schemas.idea import IdeaBrief, StartupIdeaInput
from app.schemas.research import ResearchCategory, ResearchQueryPlan

client = TestClient(app)

BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_API_TESTS", "").strip().lower() in {"1", "true", "yes"}
requires_live_api = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Live API tests are disabled by default. Set RUN_LIVE_API_TESTS=1 to run live checks.",
)

SENTINEL_GEMINI_KEY = "AIza-AGENT1-SECRET-GEMINI-9999"
SENTINEL_GROQ_KEY = "gsk_AGENT1-SECRET-GROQ-8888"
SENTINEL_TAVILY_KEY = "tvly-AGENT1-SECRET-TAVILY-7777"

FOOD_WASTE_INPUT = {
    "idea": (
        "An AI-powered platform that helps restaurants reduce food waste through "
        "demand forecasting and inventory management."
    ),
    "target_geography": "India",
}

VALID_BRIEF_DICT = {
    "problem": "Restaurants in India incur high food costs and spoilage due to inaccurate demand forecasting and manual inventory management.",
    "solution": "An AI-powered platform that forecasts restaurant demand and automates inventory management to reduce food waste.",
    "target_customers": ["Mid-sized restaurants in India", "Quick-service restaurant chains", "Cloud kitchens"],
    "industry": "FoodTech / Restaurant Operations SaaS",
    "business_model": None,
    "target_geography": "India",
    "missing_information": [
        "Monetization and pricing structure",
        "Target restaurant size and POS integration scope",
    ],
    "assumptions": [
        "Restaurants in India have sufficient historical sales data for AI demand forecasting",
    ],
}

VALID_PLAN_DICT = {
    "target_geography": "India",
    "industry": "FoodTech / Restaurant Operations SaaS",
    "queries": [
        {
            "category": "market_research",
            "query": "India restaurant food waste management software market size",
            "rationale": "Estimate addressable market size in India.",
            "priority": "high",
        },
        {
            "category": "market_research",
            "query": "India foodservice industry inventory management SaaS adoption",
            "rationale": "Gauge digital inventory software penetration among Indian restaurants.",
            "priority": "high",
        },
        {
            "category": "competitor_research",
            "query": "restaurant demand forecasting inventory software competitors India",
            "rationale": "Identify direct competitors serving Indian restaurants.",
            "priority": "high",
        },
        {
            "category": "competitor_research",
            "query": "Petpooja Posist Restroworks inventory forecasting features India",
            "rationale": "Evaluate incumbent POS software overlap.",
            "priority": "medium",
        },
        {
            "category": "customer_pain_points",
            "query": "Indian restaurant owners food waste inventory spoilage challenges",
            "rationale": "Validate operational pain points around kitchen food waste.",
            "priority": "high",
        },
        {
            "category": "customer_pain_points",
            "query": "restaurant demand forecasting accuracy challenges multi-outlet India",
            "rationale": "Understand forecasting challenges in Indian restaurant operations.",
            "priority": "medium",
        },
        {
            "category": "pricing_and_business_models",
            "query": "restaurant inventory management SaaS pricing INR per outlet India",
            "rationale": "Benchmark willingness to pay in India.",
            "priority": "high",
        },
        {
            "category": "pricing_and_business_models",
            "query": "AI food waste platform restaurant SaaS subscription pricing model",
            "rationale": "Compare monetization structures used by global and local peers.",
            "priority": "medium",
        },
        {
            "category": "industry_trends",
            "query": "FSSAI food waste sustainability hospitality initiatives India",
            "rationale": "Identify regulatory and sustainability trends.",
            "priority": "medium",
        },
        {
            "category": "industry_trends",
            "query": "AI demand forecasting restaurant supply chain trends India",
            "rationale": "Track AI adoption in Indian foodservice operations.",
            "priority": "low",
        },
    ],
    "research_gaps": [
        "Willingness to pay for standalone forecasting versus bundled POS modules",
    ],
    "assumptions_to_validate": [
        "Indian restaurants maintain clean digital sales history suitable for AI forecasting",
    ],
}


class _StubLLMProvider:
    """Stub LLM provider supporting per-schema responses or errors."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        responses_by_schema: dict | None = None,
        errors_by_schema: dict | None = None,
        call_log: list[str] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self._model_name = model_name
        self.responses_by_schema = responses_by_schema or {}
        self.errors_by_schema = errors_by_schema or {}
        self.call_log = call_log
        self.calls: list[dict] = []

    def get_model(self) -> str:
        return self._model_name

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction=None,
        response_schema=None,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append({"response_schema": response_schema})
        if self.call_log is not None:
            self.call_log.append(f"{self.provider_name}:{response_schema.__name__}")
        if response_schema in self.errors_by_schema:
            raise self.errors_by_schema[response_schema]
        if response_schema in self.responses_by_schema:
            return self.responses_by_schema[response_schema]
        raise AssertionError(f"Unstubbed schema {response_schema} for {self.provider_name}")


class _StubTavilyProvider(TavilyProvider):
    """Stub TavilyProvider supporting per-query results, errors, and call logging."""

    def __init__(
        self,
        *,
        results_by_query: dict[str, list[TavilySearchItem]] | None = None,
        errors_by_query: dict[str, Exception] | None = None,
        default_results: list[TavilySearchItem] | None = None,
        default_error: Exception | None = None,
        config_error: Exception | None = None,
        call_log: list[str] | None = None,
    ) -> None:
        super().__init__(api_key=SENTINEL_TAVILY_KEY)
        self.results_by_query = results_by_query or {}
        self.errors_by_query = errors_by_query or {}
        self.default_results = default_results
        self.default_error = default_error
        self.config_error = config_error
        self.call_log = call_log
        self.calls: list[str] = []

    def validate_config(self) -> None:
        if self.config_error is not None:
            raise self.config_error

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        search_depth: str | None = None,
    ) -> list[TavilySearchItem]:
        self.calls.append(query)
        if self.call_log is not None:
            self.call_log.append("tavily:search")
        if query in self.errors_by_query:
            raise self.errors_by_query[query]
        if self.default_error is not None:
            raise self.default_error
        if query in self.results_by_query:
            return self.results_by_query[query]
        if self.default_results is not None:
            return self.default_results
        return [
            TavilySearchItem(
                title=f"Source for {query[:30]}",
                url=f"https://example.in/articles/{abs(hash(query)) % 100000}",
                content=f"Relevant market research content discussing {query} for restaurants in India.",
                relevance_score=0.82,
                domain="example.in",
            )
        ]


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. COMPLETE SUCCESSFUL PIPELINE + EXECUTION ORDER + GEMINI PRIMARY METADATA
# ---------------------------------------------------------------------------


def test_agent1_complete_pipeline_order_and_gemini_primary_metadata():
    call_log: list[str] = []

    gemini_stub = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_PLAN_DICT),
        },
        call_log=call_log,
    )
    groq_stub = _StubLLMProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        errors_by_schema={
            IdeaBrief: AssertionError("Groq should not be called when Gemini succeeds"),
            ResearchQueryPlan: AssertionError("Groq should not be called when Gemini succeeds"),
        },
        call_log=call_log,
    )
    tavily_stub = _StubTavilyProvider(call_log=call_log)

    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub
    app.dependency_overrides[get_tavily_provider] = lambda: tavily_stub

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 200

    parsed = Agent1RunResponse.model_validate(resp.json())
    assert parsed.idea_brief.target_geography == "India"
    assert parsed.research_plan.target_geography == "India"
    assert len(parsed.research_plan.queries) == 10
    assert parsed.search_report.total_queries == 10
    assert parsed.search_report.successful_queries == 10
    assert parsed.search_report.failed_queries == 0
    assert parsed.search_report.unique_sources_count == 10
    assert all(s.url.startswith("https://") and len(s.content) > 20 for s in parsed.search_report.sources)

    meta = parsed.execution_metadata
    assert meta.idea_analysis_provider == "gemini"
    assert meta.idea_analysis_model == "gemini-3.5-flash-lite"
    assert meta.idea_analysis_fallback_used is False
    assert meta.query_planning_provider == "gemini"
    assert meta.query_planning_model == "gemini-3.5-flash-lite"
    assert meta.query_planning_fallback_used is False
    assert meta.total_queries == 10
    assert meta.successful_queries == 10
    assert meta.failed_queries == 0
    assert meta.total_sources_retrieved == 10
    assert meta.unique_sources_count == 10
    assert meta.overall_status == Agent1OverallStatus.SUCCESS
    assert meta.errors == []

    # Verify strict execution order: Step 1 -> Step 2 -> Step 3 (10 Tavily searches)
    assert call_log[0] == "gemini:IdeaBrief"
    assert call_log[1] == "gemini:ResearchQueryPlan"
    assert call_log[2:] == ["tavily:search"] * 10
    assert len(groq_stub.calls) == 0


# ---------------------------------------------------------------------------
# 2. GROQ FALLBACK METADATA (INDEPENDENT PER STEP & NO CROSS-REQUEST LEAK)
# ---------------------------------------------------------------------------


def test_agent1_independent_step_fallback_metadata_and_no_cross_request_leak():
    # Request 1: Gemini succeeds for IdeaBrief, fails (503) for ResearchQueryPlan -> Groq handles Step 2
    gemini_partial = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={IdeaBrief: json.dumps(VALID_BRIEF_DICT)},
        errors_by_schema={ResearchQueryPlan: GeminiUnavailableError("503 high demand")},
    )
    groq_backup = _StubLLMProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_PLAN_DICT),
        },
    )
    tavily_stub = _StubTavilyProvider()

    service = LLMFallbackService(primary_provider=gemini_partial, fallback_provider=groq_backup)
    runner = Agent1Runner(
        idea_analyzer=IdeaAnalyzer(llm_service=service),
        query_planner=QueryPlanner(llm_service=service),
        search_executor=SearchExecutor(provider=tavily_stub),
    )

    result_1 = runner.run(StartupIdeaInput(**FOOD_WASTE_INPUT))
    meta_1 = result_1.execution_metadata
    assert meta_1.idea_analysis_provider == "gemini"
    assert meta_1.idea_analysis_fallback_used is False
    assert meta_1.query_planning_provider == "groq"
    assert meta_1.query_planning_model == "openai/gpt-oss-20b"
    assert meta_1.query_planning_fallback_used is True

    # Request 2 on the SAME runner after Gemini recovers -> metadata must NOT leak from Request 1
    gemini_partial.errors_by_schema.clear()
    gemini_partial.responses_by_schema[ResearchQueryPlan] = json.dumps(VALID_PLAN_DICT)

    result_2 = runner.run(StartupIdeaInput(**FOOD_WASTE_INPUT))
    meta_2 = result_2.execution_metadata
    assert meta_2.idea_analysis_provider == "gemini"
    assert meta_2.idea_analysis_fallback_used is False
    assert meta_2.query_planning_provider == "gemini"
    assert meta_2.query_planning_model == "gemini-3.5-flash-lite"
    assert meta_2.query_planning_fallback_used is False


# ---------------------------------------------------------------------------
# 3. QUERY PLANNING FAILURE (BOTH PROVIDERS FAIL / INVALID SCHEMA)
# ---------------------------------------------------------------------------


def test_agent1_query_planning_failure_returns_safe_error_and_skips_tavily():
    gemini_stub = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={IdeaBrief: json.dumps(VALID_BRIEF_DICT)},
        errors_by_schema={ResearchQueryPlan: GeminiTimeoutError("timeout")},
    )
    groq_stub = _StubLLMProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        errors_by_schema={ResearchQueryPlan: GroqUnavailableError("unavailable")},
    )
    tavily_stub = _StubTavilyProvider()

    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub
    app.dependency_overrides[get_tavily_provider] = lambda: tavily_stub

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "all_llm_providers_failed"
    assert detail["attempts"] == [
        {"provider": "gemini", "error_type": "timeout"},
        {"provider": "groq", "error_type": "unavailable"},
    ]
    assert len(tavily_stub.calls) == 0
    assert SENTINEL_GEMINI_KEY not in resp.text
    assert SENTINEL_GROQ_KEY not in resp.text
    assert "Traceback" not in resp.text


# ---------------------------------------------------------------------------
# 4. TAVILY TOTAL FAILURE (CONFIG, AUTH, AND ALL-QUERIES-FAILED)
# ---------------------------------------------------------------------------


def test_agent1_tavily_total_failures_return_safe_structured_errors():
    gemini_stub = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_PLAN_DICT),
        },
    )
    groq_stub = _StubLLMProvider(provider_name="groq", model_name="openai/gpt-oss-20b")
    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub

    # Case A: Missing TAVILY_API_KEY -> 503
    app.dependency_overrides[get_tavily_provider] = lambda: _StubTavilyProvider(
        config_error=TavilyConfigError("TAVILY_API_KEY is not configured.")
    )
    r_cfg = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert r_cfg.status_code == 503
    assert r_cfg.json()["detail"]["error"] == "search_not_configured"
    assert SENTINEL_TAVILY_KEY not in r_cfg.text

    # Case B: Tavily Auth Error -> 502
    app.dependency_overrides[get_tavily_provider] = lambda: _StubTavilyProvider(
        default_error=TavilyAuthError("Tavily API authentication failed.")
    )
    r_auth = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert r_auth.status_code == 502
    assert r_auth.json()["detail"]["error"] == "search_auth_failed"
    assert SENTINEL_TAVILY_KEY not in r_auth.text

    # Case C: Every planned query times out -> 504 with safe errors list
    app.dependency_overrides[get_tavily_provider] = lambda: _StubTavilyProvider(
        default_error=TavilyTimeoutError("Tavily search request timed out.")
    )
    r_all_timeout = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert r_all_timeout.status_code == 504
    detail = r_all_timeout.json()["detail"]
    assert detail["error"] == "web_search_failed"
    assert len(detail["errors"]) == 10
    assert SENTINEL_TAVILY_KEY not in r_all_timeout.text
    assert "Traceback" not in r_all_timeout.text


# ---------------------------------------------------------------------------
# 5. TAVILY PARTIAL FAILURE -> RETURNS PARTIAL REPORT + SAFE ERROR METADATA
# ---------------------------------------------------------------------------


def test_agent1_tavily_partial_failure_returns_partial_success_report():
    gemini_stub = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_PLAN_DICT),
        },
    )
    groq_stub = _StubLLMProvider(provider_name="groq", model_name="openai/gpt-oss-20b")
    q0 = VALID_PLAN_DICT["queries"][0]["query"]
    q2 = VALID_PLAN_DICT["queries"][2]["query"]
    tavily_stub = _StubTavilyProvider(
        errors_by_query={
            q0: TavilyTimeoutError("timeout"),
            q2: TavilyRateLimitError("rate limit"),
        }
    )

    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub
    app.dependency_overrides[get_tavily_provider] = lambda: tavily_stub

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 200

    parsed = Agent1RunResponse.model_validate(resp.json())
    meta = parsed.execution_metadata
    assert meta.overall_status == Agent1OverallStatus.PARTIAL_SUCCESS
    assert meta.total_queries == 10
    assert meta.successful_queries == 8
    assert meta.failed_queries == 2
    assert meta.unique_sources_count == 8
    assert len(meta.errors) == 2
    assert {e.error_type for e in meta.errors} == {"timeout", "rate_limit"}
    assert all(e.stage == "web_search" for e in meta.errors)
    assert SENTINEL_TAVILY_KEY not in resp.text


# ---------------------------------------------------------------------------
# 6. EMPTY SOURCE RESULTS
# ---------------------------------------------------------------------------


def test_agent1_empty_source_results_returns_no_sources_status():
    gemini_stub = _StubLLMProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_PLAN_DICT),
        },
    )
    groq_stub = _StubLLMProvider(provider_name="groq", model_name="openai/gpt-oss-20b")
    tavily_stub = _StubTavilyProvider(default_results=[])

    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub
    app.dependency_overrides[get_tavily_provider] = lambda: tavily_stub

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 200

    parsed = Agent1RunResponse.model_validate(resp.json())
    assert parsed.search_report.unique_sources_count == 0
    assert parsed.search_report.sources == []
    assert parsed.execution_metadata.overall_status == Agent1OverallStatus.NO_SOURCES
    assert parsed.execution_metadata.errors == []


# ---------------------------------------------------------------------------
# 7. UNEXPECTED INTERNAL ERROR & NO SECRET LEAKAGE
# ---------------------------------------------------------------------------


def test_agent1_unexpected_internal_error_never_leaks_tracebacks_or_secrets():
    class _CrashingRunner:
        def run(self, startup_input):
            raise RuntimeError(f"Secret leak test: {SENTINEL_GEMINI_KEY}")

    app.dependency_overrides[get_agent1_runner] = lambda: _CrashingRunner()

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "internal_server_error"
    assert SENTINEL_GEMINI_KEY not in resp.text
    assert "Traceback" not in resp.text


# ---------------------------------------------------------------------------
# 8. EXISTING ENDPOINT COMPATIBILITY & SWAGGER UI VERIFICATION
# ---------------------------------------------------------------------------


def test_existing_endpoints_and_swagger_ui_include_agent1_run():
    assert client.get("/").status_code == 200
    assert client.post("/api/v1/ideas", json=FOOD_WASTE_INPUT).status_code == 200

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger-ui" in docs.text.lower()

    openapi = client.get("/openapi.json").json()
    paths = set(openapi["paths"].keys())
    assert {
        "/",
        "/api/v1/ideas",
        "/api/v1/ideas/analyze",
        "/api/v1/ideas/plan-research",
        "/api/v1/ideas/search",
        "/api/v1/agent1/run",
    }.issubset(paths)

    agent1_post = openapi["paths"]["/api/v1/agent1/run"]["post"]
    req_ref = agent1_post["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    resp_ref = agent1_post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert req_ref.endswith("/StartupIdeaInput")
    assert resp_ref.endswith("/Agent1RunResponse")

    # Invalid input rejected with 422 before hitting providers
    r_invalid = client.post("/api/v1/agent1/run", json={"idea": "short", "target_geography": "I"})
    assert r_invalid.status_code == 422


# ---------------------------------------------------------------------------
# OPTIONAL LIVE END-TO-END AGENT 1 INTEGRATION TEST (RUN_LIVE_API_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_live_api
def test_live_agent1_end_to_end_run():
    """Execute the full live POST /api/v1/agent1/run pipeline with real Gemini/Groq and Tavily."""
    if BACKEND_ENV_PATH.exists():
        load_dotenv(dotenv_path=BACKEND_ENV_PATH, override=False)

    resp = client.post("/api/v1/agent1/run", json=FOOD_WASTE_INPUT)
    assert resp.status_code == 200, f"Live Agent 1 failed with {resp.status_code}: {resp.text}"

    parsed = Agent1RunResponse.model_validate(resp.json())
    assert parsed.idea_brief.target_geography == "India"
    assert parsed.research_plan.target_geography == "India"
    assert parsed.search_report.total_queries == len(parsed.research_plan.queries)
    assert parsed.search_report.unique_sources_count >= 1
    assert parsed.execution_metadata.overall_status in {
        Agent1OverallStatus.SUCCESS,
        Agent1OverallStatus.PARTIAL_SUCCESS,
    }

    print("\n=== LIVE AGENT 1 EXECUTION METADATA ===")
    print(json.dumps(parsed.execution_metadata.model_dump(mode="json"), indent=2))
    print("\n=== LIVE AGENT 1 IDEA BRIEF ===")
    print(json.dumps(parsed.idea_brief.model_dump(mode="json"), indent=2))
    print("\n=== LIVE AGENT 1 SAMPLE QUERY & SOURCE ===")
    print("First planned query:", json.dumps(parsed.research_plan.queries[0].model_dump(mode="json"), indent=2))
    if parsed.search_report.sources:
        print("First retrieved source:", json.dumps(parsed.search_report.sources[0].model_dump(mode="json"), indent=2))
