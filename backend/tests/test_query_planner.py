"""Tests for Agent 1: Research Query Planner (schemas, standalone planner, and API endpoint)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.web_search.idea_analyzer import IdeaAnalyzer
from app.agents.web_search.query_planner import (
    InvalidQueryPlanOutputError,
    QueryPlanner,
    QueryPlanValidationError,
    plan_research_queries,
)
from app.agents.web_search.query_prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_query_planning_prompt,
)
from app.main import (
    app,
    get_gemini_provider,
    get_idea_analyzer,
    get_llm_service,
    get_query_planner,
)
from app.providers.gemini import (
    GeminiConfigError,
    GeminiInvalidResponseError,
    GeminiProvider,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from app.providers.llm_router import LLMFallbackService
from app.schemas.idea import IdeaBrief, StartupIdeaInput
from app.schemas.research import (
    QueryPriority,
    ResearchCategory,
    ResearchPlanResponse,
    ResearchQuery,
    ResearchQueryPlan,
)

client = TestClient(app)

BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_API_TESTS", "").strip().lower() in {"1", "true", "yes"}
requires_live_api = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Live API tests are disabled by default. Set RUN_LIVE_API_TESTS=1 to run live checks.",
)


RESTAURANT_FOOD_WASTE_INPUT = {
    "idea": (
        "An AI-powered kitchen analytics and smart bin monitoring platform that helps mid-sized "
        "restaurants and cloud kitchens in India measure daily food waste, optimize ingredient "
        "purchasing, and cut food costs through a monthly SaaS subscription."
    ),
    "target_geography": "India",
}

RESTAURANT_FOOD_WASTE_BRIEF = {
    "problem": (
        "Mid-sized restaurants and cloud kitchens in India suffer high food costs due to unmeasured "
        "daily kitchen food waste and inefficient ingredient purchasing."
    ),
    "solution": (
        "An AI-powered kitchen analytics and smart bin monitoring platform that tracks food waste "
        "and provides purchasing optimization insights."
    ),
    "target_customers": [
        "Mid-sized restaurants in India",
        "Cloud kitchens in India",
    ],
    "industry": "FoodTech / Restaurant Operations SaaS",
    "business_model": "Monthly SaaS subscription",
    "target_geography": "India",
    "missing_information": [
        "Hardware cost and deployment model for smart bins",
        "Target monthly subscription pricing in INR",
        "Integration requirements with Indian restaurant POS systems",
    ],
    "assumptions": [
        "Restaurant staff will consistently use smart bins during busy kitchen shifts",
        "Food cost savings will outweigh combined software and hardware costs",
    ],
}

RESTAURANT_FOOD_WASTE_PLAN = {
    "target_geography": "India",
    "industry": "FoodTech / Restaurant Operations SaaS",
    "queries": [
        {
            "category": "market_research",
            "query": "India restaurant food waste management software market size growth",
            "rationale": "Estimate addressable market size and growth trajectory in India.",
            "priority": "high",
        },
        {
            "category": "market_research",
            "query": "cloud kitchens market India number of outlets food cost percentage",
            "rationale": "Quantify target customer base and typical food cost structure in India.",
            "priority": "high",
        },
        {
            "category": "competitor_research",
            "query": "restaurant food waste tracking software competitors India Winnow Leanpath",
            "rationale": "Identify global and local competitors already serving Indian hospitality.",
            "priority": "high",
        },
        {
            "category": "competitor_research",
            "query": "Petpooja Posist UrbanPiper inventory food waste management features India",
            "rationale": "Assess overlap with incumbent Indian restaurant POS and inventory platforms.",
            "priority": "medium",
        },
        {
            "category": "customer_pain_points",
            "query": "Indian restaurant owners food waste inventory leakage challenges survey",
            "rationale": "Validate whether food waste is a top operational pain point for owners.",
            "priority": "high",
        },
        {
            "category": "customer_pain_points",
            "query": "cloud kitchen staff adoption barriers kitchen monitoring hardware India",
            "rationale": "Uncover operational friction and staff compliance hurdles in commercial kitchens.",
            "priority": "medium",
        },
        {
            "category": "pricing_and_business_models",
            "query": "restaurant SaaS inventory management pricing INR per outlet India",
            "rationale": "Benchmark willingness to pay among mid-sized Indian restaurants.",
            "priority": "high",
        },
        {
            "category": "pricing_and_business_models",
            "query": "smart bin AI food waste monitoring hardware plus SaaS pricing model",
            "rationale": "Check whether competitors bundle hardware into subscriptions or charge upfront.",
            "priority": "medium",
        },
        {
            "category": "industry_trends",
            "query": "FSSAI food waste regulations sustainability initiatives hospitality India 2025",
            "rationale": "Check regulatory tailwinds or compliance pressures around food waste in India.",
            "priority": "medium",
        },
        {
            "category": "industry_trends",
            "query": "AI computer vision kitchen waste tracking adoption trends restaurants",
            "rationale": "Track technology maturity and recent adoption trends in AI kitchen analytics.",
            "priority": "low",
        },
    ],
    "research_gaps": [
        "Unit economics of deploying smart bin hardware in mid-sized Indian kitchens",
        "Integration landscape with leading Indian POS vendors such as Petpooja and Posist",
        "Typical ROI payback period expected by Indian restaurant owners",
    ],
    "assumptions_to_validate": [
        "Mid-sized restaurants and cloud kitchens in India view food waste tracking as a standalone paid need",
        "Kitchen staff can reliably adopt smart bin workflows during peak service hours",
        "AI-driven purchasing recommendations produce measurable food cost reductions within 60-90 days",
    ],
}


class _FakeGeminiProvider(GeminiProvider):
    """Deterministic GeminiProvider double supporting both IdeaBrief and ResearchQueryPlan."""

    def __init__(
        self,
        *,
        brief_response: str | None = None,
        plan_response: str | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(api_key="test-key", model="gemini-3.5-flash-lite")
        self.brief_response = brief_response
        self.plan_response = plan_response
        self.error = error
        self.calls: list[dict] = []

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction=None,
        response_schema=None,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "response_schema": response_schema,
                "temperature": temperature,
            }
        )
        if self.error is not None:
            raise self.error
        if response_schema is IdeaBrief and self.brief_response is not None:
            return self.brief_response
        if response_schema is ResearchQueryPlan and self.plan_response is not None:
            return self.plan_response
        raise AssertionError(f"Unexpected response_schema: {response_schema}")


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def test_research_query_plan_schema_validates_category_counts_and_uniqueness():
    # Ensure schema converts cleanly for google-genai structured output
    from google.genai import _transformers

    transformed = _transformers.t_schema(None, ResearchQueryPlan)
    assert transformed is not None

    plan = ResearchQueryPlan.model_validate(RESTAURANT_FOOD_WASTE_PLAN)
    counts = Counter(q.category for q in plan.queries)
    assert set(counts.keys()) == set(ResearchCategory)
    assert all(2 <= count <= 3 for count in counts.values())

    # Reject plan missing a category (only 1 query in industry_trends)
    invalid_counts_payload = json.loads(json.dumps(RESTAURANT_FOOD_WASTE_PLAN))
    invalid_counts_payload["queries"] = [
        q for q in invalid_counts_payload["queries"] if q["category"] != "industry_trends"
    ] + [
        {
            "category": "industry_trends",
            "query": "AI food waste analytics trend India",
            "rationale": "Single trend query only.",
            "priority": "low",
        },
        {
            "category": "market_research",
            "query": "India hospitality tech adoption spending report",
            "rationale": "Extra market query.",
            "priority": "medium",
        },
    ]
    with pytest.raises(ValidationError):
        ResearchQueryPlan.model_validate(invalid_counts_payload)

    # Reject plan with >3 queries in one category
    too_many_payload = json.loads(json.dumps(RESTAURANT_FOOD_WASTE_PLAN))
    too_many_payload["queries"].extend(
        [
            {
                "category": "market_research",
                "query": "India quick service restaurant market size report",
                "rationale": "Third market query.",
                "priority": "medium",
            },
            {
                "category": "market_research",
                "query": "India commercial kitchen food waste volume statistics",
                "rationale": "Fourth market query exceeds max 3.",
                "priority": "low",
            },
        ]
    )
    with pytest.raises(ValidationError):
        ResearchQueryPlan.model_validate(too_many_payload)

    # Reject duplicate queries (case-insensitive)
    dup_payload = json.loads(json.dumps(RESTAURANT_FOOD_WASTE_PLAN))
    dup_payload["queries"][1]["query"] = dup_payload["queries"][0]["query"].upper()
    with pytest.raises(ValidationError):
        ResearchQueryPlan.model_validate(dup_payload)


def test_query_planning_prompt_contains_all_categories_and_guardrails():
    brief = IdeaBrief.model_validate(RESTAURANT_FOOD_WASTE_BRIEF)
    user_prompt = build_query_planning_prompt(brief)

    for category in ResearchCategory:
        assert category.value in QUERY_PLANNER_SYSTEM_PROMPT
        assert category.value in user_prompt

    assert "2 or 3" in QUERY_PLANNER_SYSTEM_PROMPT
    assert "target_geography" in QUERY_PLANNER_SYSTEM_PROMPT
    assert "research_gaps" in QUERY_PLANNER_SYSTEM_PROMPT
    assert "assumptions_to_validate" in QUERY_PLANNER_SYSTEM_PROMPT
    assert "India" in user_prompt
    assert brief.industry in user_prompt


def test_standalone_query_planner_accepts_idea_brief_and_preserves_geography():
    brief = IdeaBrief.model_validate(RESTAURANT_FOOD_WASTE_BRIEF)

    # Simulate model abbreviating geography to 'IN' to verify verbatim preservation
    modified_plan = json.loads(json.dumps(RESTAURANT_FOOD_WASTE_PLAN))
    modified_plan["target_geography"] = "IN"

    fake_provider = _FakeGeminiProvider(plan_response=json.dumps(modified_plan))
    plan = plan_research_queries(brief, provider=fake_provider)

    assert isinstance(plan, ResearchQueryPlan)
    assert plan.target_geography == "India"
    assert len(plan.queries) == 10
    assert len(plan.research_gaps) >= 1
    assert len(plan.assumptions_to_validate) >= 1
    assert len(fake_provider.calls) == 1
    assert fake_provider.calls[0]["response_schema"] is ResearchQueryPlan


def test_query_planner_parses_markdown_fenced_json_and_handles_invalid_outputs():
    brief = IdeaBrief.model_validate(RESTAURANT_FOOD_WASTE_BRIEF)

    fenced = "```json\n" + json.dumps(RESTAURANT_FOOD_WASTE_PLAN) + "\n```"
    plan = QueryPlanner(provider=_FakeGeminiProvider(plan_response=fenced)).plan(brief)
    assert plan.target_geography == "India"

    with pytest.raises(InvalidQueryPlanOutputError):
        QueryPlanner(provider=_FakeGeminiProvider(plan_response="{not json")).plan(brief)

    with pytest.raises(InvalidQueryPlanOutputError):
        QueryPlanner(provider=_FakeGeminiProvider(plan_response='["not-an-object"]')).plan(brief)

    with pytest.raises(QueryPlanValidationError):
        QueryPlanner(
            provider=_FakeGeminiProvider(plan_response=json.dumps({"target_geography": "India"}))
        ).plan(brief)


def test_plan_research_endpoint_reuses_shared_gemini_provider_and_idea_analyzer():
    fake_provider = _FakeGeminiProvider(
        brief_response=json.dumps(RESTAURANT_FOOD_WASTE_BRIEF),
        plan_response=json.dumps(RESTAURANT_FOOD_WASTE_PLAN),
    )
    app.dependency_overrides[get_gemini_provider] = lambda: fake_provider

    response = client.post("/api/v1/ideas/plan-research", json=RESTAURANT_FOOD_WASTE_INPUT)
    assert response.status_code == 200

    parsed = ResearchPlanResponse.model_validate(response.json())
    assert parsed.idea_brief.target_geography == "India"
    assert parsed.research_plan.target_geography == "India"
    assert len(parsed.research_plan.queries) == 10

    # Confirm a single shared GeminiProvider handled both steps in sequence
    assert len(fake_provider.calls) == 2
    assert fake_provider.calls[0]["response_schema"] is IdeaBrief
    assert fake_provider.calls[1]["response_schema"] is ResearchQueryPlan


@pytest.mark.parametrize(
    ("error_to_raise", "expected_status"),
    [
        (GeminiConfigError("missing key"), 503),
        (GeminiTimeoutError("timeout"), 504),
        (GeminiUnavailableError("unavailable"), 503),
        (GeminiInvalidResponseError("empty"), 502),
    ],
)
def test_plan_research_endpoint_maps_provider_errors_safely(error_to_raise, expected_status):
    fake_provider = _FakeGeminiProvider(error=error_to_raise)
    app.dependency_overrides[get_llm_service] = lambda: LLMFallbackService(
        primary_provider=fake_provider,
        fallback_provider=None,
        enable_fallback=False,
    )

    resp = client.post("/api/v1/ideas/plan-research", json=RESTAURANT_FOOD_WASTE_INPUT)
    assert resp.status_code == expected_status
    assert "Traceback" not in resp.json()["detail"]


def test_plan_research_endpoint_maps_planner_invalid_output_and_422():
    # Valid IdeaBrief, invalid QueryPlanner JSON -> 502
    app.dependency_overrides[get_idea_analyzer] = lambda: IdeaAnalyzer(
        provider=_FakeGeminiProvider(brief_response=json.dumps(RESTAURANT_FOOD_WASTE_BRIEF))
    )
    app.dependency_overrides[get_query_planner] = lambda: QueryPlanner(
        provider=_FakeGeminiProvider(plan_response="{broken json")
    )
    r_bad_json = client.post("/api/v1/ideas/plan-research", json=RESTAURANT_FOOD_WASTE_INPUT)
    assert r_bad_json.status_code == 502

    # Valid IdeaBrief, incomplete QueryPlanner schema -> 502
    app.dependency_overrides[get_query_planner] = lambda: QueryPlanner(
        provider=_FakeGeminiProvider(plan_response=json.dumps({"target_geography": "India", "queries": []}))
    )
    r_bad_schema = client.post("/api/v1/ideas/plan-research", json=RESTAURANT_FOOD_WASTE_INPUT)
    assert r_bad_schema.status_code == 502

    # Client validation error -> 422
    r_short = client.post(
        "/api/v1/ideas/plan-research",
        json={"idea": "Too short", "target_geography": "India"},
    )
    assert r_short.status_code == 422


@pytest.mark.live
@requires_live_api
def test_live_plan_research_for_restaurant_food_waste_india():
    """Live end-to-end check: IdeaAnalyzer -> QueryPlanner for restaurant food waste startup in India."""
    if BACKEND_ENV_PATH.exists():
        load_dotenv(dotenv_path=BACKEND_ENV_PATH, override=False)

    provider = GeminiProvider()
    brief = IdeaAnalyzer(provider=provider).analyze(
        StartupIdeaInput(**RESTAURANT_FOOD_WASTE_INPUT)
    )
    plan = QueryPlanner(provider=provider).plan(brief)

    assert plan.target_geography == "India"
    counts = Counter(q.category for q in plan.queries)
    assert set(counts.keys()) == set(ResearchCategory)
    assert all(2 <= count <= 3 for count in counts.values())
    assert len(plan.research_gaps) > 0
    assert len(plan.assumptions_to_validate) > 0

    print("\n=== LIVE EXAMPLE OUTPUT (Restaurant Food Waste Startup in India) ===")
    print(json.dumps({"idea_brief": brief.model_dump(mode="json"), "research_plan": plan.model_dump(mode="json")}, indent=2))
