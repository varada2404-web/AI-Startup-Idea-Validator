"""Mocked tests for Gemini -> Groq provider fallback, GroqProvider, and safe error handling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.agents.web_search.idea_analyzer import IdeaAnalyzer, IdeaValidationError
from app.agents.web_search.query_planner import QueryPlanner, QueryPlanValidationError
from app.main import app, get_gemini_provider, get_groq_provider
from app.providers.gemini import (
    GeminiInvalidResponseError,
    GeminiProvider,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from app.providers.groq import (
    GroqAuthError,
    GroqConfigError,
    GroqInvalidResponseError,
    GroqProvider,
    GroqRateLimitError,
    GroqTimeoutError,
    GroqUnavailableError,
)
from app.providers.llm_router import (
    LLMAllProvidersFailedError,
    LLMFallbackService,
    LLMOutputValidationError,
)
from app.schemas.idea import IdeaBrief, StartupIdeaInput
from app.schemas.research import ResearchPlanResponse, ResearchQueryPlan

client = TestClient(app)

SENTINEL_GEMINI_KEY = "AIza-SENTINEL-GEMINI-DO-NOT-LEAK-1111"
SENTINEL_GROQ_KEY = "gsk_SENTINEL-GROQ-DO-NOT-LEAK-2222"

SAMPLE_IDEA_INPUT = {
    "idea": (
        "An AI-powered kitchen analytics and smart bin monitoring platform that helps mid-sized "
        "restaurants in India cut daily food waste through a monthly SaaS subscription."
    ),
    "target_geography": "India",
}

VALID_IDEA_BRIEF_DICT = {
    "problem": "Mid-sized Indian restaurants lose margin due to unmeasured daily kitchen food waste.",
    "solution": "AI kitchen analytics paired with smart bin monitoring to optimize purchasing.",
    "target_customers": ["Mid-sized restaurants in India", "Cloud kitchens in India"],
    "industry": "FoodTech / Restaurant Operations SaaS",
    "business_model": "Monthly SaaS subscription",
    "target_geography": "India",
    "missing_information": ["Hardware cost per kitchen", "POS integration scope"],
    "assumptions": ["Kitchen staff will consistently use smart bins during peak shifts"],
}

VALID_RESEARCH_PLAN_DICT = {
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
            "query": "cloud kitchens market India food cost percentage report",
            "rationale": "Quantify target segment economics.",
            "priority": "high",
        },
        {
            "category": "competitor_research",
            "query": "restaurant food waste tracking software competitors India",
            "rationale": "Identify direct competitors operating in India.",
            "priority": "high",
        },
        {
            "category": "competitor_research",
            "query": "Petpooja Posist inventory waste management features India",
            "rationale": "Evaluate incumbent POS platform overlap.",
            "priority": "medium",
        },
        {
            "category": "customer_pain_points",
            "query": "Indian restaurant owners kitchen food waste leakage challenges",
            "rationale": "Validate severity of kitchen waste pain points.",
            "priority": "high",
        },
        {
            "category": "customer_pain_points",
            "query": "commercial kitchen staff adoption barriers monitoring hardware India",
            "rationale": "Identify workflow adoption barriers.",
            "priority": "medium",
        },
        {
            "category": "pricing_and_business_models",
            "query": "restaurant inventory SaaS pricing INR per outlet India",
            "rationale": "Benchmark willingness to pay in India.",
            "priority": "high",
        },
        {
            "category": "pricing_and_business_models",
            "query": "smart bin AI food waste hardware plus SaaS subscription pricing",
            "rationale": "Check how competitors monetize hardware and software.",
            "priority": "medium",
        },
        {
            "category": "industry_trends",
            "query": "FSSAI food waste sustainability hospitality trends India",
            "rationale": "Identify regulatory and sustainability trends.",
            "priority": "medium",
        },
        {
            "category": "industry_trends",
            "query": "AI computer vision kitchen waste analytics restaurant adoption",
            "rationale": "Track technology adoption trends.",
            "priority": "low",
        },
    ],
    "research_gaps": ["Smart bin unit economics in India"],
    "assumptions_to_validate": ["Indian restaurants will pay monthly SaaS fees for waste tracking"],
}


class _StubProvider:
    """Configurable stub matching StructuredLLMProvider protocol."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        responses_by_schema: dict | None = None,
        raw_response: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.provider_name = provider_name
        self._model_name = model_name
        self.responses_by_schema = responses_by_schema or {}
        self.raw_response = raw_response
        self.error = error
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
        self.calls.append({"prompt": prompt, "response_schema": response_schema})
        if self.error is not None:
            raise self.error
        if response_schema in self.responses_by_schema:
            return self.responses_by_schema[response_schema]
        if self.raw_response is not None:
            return self.raw_response
        raise AssertionError(f"No stubbed response configured for {self.provider_name}")


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. GEMINI SUCCESS -> GROQ MUST NOT BE CALLED
# ---------------------------------------------------------------------------


def test_gemini_success_does_not_call_groq_for_analyzer_and_planner():
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_IDEA_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_RESEARCH_PLAN_DICT),
        },
    )
    groq_stub = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        error=AssertionError("Groq fallback must not be called when Gemini succeeds"),
    )

    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub

    resp_analyze = client.post("/api/v1/ideas/analyze", json=SAMPLE_IDEA_INPUT)
    assert resp_analyze.status_code == 200
    assert IdeaBrief.model_validate(resp_analyze.json()).target_geography == "India"
    assert resp_analyze.headers["X-LLM-Provider"] == "gemini"
    assert resp_analyze.headers["X-LLM-Model"] == "gemini-3.5-flash-lite"
    assert resp_analyze.headers["X-LLM-Fallback-Used"] == "false"

    resp_plan = client.post("/api/v1/ideas/plan-research", json=SAMPLE_IDEA_INPUT)
    assert resp_plan.status_code == 200
    parsed = ResearchPlanResponse.model_validate(resp_plan.json())
    assert parsed.research_plan.target_geography == "India"
    assert resp_plan.headers["X-LLM-Provider"] == "gemini"
    assert resp_plan.headers["X-LLM-Fallback-Used"] == "false"

    assert len(gemini_stub.calls) == 3
    assert len(groq_stub.calls) == 0


# ---------------------------------------------------------------------------
# 2-4. GEMINI TIMEOUT, RATE LIMIT, TEMPORARY SERVER ERROR -> GROQ SUCCEEDS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gemini_error", "expected_reason"),
    [
        (GeminiTimeoutError("Gemini API request timed out."), "timeout"),
        (GeminiRateLimitError("Gemini API rate limit exceeded."), "rate_limit"),
        (GeminiUnavailableError("Gemini 503 temporary high demand."), "unavailable"),
    ],
)
def test_gemini_retryable_errors_fallback_to_groq_and_succeed(gemini_error, expected_reason):
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        error=gemini_error,
    )
    groq_stub = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_IDEA_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_RESEARCH_PLAN_DICT),
        },
    )

    service = LLMFallbackService(primary_provider=gemini_stub, fallback_provider=groq_stub)
    analyzer = IdeaAnalyzer(llm_service=service)
    brief = analyzer.analyze(StartupIdeaInput(**SAMPLE_IDEA_INPUT))

    assert isinstance(brief, IdeaBrief)
    assert brief.target_geography == "India"
    assert analyzer.last_execution_metadata is not None
    assert analyzer.last_execution_metadata.provider == "groq"
    assert analyzer.last_execution_metadata.model == "openai/gpt-oss-20b"
    assert analyzer.last_execution_metadata.fallback_used is True
    assert analyzer.last_execution_metadata.attempted_providers == ["gemini", "groq"]
    assert analyzer.last_execution_metadata.fallback_reason == expected_reason

    # Verify QueryPlanner reuses the exact same fallback behavior
    planner = QueryPlanner(llm_service=service)
    plan = planner.plan(brief)
    assert isinstance(plan, ResearchQueryPlan)
    assert plan.target_geography == "India"
    assert planner.last_execution_metadata is not None
    assert planner.last_execution_metadata.provider == "groq"
    assert planner.last_execution_metadata.fallback_used is True
    assert planner.last_execution_metadata.fallback_reason == expected_reason


def test_endpoint_exposes_safe_fallback_metadata_headers_when_groq_recovers():
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        error=GeminiUnavailableError("503 UNAVAILABLE"),
    )
    groq_stub = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        responses_by_schema={
            IdeaBrief: json.dumps(VALID_IDEA_BRIEF_DICT),
            ResearchQueryPlan: json.dumps(VALID_RESEARCH_PLAN_DICT),
        },
    )
    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub

    resp = client.post("/api/v1/ideas/plan-research", json=SAMPLE_IDEA_INPUT)
    assert resp.status_code == 200
    assert resp.headers["X-LLM-Provider"] == "groq"
    assert resp.headers["X-LLM-Model"] == "openai/gpt-oss-20b"
    assert resp.headers["X-LLM-Fallback-Used"] == "true"


# ---------------------------------------------------------------------------
# 5. GEMINI INVALID STRUCTURED OUTPUT -> CONTROLLED FALLBACK TO GROQ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gemini_kwargs", "expected_reason"),
    [
        ({"error": GeminiInvalidResponseError("Empty response")}, "invalid_output"),
        ({"raw_response": "{not valid json"}, "invalid_output"),
        ({"raw_response": json.dumps({"problem": "Missing required fields"})}, "validation_error"),
    ],
)
def test_gemini_invalid_structured_output_triggers_controlled_fallback(gemini_kwargs, expected_reason):
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        **gemini_kwargs,
    )
    groq_stub = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        responses_by_schema={IdeaBrief: json.dumps(VALID_IDEA_BRIEF_DICT)},
    )

    service = LLMFallbackService(
        primary_provider=gemini_stub,
        fallback_provider=groq_stub,
        fallback_on_invalid_output=True,
    )
    analyzer = IdeaAnalyzer(llm_service=service)
    brief = analyzer.analyze(StartupIdeaInput(**SAMPLE_IDEA_INPUT))

    assert isinstance(brief, IdeaBrief)
    assert analyzer.last_execution_metadata is not None
    assert analyzer.last_execution_metadata.provider == "groq"
    assert analyzer.last_execution_metadata.fallback_used is True
    assert analyzer.last_execution_metadata.fallback_reason == expected_reason


# ---------------------------------------------------------------------------
# 6. BOTH PROVIDERS FAIL -> SAFE STRUCTURED ERROR
# ---------------------------------------------------------------------------


def test_both_providers_fail_returns_safe_structured_error_without_secrets():
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        error=GeminiTimeoutError("Gemini API request timed out."),
    )
    groq_stub = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        error=GroqUnavailableError("Groq API unavailable."),
    )
    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub

    # Direct service check
    service = LLMFallbackService(primary_provider=gemini_stub, fallback_provider=groq_stub)
    with pytest.raises(LLMAllProvidersFailedError) as exc_info:
        service.generate_structured(
            prompt="test",
            system_instruction="Return JSON",
            response_schema=IdeaBrief,
        )
    assert exc_info.value.attempts == [
        {"provider": "gemini", "error_type": "timeout"},
        {"provider": "groq", "error_type": "unavailable"},
    ]

    # Endpoint check
    resp = client.post("/api/v1/ideas/analyze", json=SAMPLE_IDEA_INPUT)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["error"] == "all_llm_providers_failed"
    assert detail["attempts"] == [
        {"provider": "gemini", "error_type": "timeout"},
        {"provider": "groq", "error_type": "unavailable"},
    ]
    assert SENTINEL_GEMINI_KEY not in resp.text
    assert SENTINEL_GROQ_KEY not in resp.text
    assert "Traceback" not in resp.text


# ---------------------------------------------------------------------------
# 7. GROQ INVALID OUTPUT -> SCHEMA VALIDATION FAILURE
# ---------------------------------------------------------------------------


def test_groq_invalid_output_fails_schema_validation_cleanly():
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        error=GeminiRateLimitError("429 rate limited"),
    )
    groq_bad_brief = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        raw_response=json.dumps({"problem": "Incomplete IdeaBrief missing required fields"}),
    )

    service = LLMFallbackService(primary_provider=gemini_stub, fallback_provider=groq_bad_brief)
    analyzer = IdeaAnalyzer(llm_service=service)
    with pytest.raises(IdeaValidationError):
        analyzer.analyze(StartupIdeaInput(**SAMPLE_IDEA_INPUT))

    # Also verify QueryPlanner rejects invalid ResearchQueryPlan from Groq
    groq_bad_plan = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        raw_response=json.dumps({"target_geography": "India", "industry": "FoodTech", "queries": []}),
    )
    planner = QueryPlanner(
        llm_service=LLMFallbackService(primary_provider=gemini_stub, fallback_provider=groq_bad_plan)
    )
    with pytest.raises(QueryPlanValidationError):
        planner.plan(IdeaBrief.model_validate(VALID_IDEA_BRIEF_DICT))

    # And via FastAPI endpoint -> HTTP 502
    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_bad_brief
    resp = client.post("/api/v1/ideas/analyze", json=SAMPLE_IDEA_INPUT)
    assert resp.status_code == 502
    assert "Traceback" not in resp.text


# ---------------------------------------------------------------------------
# 8. INVALID USER INPUT & UNRELATED PROGRAMMING ERRORS -> NO FALLBACK
# ---------------------------------------------------------------------------


def test_invalid_user_input_and_programming_errors_do_not_trigger_fallback():
    gemini_stub = _StubProvider(
        provider_name="gemini",
        model_name="gemini-3.5-flash-lite",
        error=RuntimeError("Unrelated bug inside code"),
    )
    groq_stub = _StubProvider(
        provider_name="groq",
        model_name="openai/gpt-oss-20b",
        responses_by_schema={IdeaBrief: json.dumps(VALID_IDEA_BRIEF_DICT)},
    )
    app.dependency_overrides[get_gemini_provider] = lambda: gemini_stub
    app.dependency_overrides[get_groq_provider] = lambda: groq_stub

    # Invalid user input -> FastAPI 422 before calling any LLM
    resp_analyze = client.post(
        "/api/v1/ideas/analyze",
        json={"idea": "Too short", "target_geography": "India"},
    )
    assert resp_analyze.status_code == 422

    resp_plan = client.post(
        "/api/v1/ideas/plan-research",
        json={"idea": "Too short", "target_geography": "I"},
    )
    assert resp_plan.status_code == 422
    assert len(gemini_stub.calls) == 0
    assert len(groq_stub.calls) == 0

    # Invalid argument type to IdeaAnalyzer / QueryPlanner -> TypeError, zero calls
    service = LLMFallbackService(primary_provider=gemini_stub, fallback_provider=groq_stub)
    with pytest.raises(TypeError):
        IdeaAnalyzer(llm_service=service).analyze("not-a-model")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        QueryPlanner(llm_service=service).plan("not-a-brief")  # type: ignore[arg-type]
    assert len(groq_stub.calls) == 0

    # Unrelated RuntimeError from primary -> propagates immediately, Groq NOT called
    with pytest.raises(RuntimeError, match="Unrelated bug"):
        service.generate_structured(
            prompt="test",
            system_instruction="Return JSON",
            response_schema=IdeaBrief,
        )
    assert len(groq_stub.calls) == 0


# ---------------------------------------------------------------------------
# PROVIDER UNIT TESTS (GEMINI RATE LIMIT & GROQ PROVIDER)
# ---------------------------------------------------------------------------


def test_gemini_provider_detects_rate_limit_without_leaking_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SENTINEL_GEMINI_KEY)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = genai_errors.APIError(
        429, {"error": {"message": "RESOURCE_EXHAUSTED: quota exceeded"}}
    )

    with patch("app.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider()
        with pytest.raises(GeminiRateLimitError) as exc_info:
            provider.generate_json("prompt")
        assert SENTINEL_GEMINI_KEY not in str(exc_info.value)


def test_groq_provider_config_success_and_errors_without_leaking_key(monkeypatch, caplog):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqConfigError):
        GroqProvider().generate_json("prompt")

    monkeypatch.setenv("GROQ_API_KEY", SENTINEL_GROQ_KEY)
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "20")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
    )

    with patch("app.providers.groq.Groq", return_value=mock_client) as mock_cls:
        provider = GroqProvider()
        raw = provider.generate_json("Return JSON", system_instruction="Analyze idea")
        assert raw == '{"ok": true}'
        _, init_kwargs = mock_cls.call_args
        assert init_kwargs["api_key"] == SENTINEL_GROQ_KEY
        assert init_kwargs["timeout"] == 20.0

        # Timeout
        mock_client.chat.completions.create.side_effect = httpx.ReadTimeout("timeout")
        with pytest.raises(GroqTimeoutError) as e_to:
            provider.generate_json("prompt")
        assert SENTINEL_GROQ_KEY not in str(e_to.value)

        # Rate limit
        fake_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        fake_429 = httpx.Response(429, request=fake_req)
        mock_client.chat.completions.create.side_effect = groq.RateLimitError(
            "rate limit", response=fake_429, body=None
        )
        with pytest.raises(GroqRateLimitError) as e_rl:
            provider.generate_json("prompt")
        assert SENTINEL_GROQ_KEY not in str(e_rl.value)

        # Auth error
        fake_401 = httpx.Response(401, request=fake_req)
        mock_client.chat.completions.create.side_effect = groq.AuthenticationError(
            "unauthorized", response=fake_401, body=None
        )
        with pytest.raises(GroqAuthError) as e_auth:
            provider.generate_json("prompt")
        assert SENTINEL_GROQ_KEY not in str(e_auth.value)

        # Empty response
        mock_client.chat.completions.create.side_effect = None
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
        )
        with pytest.raises(GroqInvalidResponseError):
            provider.generate_json("prompt")

    assert SENTINEL_GROQ_KEY not in caplog.text
