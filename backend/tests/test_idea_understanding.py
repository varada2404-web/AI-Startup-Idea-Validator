"""Tests for Stage 2: Gemini-powered Idea Understanding."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.agents.web_search.idea_analyzer import (
    IdeaAnalyzer,
    IdeaValidationError,
    InvalidModelOutputError,
)
from app.agents.web_search.prompts import (
    IDEA_UNDERSTANDING_SYSTEM_PROMPT,
    build_idea_understanding_prompt,
)
from app.main import app, get_idea_analyzer
from app.providers.gemini import (
    GeminiConfigError,
    GeminiInvalidResponseError,
    GeminiProvider,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from app.schemas.idea import IdeaBrief, StartupIdeaInput

client = TestClient(app)


THREE_STARTUP_TEST_CASES = [
    {
        "input": {
            "idea": (
                "A mobile app that lets small neighborhood grocery shop owners track stock levels "
                "by taking daily smartphone photos of their shelves instead of manual spreadsheets."
            ),
            "target_geography": "India",
        },
        "model_output": {
            "problem": "Small neighborhood grocery shop owners struggle with time-consuming manual inventory tracking using spreadsheets.",
            "solution": "A mobile application that automatically tracks shelf stock levels from daily smartphone photos.",
            "target_customers": ["Small neighborhood grocery shop owners"],
            "industry": "Retail Tech",
            "business_model": None,
            "target_geography": "India",
            "missing_information": [
                "Monetization strategy and pricing model",
                "Expected accuracy of photo-based stock counting in crowded shelves",
                "Customer acquisition and distribution channels",
            ],
            "assumptions": [
                "Shop owners have smartphones with cameras and are willing to take daily shelf photos",
                "Visual recognition can reliably distinguish local packaged goods",
            ],
        },
    },
    {
        "input": {
            "idea": (
                "A B2B SaaS platform with Bluetooth temperature sensors that alerts independent pharmacies "
                "before vaccine refrigerators drift out of regulatory compliance, billed as a monthly subscription."
            ),
            "target_geography": "Germany",
        },
        "model_output": {
            "problem": "Independent pharmacies risk vaccine spoilage and regulatory non-compliance when refrigerator temperatures drift undetected.",
            "solution": "Bluetooth temperature sensors paired with a B2B SaaS alerting platform for real-time compliance monitoring.",
            "target_customers": ["Independent pharmacies"],
            "industry": "HealthTech",
            "business_model": "Monthly B2B SaaS subscription",
            "target_geography": "DE",  # Intentionally abbreviated to test verbatim preservation
            "missing_information": [
                "Hardware sensor cost and whether hardware is bundled or sold upfront",
                "Specific German pharmacy regulatory certification requirements",
                "Target subscription price per pharmacy",
            ],
            "assumptions": [
                "Independent pharmacies lack automated compliance monitoring today",
                "Pharmacy staff will act promptly on mobile or web alerts",
            ],
        },
    },
    {
        "input": {
            "idea": (
                "A pay-as-you-go software platform for managing solar microgrid billing and predictive "
                "battery maintenance alerts for off-grid rural medical clinics."
            ),
            "target_geography": "Kenya",
        },
        "model_output": {
            "problem": "Off-grid rural medical clinics face unreliable solar power due to unpredicted battery failures and manual microgrid billing.",
            "solution": "A software platform providing solar microgrid billing and predictive battery maintenance alerts.",
            "target_customers": ["Off-grid rural medical clinics", "Solar microgrid operators"],
            "industry": "CleanTech / Energy",
            "business_model": "Pay-as-you-go",
            "target_geography": "Kenya",
            "missing_information": [
                "Integration method with existing inverters and battery management hardware",
                "Mobile money payment integration details",
                "Connectivity fallback in low-signal rural areas",
            ],
            "assumptions": [
                "Inferred that solar microgrid operators may also be direct software users alongside clinics",
                "Clinics have sufficient telemetry hardware to enable predictive maintenance alerts",
            ],
        },
    },
]


class _FakeGeminiProvider(GeminiProvider):
    """Deterministic provider double for testing IdeaAnalyzer and FastAPI integration."""

    def __init__(self, responses_by_substring: dict[str, str] | None = None, raw_response: str | None = None, error: Exception | None = None) -> None:
        super().__init__(api_key="test-key", model="gemini-2.5-flash")
        self.responses_by_substring = responses_by_substring or {}
        self.raw_response = raw_response
        self.error = error
        self.calls: list[dict] = []

    def generate_json(self, prompt: str, *, system_instruction=None, response_schema=None, temperature: float = 0.2) -> str:
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
        if self.raw_response is not None:
            return self.raw_response
        for key, value in self.responses_by_substring.items():
            if key in prompt:
                return value
        raise AssertionError(f"Unexpected prompt in fake provider: {prompt}")


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def test_existing_stage_1_endpoints_still_work():
    root = client.get("/")
    assert root.status_code == 200
    assert root.json() == {"message": "AI Startup Idea Validator API is running"}

    idea_resp = client.post(
        "/api/v1/ideas",
        json={
            "idea": "An AI assistant that drafts compliance checklists for small food exporters.",
            "target_geography": "India",
        },
    )
    assert idea_resp.status_code == 200
    assert idea_resp.json()["target_geography"] == "India"


def test_prompt_contains_required_guardrails():
    sample = StartupIdeaInput(
        idea="A booking and billing platform for independent physiotherapy clinics.",
        target_geography="United Kingdom",
    )
    user_prompt = build_idea_understanding_prompt(sample)

    assert "United Kingdom" in user_prompt
    assert sample.idea in user_prompt
    assert "Never invent" in IDEA_UNDERSTANDING_SYSTEM_PROMPT
    assert "business_model" in IDEA_UNDERSTANDING_SYSTEM_PROMPT
    assert "missing_information" in IDEA_UNDERSTANDING_SYSTEM_PROMPT
    assert "assumptions" in IDEA_UNDERSTANDING_SYSTEM_PROMPT
    assert "target_geography" in IDEA_UNDERSTANDING_SYSTEM_PROMPT


def test_analyze_three_distinct_startup_ideas_via_endpoint():
    response_map = {
        "grocery shop owners": json.dumps(THREE_STARTUP_TEST_CASES[0]["model_output"]),
        "vaccine refrigerators": json.dumps(THREE_STARTUP_TEST_CASES[1]["model_output"]),
        "solar microgrid": json.dumps(THREE_STARTUP_TEST_CASES[2]["model_output"]),
    }
    fake_provider = _FakeGeminiProvider(responses_by_substring=response_map)
    app.dependency_overrides[get_idea_analyzer] = lambda: IdeaAnalyzer(provider=fake_provider)

    briefs: list[IdeaBrief] = []
    for case in THREE_STARTUP_TEST_CASES:
        resp = client.post("/api/v1/ideas/analyze", json=case["input"])
        assert resp.status_code == 200
        brief = IdeaBrief.model_validate(resp.json())
        briefs.append(brief)

        # Geography preserved verbatim (including case 2 where model returned 'DE')
        assert brief.target_geography == case["input"]["target_geography"]
        # Missing information and assumptions are populated lists
        assert isinstance(brief.missing_information, list) and len(brief.missing_information) > 0
        assert isinstance(brief.assumptions, list) and len(brief.assumptions) > 0

    # Verify all three ideas produced distinct structured outputs
    assert len({b.problem for b in briefs}) == 3
    assert len({b.solution for b in briefs}) == 3
    assert len({b.industry for b in briefs}) == 3
    assert len({b.target_geography for b in briefs}) == 3

    # Verify business_model is None when unsupported and populated when stated
    assert briefs[0].business_model is None
    assert briefs[1].business_model == "Monthly B2B SaaS subscription"
    assert briefs[2].business_model == "Pay-as-you-go"

    # Verify structured schema was passed to Gemini provider
    assert all(call["response_schema"] is IdeaBrief for call in fake_provider.calls)


def test_analyzer_parses_markdown_fenced_json():
    fenced = "```json\n" + json.dumps(THREE_STARTUP_TEST_CASES[0]["model_output"]) + "\n```"
    analyzer = IdeaAnalyzer(provider=_FakeGeminiProvider(raw_response=fenced))
    brief = analyzer.analyze(StartupIdeaInput(**THREE_STARTUP_TEST_CASES[0]["input"]))
    assert brief.industry == "Retail Tech"
    assert brief.target_geography == "India"


def test_analyzer_rejects_invalid_json_and_schema_mismatch():
    sample = StartupIdeaInput(**THREE_STARTUP_TEST_CASES[0]["input"])

    bad_json_analyzer = IdeaAnalyzer(provider=_FakeGeminiProvider(raw_response="not valid json {"))
    with pytest.raises(InvalidModelOutputError):
        bad_json_analyzer.analyze(sample)

    non_object_analyzer = IdeaAnalyzer(provider=_FakeGeminiProvider(raw_response='["not", "an", "object"]'))
    with pytest.raises(InvalidModelOutputError):
        non_object_analyzer.analyze(sample)

    invalid_schema_analyzer = IdeaAnalyzer(
        provider=_FakeGeminiProvider(raw_response=json.dumps({"problem": "Only problem present"}))
    )
    with pytest.raises(IdeaValidationError):
        invalid_schema_analyzer.analyze(sample)


def test_gemini_provider_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider()
    with pytest.raises(GeminiConfigError):
        provider.generate_json("hello")


def test_gemini_provider_uses_env_model_and_structured_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-secret-key-do-not-leak")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "15")

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(THREE_STARTUP_TEST_CASES[0]["model_output"])
    )

    with patch("app.providers.gemini.genai.Client", return_value=mock_client) as mock_cls:
        provider = GeminiProvider()
        result = provider.generate_json(
            "analyze this",
            system_instruction="be precise",
            response_schema=IdeaBrief,
        )

    assert json.loads(result)["industry"] == "Retail Tech"
    _, client_kwargs = mock_cls.call_args
    assert client_kwargs["api_key"] == "fake-secret-key-do-not-leak"
    assert client_kwargs["http_options"].timeout == 15000

    _, call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs["model"] == "gemini-2.5-pro"
    assert call_kwargs["config"].response_mime_type == "application/json"
    assert call_kwargs["config"].response_schema is IdeaBrief


def test_gemini_provider_handles_timeout_api_error_and_empty_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-secret-key")

    mock_client = MagicMock()
    with patch("app.providers.gemini.genai.Client", return_value=mock_client):
        provider = GeminiProvider()

        mock_client.models.generate_content.side_effect = httpx.ReadTimeout("timed out")
        with pytest.raises(GeminiTimeoutError):
            provider.generate_json("prompt")

        mock_client.models.generate_content.side_effect = genai_errors.APIError(
            503, {"error": {"message": "Service unavailable"}}
        )
        with pytest.raises(GeminiUnavailableError):
            provider.generate_json("prompt")

        mock_client.models.generate_content.side_effect = None
        mock_client.models.generate_content.return_value = SimpleNamespace(text="   ")
        with pytest.raises(GeminiInvalidResponseError):
            provider.generate_json("prompt")


@pytest.mark.parametrize(
    ("error_to_raise", "expected_status"),
    [
        (GeminiConfigError("missing key"), 503),
        (GeminiTimeoutError("timeout"), 504),
        (GeminiUnavailableError("down"), 503),
        (GeminiInvalidResponseError("empty"), 502),
    ],
)
def test_endpoint_maps_provider_errors_safely(error_to_raise, expected_status):
    fake_provider = _FakeGeminiProvider(error=error_to_raise)
    app.dependency_overrides[get_idea_analyzer] = lambda: IdeaAnalyzer(provider=fake_provider)

    resp = client.post("/api/v1/ideas/analyze", json=THREE_STARTUP_TEST_CASES[0]["input"])
    assert resp.status_code == expected_status
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and len(detail) > 0
    assert "Traceback" not in detail


def test_endpoint_maps_invalid_json_and_validation_errors_to_502():
    app.dependency_overrides[get_idea_analyzer] = lambda: IdeaAnalyzer(
        provider=_FakeGeminiProvider(raw_response="{bad json")
    )
    r_bad_json = client.post("/api/v1/ideas/analyze", json=THREE_STARTUP_TEST_CASES[0]["input"])
    assert r_bad_json.status_code == 502

    app.dependency_overrides[get_idea_analyzer] = lambda: IdeaAnalyzer(
        provider=_FakeGeminiProvider(raw_response=json.dumps({"problem": "incomplete"}))
    )
    r_bad_schema = client.post("/api/v1/ideas/analyze", json=THREE_STARTUP_TEST_CASES[0]["input"])
    assert r_bad_schema.status_code == 502


def test_endpoint_rejects_invalid_input_with_422():
    r_short = client.post(
        "/api/v1/ideas/analyze",
        json={"idea": "Too short", "target_geography": "India"},
    )
    assert r_short.status_code == 422

    r_missing = client.post("/api/v1/ideas/analyze", json={})
    assert r_missing.status_code == 422
