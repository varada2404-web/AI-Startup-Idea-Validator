"""API integration verification tests (mocked and live) for Gemini, Groq, and Tavily.

Mocked tests run by default with `pytest`.
Live API tests are marked with `@pytest.mark.live` and only execute when
`RUN_LIVE_API_TESTS=1` is set in the environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

from app.agents.web_search.idea_analyzer import IdeaAnalyzer
from app.providers.gemini import DEFAULT_GEMINI_MODEL, GeminiProvider
from app.schemas.idea import IdeaBrief, StartupIdeaInput

BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_API_TESTS", "").strip().lower() in {"1", "true", "yes"}
requires_live_api = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Live API tests are disabled by default. Set RUN_LIVE_API_TESTS=1 to run live checks.",
)


class GroqStructuredResponse(BaseModel):
    """Schema used to validate structured JSON output from Groq."""

    industry: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    key_risks: list[str] = Field(..., min_length=1)


class TavilySearchResult(BaseModel):
    """Schema used to validate individual Tavily search results."""

    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


def _load_backend_env() -> None:
    """Load backend/.env without overriding already-set process environment variables."""
    if BACKEND_ENV_PATH.exists():
        load_dotenv(dotenv_path=BACKEND_ENV_PATH, override=False)


def _run_groq_structured_check(api_key: str, model: str) -> tuple[str, GroqStructuredResponse]:
    """Check Groq model availability/connectivity, then send and validate a structured JSON request."""
    client = Groq(api_key=api_key, timeout=30.0)
    models_page = client.models.list()
    available_ids = {m.id for m in models_page.data}
    assert model in available_ids, f"Configured Groq model '{model}' not found in available models."

    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY a valid JSON object with keys: "
                    "industry (non-empty string), summary (non-empty string), "
                    "key_risks (non-empty array of strings)."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Briefly analyze a startup building AI-assisted invoice reconciliation "
                    "for small logistics operators in India."
                ),
            },
        ],
    )
    raw_content = completion.choices[0].message.content
    assert isinstance(raw_content, str) and raw_content.strip()
    parsed = json.loads(raw_content)
    validated = GroqStructuredResponse.model_validate(parsed)
    used_model = getattr(completion, "model", None) or model
    return used_model, validated


def _run_tavily_market_search(api_key: str, query: str) -> list[TavilySearchResult]:
    """Perform a Tavily web search and validate returned titles, URLs, and content."""
    response = httpx.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    raw_results = payload.get("results", [])
    assert isinstance(raw_results, list) and len(raw_results) > 0
    return [TavilySearchResult.model_validate(item) for item in raw_results]


# ---------------------------------------------------------------------------
# MOCKED INTEGRATION TESTS (safe, deterministic, run without external network)
# ---------------------------------------------------------------------------


def test_mocked_env_loading(tmp_path, monkeypatch):
    """Verify dotenv loads required keys from a .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=mock-gemini\n"
        f"GEMINI_MODEL={DEFAULT_GEMINI_MODEL}\n"
        "GROQ_API_KEY=mock-groq\n"
        f"GROQ_MODEL={DEFAULT_GROQ_MODEL}\n"
        "TAVILY_API_KEY=mock-tavily\n",
        encoding="utf-8",
    )
    for var in ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    load_dotenv(dotenv_path=env_file, override=False)

    assert os.getenv("GEMINI_API_KEY") == "mock-gemini"
    assert os.getenv("GEMINI_MODEL") == DEFAULT_GEMINI_MODEL
    assert os.getenv("GROQ_API_KEY") == "mock-groq"
    assert os.getenv("GROQ_MODEL") == DEFAULT_GROQ_MODEL
    assert os.getenv("TAVILY_API_KEY") == "mock-tavily"


def test_mocked_gemini_provider_and_idea_analyzer(monkeypatch):
    """Mocked verification of GeminiProvider + IdeaAnalyzer returning a valid IdeaBrief."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    expected_brief = {
        "problem": "Small neighborhood retail stores lose margin due to manual stock tracking.",
        "solution": "Mobile app automating inventory updates via invoice scans and shelf photos.",
        "target_customers": ["Small neighborhood retail shop owners"],
        "industry": "Retail Tech",
        "business_model": "Monthly SaaS subscription",
        "target_geography": "India",
        "missing_information": ["Exact pricing tier", "POS integration requirements"],
        "assumptions": ["Shop owners use camera-equipped smartphones daily"],
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(expected_brief)
    )

    with patch("app.providers.gemini.genai.Client", return_value=mock_client):
        analyzer = IdeaAnalyzer(provider=GeminiProvider())
        brief = analyzer.analyze(
            StartupIdeaInput(
                idea="A mobile app helping small retail shops track inventory via invoice scans and shelf photos on a monthly SaaS subscription.",
                target_geography="India",
            )
        )

    assert isinstance(brief, IdeaBrief)
    assert brief.target_geography == "India"
    assert brief.industry == "Retail Tech"
    assert brief.business_model == "Monthly SaaS subscription"


def test_mocked_groq_structured_output():
    """Mocked verification of Groq structured JSON completion and Pydantic validation."""
    mock_completion = SimpleNamespace(
        model=DEFAULT_GROQ_MODEL,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "industry": "Logistics Tech",
                            "summary": "Automates freight invoice matching for small logistics firms.",
                            "key_risks": ["ERP integration complexity", "Low SMB software budgets"],
                        }
                    )
                )
            )
        ],
    )
    mock_groq_client = MagicMock()
    mock_groq_client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id=DEFAULT_GROQ_MODEL)]
    )
    mock_groq_client.chat.completions.create.return_value = mock_completion

    with patch(f"{__name__}.Groq", return_value=mock_groq_client):
        model_used, result = _run_groq_structured_check("mock-groq-key", DEFAULT_GROQ_MODEL)

    assert model_used == DEFAULT_GROQ_MODEL
    assert result.industry == "Logistics Tech"
    assert len(result.key_risks) == 2


def test_mocked_tavily_market_search():
    """Mocked verification of Tavily search parsing and result validation."""
    fake_payload = {
        "results": [
            {
                "title": "India Retail Tech Market Overview",
                "url": "https://example.com/india-retail-tech",
                "content": "Small retail stores in India are rapidly adopting mobile inventory SaaS tools.",
            }
        ]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = fake_payload
    mock_response.raise_for_status.return_value = None

    with patch(f"{__name__}.httpx.post", return_value=mock_response):
        results = _run_tavily_market_search(
            "mock-tavily-key", "small retail inventory management SaaS market India"
        )

    assert len(results) == 1
    assert results[0].title == "India Retail Tech Market Overview"
    assert results[0].url.startswith("https://")
    assert "inventory" in results[0].content.lower()


# ---------------------------------------------------------------------------
# LIVE API INTEGRATION TESTS (execute real external API requests)
# ---------------------------------------------------------------------------


@pytest.mark.live
@requires_live_api
def test_live_env_vars_loaded():
    """Verify real API keys load from backend/.env without exposing secret values."""
    _load_backend_env()
    assert bool(os.getenv("GEMINI_API_KEY", "").strip()), "GEMINI_API_KEY is not loaded from .env"
    assert bool(os.getenv("GROQ_API_KEY", "").strip()), "GROQ_API_KEY is not loaded from .env"
    assert bool(os.getenv("TAVILY_API_KEY", "").strip()), "TAVILY_API_KEY is not loaded from .env"


@pytest.mark.live
@requires_live_api
def test_live_gemini_idea_analyzer():
    """Live API check: GeminiProvider + IdeaAnalyzer return valid JSON matching IdeaBrief."""
    _load_backend_env()
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    startup_input = StartupIdeaInput(
        idea=(
            "A mobile app that helps small neighborhood retail shops in India track inventory "
            "by scanning daily supplier invoices and shelf photos, priced as a monthly SaaS subscription."
        ),
        target_geography="India",
    )
    analyzer = IdeaAnalyzer(provider=GeminiProvider())
    brief = analyzer.analyze(startup_input)

    assert isinstance(brief, IdeaBrief)
    assert brief.target_geography == "India"
    assert len(brief.problem) > 0
    assert len(brief.solution) > 0
    assert len(brief.target_customers) > 0
    assert len(brief.industry) > 0
    assert isinstance(brief.missing_information, list)
    assert isinstance(brief.assumptions, list)
    print(f"\n[LIVE GEMINI] model={model} industry={brief.industry!r} business_model={brief.business_model!r}")


@pytest.mark.live
@requires_live_api
def test_live_groq_structured_output():
    """Live API check: Groq connectivity and structured JSON response validation."""
    _load_backend_env()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL

    model_used, result = _run_groq_structured_check(api_key=api_key, model=model)
    assert model_used
    assert isinstance(result, GroqStructuredResponse)
    assert len(result.key_risks) >= 1
    print(f"\n[LIVE GROQ] model={model_used} industry={result.industry!r} risks={len(result.key_risks)}")


@pytest.mark.live
@requires_live_api
def test_live_tavily_market_search():
    """Live API check: Tavily web search returns titles, URLs, and relevant content."""
    _load_backend_env()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    query = "small retail inventory management SaaS market India"

    results = _run_tavily_market_search(api_key=api_key, query=query)
    assert len(results) >= 1
    for item in results:
        assert item.url.startswith("http")
        assert len(item.title) > 0
        assert len(item.content) > 20
    print(f"\n[LIVE TAVILY] results={len(results)} top_title={results[0].title!r} top_url={results[0].url}")
