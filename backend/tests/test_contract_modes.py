"""Contract and configuration coverage for demo and future live mode."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.core.config import ApplicationConfig, load_config
from app.main import create_app
from app.schemas.contracts import (
    IdeaBrief,
    ProviderError,
    Source,
    ValidationAccepted,
    ValidationRequest,
    ValidationRun,
)


def make_run(**overrides: object) -> ValidationRun:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "run_id": "contract-run",
        "created_at": now,
        "updated_at": now,
        "request": ValidationRequest(idea="A useful tool for small local businesses."),
    }
    values.update(overrides)
    return ValidationRun(**values)


def test_legacy_demo_defaults_remain_compatible() -> None:
    run = make_run()
    accepted = ValidationAccepted(run_id=run.run_id, status="queued")
    source = Source(
        id="S1",
        url="mock://demo.invalid/market",
        domain="demo.invalid",
        title="Synthetic market record",
        query="market query",
        categories=["market"],
        cluster_id="C1",
        text="Synthetic placeholder text",
    )

    assert run.mode == "demo" and run.demo_mode is True
    assert accepted.mode == "demo" and accepted.demo_mode is True
    assert source.source_type == "mock" and source.simulated is True
    assert source.extraction_status == "not_attempted"
    assert source.snippet is None  # Old fixtures need not provide new optional fields.


def test_live_shaped_idea_source_and_run_contracts() -> None:
    request = ValidationRequest(idea="A useful tool for small local businesses.")
    brief = IdeaBrief(
        original_text=request.idea,
        problem="Small teams report a workflow burden.",
        solution="A shared workflow tool.",
        target_customer={"value": "small businesses", "origin": "inferred"},
        geography={"value": "India", "origin": "default"},
        industry="Business software",
        hypotheses=[
            {
                "id": f"H{number}",
                "kind": kind,
                "statement": f"Unassessed hypothesis {number}",
                "critical": number <= 2,
                "status": "unknown",
            }
            for number, kind in enumerate(
                ("demand", "willingness_to_pay", "competition", "feasibility", "timing"),
                start=1,
            )
        ],
        interpretation_mode="gemini_live",
    )
    source = Source(
        id="S1",
        url="https://example.org/research",
        domain="example.org",
        title="Research page",
        query="research query",
        categories=["market"],
        source_type="tavily",
        cluster_id="C1",
        text="Legacy-compatible source text",
        simulated=False,
        snippet="Provider search snippet.",
        extracted_text="Extracted page content.",
        extraction_status="succeeded",
    )
    error = ProviderError(
        provider="tavily",
        stage="search",
        code="rate_limited",
        message="Search provider temporarily unavailable.",
        retryable=True,
    )
    run = make_run(
        request=request,
        idea_brief=brief,
        mode="live",
        demo_mode=False,
        simulation_notice=None,
        quality="degraded",
        sources=[source],
        provider_errors=[error],
    )

    assert brief.interpretation_mode == "gemini_live"
    assert len(brief.hypotheses) == 5
    assert {hypothesis.status for hypothesis in brief.hypotheses} == {"unknown"}
    assert source.snippet != source.extracted_text
    assert run.mode == "live" and run.demo_mode is False
    assert run.quality == "degraded"
    assert run.provider_errors[0].provider == "tavily"


def test_source_rejects_inconsistent_provenance_and_extraction() -> None:
    common = {
        "id": "S1",
        "url": "https://example.org/research",
        "domain": "example.org",
        "title": "Research page",
        "query": "research query",
        "categories": ["market"],
        "cluster_id": "C1",
        "text": "Source text",
    }
    with pytest.raises(ValidationError):
        Source(**common, source_type="tavily", simulated=True)
    with pytest.raises(ValidationError):
        Source(**common, extraction_status="succeeded")
    with pytest.raises(ValidationError):
        Source(**common, extraction_status="failed", extracted_text="should not exist")


def test_live_configuration_requires_both_credentials_without_echoing_them() -> None:
    with pytest.raises(
        ValidationError, match="GEMINI_API_KEY, TAVILY_API_KEY, GEMINI_MODEL"
    ):
        load_config({"VALIDATOR_MODE": "live"})

    sentinel_secret = "never-print-this-test-secret"
    config = load_config(
        {
            "VALIDATOR_MODE": "live",
            "GEMINI_API_KEY": sentinel_secret,
            "TAVILY_API_KEY": "test-tavily-placeholder",
            "GEMINI_MODEL": "test-model",
        }
    )
    assert config.mode == "live"
    assert sentinel_secret not in repr(config)


def test_demo_configuration_and_provider_limits_are_explicit() -> None:
    config = load_config({})
    assert config.mode == "demo"
    assert config.max_search_results_per_query == 5
    assert config.max_follow_up_queries == 4
    assert config.max_extract_sources_per_run == 8
    assert config.max_concurrent_provider_calls == 4
    assert config.tavily_timeout_seconds == 20
    assert config.tavily_max_retries == 1

    with pytest.raises(ValidationError):
        ApplicationConfig(max_search_results_per_query=6)
    with pytest.raises(ValidationError):
        ApplicationConfig(max_extract_sources_per_run=9)


def test_tavily_configuration_loads_bounds_without_exposing_key() -> None:
    config = load_config(
        {
            "VALIDATOR_MODE": "live",
            "GEMINI_API_KEY": "gemini-placeholder",
            "GEMINI_MODEL": "model-placeholder",
            "TAVILY_API_KEY": "tavily-placeholder",
            "TAVILY_TIMEOUT_SECONDS": "11",
            "TAVILY_MAX_RETRIES": "2",
            "TAVILY_MAX_RETRY_DELAY_SECONDS": "7",
        }
    )
    assert config.tavily_timeout_seconds == 11
    assert config.tavily_max_retries == 2
    assert config.tavily_max_retry_delay_seconds == 7
    assert "tavily-placeholder" not in repr(config)


def test_live_application_selects_live_providers(tmp_path) -> None:
    config = ApplicationConfig(
        mode="live",
        gemini_api_key="test-gemini-placeholder",
        tavily_api_key="test-tavily-placeholder",
        gemini_model="test-model",
    )
    app = create_app(tmp_path / "live.sqlite3", config=config)
    assert app.state.pipeline_runner.provider.__class__.__name__ == "TavilyProvider"
    assert app.state.pipeline_runner.interpreter.__class__.__name__ == (
        "GeminiIdeaUnderstandingProvider"
    )
