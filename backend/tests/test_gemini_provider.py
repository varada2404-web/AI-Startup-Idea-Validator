"""Gemini adapter tests with an injected, network-free SDK-shaped client."""

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.core.config import ApplicationConfig, load_config
from app.pipeline.runner import PipelineRunner
from app.pipeline.store import RunStore
from app.providers.llm.errors import ProviderFailure
from app.providers.llm.factory import create_idea_understanding_provider
from app.providers.llm.gemini import (
    PROMPT_VERSION,
    GeminiIdeaUnderstandingProvider,
)
from app.providers.llm.mock import MockIdeaUnderstandingProvider
from app.providers.search.mock import MockSearchProvider
from app.schemas.contracts import ValidationRequest, ValidationRun

SENTINEL_SECRET = "gemini-test-secret-must-never-appear"


def live_config(**overrides: object) -> ApplicationConfig:
    values: dict[str, object] = {
        "mode": "live",
        "gemini_api_key": SENTINEL_SECRET,
        "tavily_api_key": "tavily-test-placeholder",
        "gemini_model": "model-test-only",
    }
    values.update(overrides)
    return ApplicationConfig(**values)


def valid_output(idea: str) -> dict[str, object]:
    return {
        "original_text": idea,
        "problem": "A possible workflow difficulty for the stated audience.",
        "solution": "The submitted idea may provide a workflow tool.",
        "target_customer": {"value": "small businesses", "origin": "inferred"},
        "geography": {"value": "India", "origin": "default"},
        "industry": "Business software",
        "business_model": "unclear",
        "hypotheses": [
            {
                "id": f"H{index}",
                "kind": kind,
                "statement": f"Unassessed {kind} hypothesis.",
                "critical": index <= 2,
                "status": "unknown",
            }
            for index, kind in enumerate(
                ("demand", "willingness_to_pay", "competition", "feasibility", "timing"),
                start=1,
            )
        ],
        "missing_info": ["Confirm the inferred audience and problem with the founder."],
        "search_keywords": ["unrelated", "model", "generated", "scope"],
        "interpretation_mode": "gemini_live",
    }


class FakeModels:
    def __init__(self, response: object = None, exception: Exception | None = None) -> None:
        self.response = response
        self.exception = exception
        self.calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.exception is not None:
            raise self.exception
        return self.response


class FakeSDKClient:
    def __init__(self, response: object = None, exception: Exception | None = None) -> None:
        self.models = FakeModels(response, exception)


def test_success_uses_one_structured_call_and_validates_output() -> None:
    idea = "A scheduling tool for small businesses with missed appointments."
    request = ValidationRequest(idea=idea)
    sdk = FakeSDKClient(SimpleNamespace(parsed=valid_output(idea), text=None))
    provider = GeminiIdeaUnderstandingProvider(live_config(), client=sdk)

    brief = asyncio.run(provider.extract_idea_brief(request))

    assert len(sdk.models.calls) == 1
    call = sdk.models.calls[0]
    assert call["model"] == "model-test-only"
    assert isinstance(call["contents"], str)
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema.__name__ == "GeminiIdeaBriefOutput"
    assert call["config"].max_output_tokens == 2048
    if "automatic_continuation" in type(call["config"]).model_fields:
        assert call["config"].automatic_continuation is False
    assert brief.original_text == idea
    assert brief.interpretation_mode == "gemini_live"
    assert {hypothesis.kind for hypothesis in brief.hypotheses} == {
        "demand",
        "willingness_to_pay",
        "competition",
        "feasibility",
        "timing",
    }
    assert {hypothesis.status for hypothesis in brief.hypotheses} == {"unknown"}
    assert brief.search_keywords != valid_output(idea)["search_keywords"]
    assert provider.provenance.model_identifier == "model-test-only"
    assert provider.provenance.prompt_version == PROMPT_VERSION
    from app.providers.llm.gemini import PROMPT_PATH

    expected_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    assert provider.provenance.prompt_hash == expected_hash


def test_truncated_response_fails_after_exactly_one_generation_request() -> None:
    idea = "A scheduling tool for small businesses with missed appointments."
    response = SimpleNamespace(
        parsed=valid_output(idea),
        text=None,
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    sdk = FakeSDKClient(response)
    provider = GeminiIdeaUnderstandingProvider(live_config(), client=sdk)

    with pytest.raises(ProviderFailure) as raised:
        asyncio.run(provider.extract_idea_brief(ValidationRequest(idea=idea)))

    assert raised.value.error.code == "incomplete_response"
    assert len(sdk.models.calls) == 1


def test_generation_config_explicitly_disables_automatic_continuation(monkeypatch) -> None:
    class ContinuationConfig(types.GenerateContentConfig):
        automatic_continuation: bool = True

    monkeypatch.setattr("app.providers.llm.gemini.types.GenerateContentConfig", ContinuationConfig)
    config = GeminiIdeaUnderstandingProvider._generation_config(max_output_tokens=256)
    assert config.automatic_continuation is False


def test_generation_config_fails_clearly_when_sdk_lacks_continuation_control(monkeypatch) -> None:
    class LegacyConfig:
        model_fields = {}

    monkeypatch.setattr("app.providers.llm.gemini.types.GenerateContentConfig", LegacyConfig)
    with pytest.raises(ProviderFailure) as raised:
        GeminiIdeaUnderstandingProvider._generation_config(max_output_tokens=256)
    assert raised.value.error.code == "sdk_incompatible"
    assert "Install the project dependencies" in raised.value.error.message


def test_oversized_gemini_response_is_a_typed_provider_error() -> None:
    sdk = FakeSDKClient(SimpleNamespace(parsed=None, text='{"padding":"' + ("x" * 1_100) + '"}'))
    provider = GeminiIdeaUnderstandingProvider(
        live_config(max_provider_response_bytes=1_024), client=sdk
    )
    with pytest.raises(ProviderFailure) as raised:
        asyncio.run(
            provider.extract_idea_brief(
                ValidationRequest(idea="A scheduling tool for small businesses.")
            )
        )
    assert raised.value.error.code == "response_too_large"
    assert len(sdk.models.calls) == 1


def test_user_fields_are_preserved_and_marked_as_user_provided() -> None:
    idea = "A scheduling tool for clinics with missed appointments."
    output = valid_output(idea)
    output["target_customer"] = {"value": "Independent clinics", "origin": "user"}
    output["geography"] = {"value": "India", "origin": "user"}
    provider = GeminiIdeaUnderstandingProvider(
        live_config(), client=FakeSDKClient(SimpleNamespace(parsed=output))
    )

    brief = asyncio.run(
        provider.extract_idea_brief(
            ValidationRequest(idea=idea, target_customer="Independent clinics", geography="India")
        )
    )

    assert brief.target_customer.value == "Independent clinics"
    assert brief.target_customer.origin == "user"
    assert brief.geography.origin == "user"


def test_pydantic_validation_failure_is_sanitized() -> None:
    idea = "A scheduling tool for small businesses with missed appointments."
    output = valid_output(idea)
    output["hypotheses"][0]["status"] = "supported"
    provider = GeminiIdeaUnderstandingProvider(
        live_config(), client=FakeSDKClient(SimpleNamespace(parsed=output))
    )

    with pytest.raises(ProviderFailure) as raised:
        asyncio.run(provider.extract_idea_brief(ValidationRequest(idea=idea)))

    assert raised.value.error.code == "schema_validation_failed"
    assert "supported" not in str(raised.value)


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (SimpleNamespace(parsed=None, text="not json"), "malformed_response"),
        (
            SimpleNamespace(parsed={"original_text": "incomplete"}, text=None),
            "schema_validation_failed",
        ),
        (
            SimpleNamespace(parsed=None, text=json.dumps({"original_text": "incomplete"})),
            "schema_validation_failed",
        ),
        (SimpleNamespace(parsed=None, text=None), "malformed_response"),
    ],
)
def test_malformed_or_incomplete_response_fails_without_brief(
    response: object, expected_code: str
) -> None:
    provider = GeminiIdeaUnderstandingProvider(live_config(), client=FakeSDKClient(response))

    with pytest.raises(ProviderFailure) as raised:
        asyncio.run(
            provider.extract_idea_brief(
                ValidationRequest(idea="A scheduling tool for small businesses.")
            )
        )

    assert raised.value.error.code == expected_code


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    [
        (
            genai_errors.ClientError(401, {"error": {"message": "bad credential"}}),
            "authentication_failed",
        ),
        (
            genai_errors.ServerError(429, {"error": {"message": "too many requests"}}),
            "rate_limited",
        ),
        (OSError(SENTINEL_SECRET), "network_error"),
        (genai_errors.ServerError(500, {"error": {"message": SENTINEL_SECRET}}), "provider_error"),
    ],
)
def test_provider_errors_are_classified_and_secret_redacted(
    exception: Exception, expected_code: str
) -> None:
    provider = GeminiIdeaUnderstandingProvider(
        live_config(), client=FakeSDKClient(exception=exception)
    )

    with pytest.raises(ProviderFailure) as raised:
        asyncio.run(
            provider.extract_idea_brief(
                ValidationRequest(idea="A scheduling tool for small businesses.")
            )
        )

    assert raised.value.error.code == expected_code
    assert SENTINEL_SECRET not in str(raised.value)
    assert SENTINEL_SECRET not in raised.value.error.model_dump_json()


def test_timeout_is_enforced_and_reported() -> None:
    class HangingModels(FakeModels):
        async def generate_content(self, **kwargs: object) -> object:
            await asyncio.sleep(1)
            return self.response

    sdk = FakeSDKClient()
    sdk.models = HangingModels()
    provider = GeminiIdeaUnderstandingProvider(
        live_config(gemini_timeout_seconds=0.01), client=sdk
    )

    with pytest.raises(ProviderFailure) as raised:
        asyncio.run(
            provider.extract_idea_brief(
                ValidationRequest(idea="A scheduling tool for small businesses.")
            )
        )

    assert raised.value.error.code == "timeout"


def test_factory_selects_gemini_in_live_and_mock_in_demo() -> None:
    demo_provider = create_idea_understanding_provider(load_config({}))
    live_provider = create_idea_understanding_provider(live_config())

    assert isinstance(demo_provider, MockIdeaUnderstandingProvider)
    assert isinstance(live_provider, GeminiIdeaUnderstandingProvider)


def test_pipeline_persists_sanitized_gemini_error_and_provenance(tmp_path, caplog) -> None:
    request = ValidationRequest(idea="A scheduling tool for small businesses.")
    now = datetime.now(UTC)
    run = ValidationRun(
        run_id="gemini-failure-run",
        created_at=now,
        updated_at=now,
        request=request,
    )
    store = RunStore(tmp_path / "gemini-failure.sqlite3")
    store.save(run)
    sdk = FakeSDKClient(
        exception=genai_errors.ClientError(
            401, {"error": {"message": f"SDK echoed {SENTINEL_SECRET}"}}
        )
    )
    provider = GeminiIdeaUnderstandingProvider(live_config(), client=sdk)
    runner = PipelineRunner(store, MockSearchProvider(), interpreter=provider)

    with caplog.at_level(logging.WARNING, logger="app.pipeline.runner"):
        asyncio.run(runner.execute(run.run_id))

    failed_run = store.get(run.run_id)
    assert failed_run is not None
    assert failed_run.status == "failed"
    assert failed_run.idea_brief is None
    assert failed_run.provider_errors[0].code == "authentication_failed"
    assert failed_run.error == failed_run.provider_errors[0].message
    assert failed_run.manifest.idea_understanding is not None
    assert failed_run.manifest.idea_understanding.model_identifier == "model-test-only"
    serialized = failed_run.model_dump_json()
    assert SENTINEL_SECRET not in serialized
    assert SENTINEL_SECRET not in caplog.text


def test_config_requires_model_timeout_and_output_limit() -> None:
    with pytest.raises(ValidationError, match="GEMINI_MODEL"):
        load_config(
            {
                "VALIDATOR_MODE": "live",
                "GEMINI_API_KEY": "test-key",
                "TAVILY_API_KEY": "test-key",
            }
        )
    config = load_config(
        {
            "VALIDATOR_MODE": "live",
            "GEMINI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
            "GEMINI_TIMEOUT_SECONDS": "17",
            "GEMINI_MAX_OUTPUT_TOKENS": "1536",
        }
    )
    assert config.gemini_timeout_seconds == 17
    assert config.gemini_max_output_tokens == 1536
