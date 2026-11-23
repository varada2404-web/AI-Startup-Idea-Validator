"""Network-free Tavily mapping, failure, limit, and pipeline tests."""

import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx
import pytest

from app.core.config import ApplicationConfig
from app.pipeline.runner import PipelineRunner
from app.pipeline.store import RunStore
from app.providers.llm.errors import ProviderFailure
from app.providers.search.base import SearchResult
from app.providers.search.factory import create_search_provider
from app.providers.search.tavily import TavilyProvider
from app.schemas.contracts import (
    ProviderError,
    ProviderRunProvenance,
    QueryPlanItem,
    Source,
    ValidationRequest,
    ValidationRun,
)
from app.services.idea_understanding import build_idea_brief
from app.services.source_processing import deduplicate_results, select_sources_for_extraction

TEST_SECRET = "test-only-tavily-secret-never-log"


def live_config(**overrides: object) -> ApplicationConfig:
    values: dict[str, object] = {
        "mode": "live",
        "gemini_api_key": "test-gemini-placeholder",
        "tavily_api_key": TEST_SECRET,
        "gemini_model": "test-model",
        "tavily_max_retries": 0,
        "tavily_max_retry_delay_seconds": 0,
    }
    values.update(overrides)
    return ApplicationConfig(**values)


def mocked_provider(handler, **config_overrides):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TavilyProvider(live_config(**config_overrides), client=client), client


def test_search_maps_snippet_provenance_limits_and_metadata() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://Example.org/story?utm_source=x",
                        "title": "A research page",
                        "content": "A search snippet only.",
                        "score": 0.82,
                        "published_date": "2026-01-02T00:00:00Z",
                    }
                    for _ in range(9)
                ]
            },
        )

    provider, client = mocked_provider(handler)
    try:
        results = asyncio.run(provider.search("clinic scheduling", "market", 1))
    finally:
        asyncio.run(client.aclose())

    assert len(results) == 5
    payload = json.loads(captured[0].content)
    assert captured[0].headers["Authorization"] == f"Bearer {TEST_SECRET}"
    assert "api_key" not in payload
    assert TEST_SECRET.encode() not in captured[0].content
    assert payload["max_results"] == 5
    assert results[0].url == "https://Example.org/story?utm_source=x"
    assert results[0].domain == "example.org"
    assert results[0].query == "clinic scheduling"
    assert results[0].category == "market"
    assert results[0].snippet == "A search snippet only."
    assert results[0].text == results[0].snippet
    assert results[0].provider_score == 0.82
    assert results[0].published_at == datetime(2026, 1, 2, tzinfo=UTC)
    sources, duplicates = deduplicate_results(results[:2])
    assert duplicates == 1
    assert sources[0].canonical_url == "https://example.org/story"
    assert sources[0].snippet == "A search snippet only."
    assert sources[0].extracted_text is None


def test_search_parses_rfc_publication_date() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [{
                    "url": "https://example.org/story",
                    "title": "Dated page",
                    "content": "A dated snippet.",
                    "published_date": "Fri, 02 Jan 2026 00:00:00 GMT",
                }]
            },
        )

    provider, client = mocked_provider(handler)
    try:
        results = asyncio.run(provider.search("query", "market", 1))
    finally:
        asyncio.run(client.aclose())
    assert results[0].published_at == datetime(2026, 1, 2, tzinfo=UTC)


def test_extraction_selection_preserves_category_diversity_before_relevance_fill() -> None:
    categories = (
        "customer_pain", "market", "direct_competitors", "alternatives",
        "pricing", "recent_developments", "negative_evidence",
    )
    sources = [
        # Repeated early-category results deliberately have much higher scores.
        *[
            _source(f"S{index}", "customer_pain", 0.99 - index / 100)
            for index in range(1, 6)
        ],
        *[
            _source(f"S{index + 5}", category, 0.3)
            for index, category in enumerate(categories[1:], start=1)
        ],
    ]
    selected = select_sources_for_extraction(sources, 8)
    assert len(selected) == 8
    assert set().union(*(set(source.categories) for source in selected)) >= set(categories)
    assert selected == select_sources_for_extraction(sources, 8)


def test_provider_semaphore_bounds_calls_across_concurrent_runs(tmp_path) -> None:
    class SlowProvider:
        active = 0
        maximum_active = 0

        async def search(self, query, category, round_number):
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return []

    async def run_searches() -> int:
        provider = SlowProvider()
        request = ValidationRequest(idea="A scheduling tool for small local clinics.")
        now = datetime.now(UTC)
        store = RunStore(tmp_path / "concurrency.sqlite3")
        run = ValidationRun(
            run_id="concurrent-run",
            created_at=now,
            updated_at=now,
            request=request,
        )
        store.save(run)
        shared_semaphore = asyncio.Semaphore(2)
        runner = PipelineRunner(
            store,
            provider,
            max_concurrent_calls=2,
            provider_semaphore=shared_semaphore,
        )
        queries = [
            QueryPlanItem(category=category, query=f"query {index}")
            for index, category in enumerate(
                (
                    "customer_pain", "market", "direct_competitors", "alternatives",
                    "pricing", "recent_developments", "negative_evidence",
                )
            )
        ]
        await asyncio.gather(runner._search(queries), runner._search(queries))
        return provider.maximum_active

    assert asyncio.run(run_searches()) == 2


def _source(source_id: str, category: str, score: float) -> Source:
    return Source(
        id=source_id,
        url=f"https://example.org/{source_id}",
        domain="example.org",
        title=source_id,
        query="query",
        categories=[category],
        cluster_id=source_id,
        text="snippet",
        snippet="snippet",
        source_type="tavily",
        simulated=False,
        provider_score=score,
    )


def test_extract_is_capped_and_maps_success_and_failure() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        payload = json.loads(request.content)
        urls = payload["urls"]
        return httpx.Response(
            200,
            json={
                "results": [{"url": urls[0], "raw_content": "X" * 900}],
                "failed_results": [{"url": urls[1], "error": "provider internal text"}],
            },
        )

    provider, client = mocked_provider(handler, max_source_text_chars=500)
    urls = [f"https://example.org/{number}" for number in range(10)]
    try:
        results = asyncio.run(provider.extract(urls))
    finally:
        asyncio.run(client.aclose())

    payload = json.loads(captured[0].content)
    assert captured[0].headers["Authorization"] == f"Bearer {TEST_SECRET}"
    assert "api_key" not in payload
    assert TEST_SECRET.encode() not in captured[0].content
    assert len(payload["urls"]) == 8
    assert len(results) == 8
    assert results[0].content == "X" * 500
    assert not results[0].failed
    assert results[1].failed and results[1].content is None
    assert all(result.failed for result in results[2:])


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "authentication_failed"), (403, "authentication_failed"), (429, "rate_limited")],
)
def test_http_auth_and_rate_errors_are_typed_and_sanitized(status: int, expected: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"provider echoed {TEST_SECRET}")

    provider, client = mocked_provider(handler)
    try:
        with pytest.raises(ProviderFailure) as raised:
            asyncio.run(provider.search("query", "market", 1))
    finally:
        asyncio.run(client.aclose())
    assert raised.value.error.code == expected
    assert TEST_SECRET not in str(raised.value)
    assert TEST_SECRET not in raised.value.error.message


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (httpx.ReadTimeout("secret-ish timeout"), "timeout"),
        (httpx.ConnectError("secret-ish network failure"), "network_error"),
    ],
)
def test_timeout_and_network_failures_are_sanitized(exception: Exception, expected: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise exception

    provider, client = mocked_provider(handler)
    try:
        with pytest.raises(ProviderFailure) as raised:
            asyncio.run(provider.search("query", "market", 1))
    finally:
        asyncio.run(client.aclose())
    assert raised.value.error.code == expected
    assert "secret-ish" not in str(raised.value)


def test_request_timeout_is_enforced_by_adapter() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"results": []})

    provider, client = mocked_provider(handler, tavily_timeout_seconds=0.005)
    try:
        with pytest.raises(ProviderFailure) as raised:
            asyncio.run(provider.search("query", "market", 1))
    finally:
        asyncio.run(client.aclose())
    assert raised.value.error.code == "timeout"


def test_oversized_provider_response_is_rejected_before_json_parsing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        body = json.dumps({"results": [], "padding": "x" * 1_200}).encode()
        return httpx.Response(200, content=body, headers={"Content-Length": str(len(body))})

    provider, client = mocked_provider(handler, max_provider_response_bytes=1_024)
    try:
        with pytest.raises(ProviderFailure) as raised:
            asyncio.run(provider.search("query", "market", 1))
    finally:
        asyncio.run(client.aclose())
    assert raised.value.error.code == "response_too_large"


def test_rate_limit_retry_respects_retry_after() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"results": []})

    provider, client = mocked_provider(handler, tavily_max_retries=1)
    try:
        assert asyncio.run(provider.search("query", "market", 1)) == []
    finally:
        asyncio.run(client.aclose())
    assert attempts == 2


def test_live_and_demo_factories_select_explicit_providers() -> None:
    live = create_search_provider(live_config())
    demo = create_search_provider(ApplicationConfig())
    assert isinstance(live, TavilyProvider)
    assert demo.__class__.__name__ == "MockSearchProvider"


class FakeGemini:
    provenance = ProviderRunProvenance(
        provider="gemini", model_identifier="fake-gemini", prompt_version="test-prompt"
    )

    async def extract_idea_brief(self, request: ValidationRequest):
        brief = build_idea_brief(request)
        brief.interpretation_mode = "gemini_live"
        return brief


def test_live_pipeline_with_fake_gemini_and_tavily_preserves_extraction_failure(tmp_path) -> None:
    active = 0
    maximum_active = 0
    search_count = 0
    extract_payload: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active, search_count, extract_payload
        data = json.loads(request.content)
        if request.url.path == "/extract":
            extract_payload = data
            urls = data["urls"]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": url,
                            "raw_content": f"Extracted page content: {url}" + ("E" * 900),
                        }
                        for url in urls[:-1]
                    ],
                    "failed_results": [{"url": urls[-1], "error": "private provider text"}],
                },
            )
        active += 1
        maximum_active = max(maximum_active, active)
        search_count += 1
        index = search_count
        await asyncio.sleep(0.005)
        active -= 1
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": f"https://research.example.org/page-{index}",
                        "title": f"Research record {index}",
                        "content": f"Search snippet for query {index}. " + ("S" * 900),
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = live_config()
    tavily = TavilyProvider(config, client=client)
    request = ValidationRequest(
        idea="A scheduling tool for small businesses with missed appointments."
    )
    now = datetime.now(UTC)
    run = ValidationRun(
        run_id="tavily-live-test",
        created_at=now,
        updated_at=now,
        request=request,
        mode="live",
        demo_mode=False,
        simulation_notice=None,
    )
    store = RunStore(tmp_path / "tavily-live.sqlite3")
    store.save(run)
    runner = PipelineRunner(
        store,
        tavily,
        interpreter=FakeGemini(),
        max_concurrent_calls=4,
        max_extract_sources=8,
        max_source_text_chars=600,
    )

    try:
        asyncio.run(runner.execute(run.run_id))
    finally:
        asyncio.run(client.aclose())

    result = store.get(run.run_id)
    assert result is not None and result.status == "completed"
    assert result.mode == "live" and result.demo_mode is False
    assert result.manifest.idea_understanding is not None
    assert result.manifest.idea_understanding.provider == "gemini"
    assert len(result.queries) == 11
    assert result.raw_result_count == 11
    assert len(result.sources) == 11
    assert maximum_active <= 4
    assert len(extract_payload["urls"]) == 8
    assert sum(source.extraction_status == "succeeded" for source in result.sources) == 7
    failed = [source for source in result.sources if source.extraction_status == "failed"]
    assert len(failed) == 1
    assert failed[0].snippet and failed[0].extracted_text is None
    assert failed[0].extraction_error is not None
    assert all(len(source.text) <= 600 for source in result.sources)
    assert all(
        source.extracted_text is None or len(source.extracted_text) <= 600
        for source in result.sources
    )
    assert len(result.model_dump_json()) < 100_000


class FailingSearchProvider:
    async def search(self, query, category, round_number):
        raise ProviderFailure(
            ProviderError(
                provider="tavily",
                stage="search",
                code="network_error",
                message="Tavily could not be reached.",
                retryable=True,
            )
        )


def test_all_search_failures_fail_research_without_fabricated_sources(tmp_path) -> None:
    request = ValidationRequest(
        idea="A scheduling tool for small businesses with missed appointments."
    )
    now = datetime.now(UTC)
    run = ValidationRun(
        run_id="all-searches-fail",
        created_at=now,
        updated_at=now,
        request=request,
        mode="live",
        demo_mode=False,
        simulation_notice=None,
    )
    store = RunStore(tmp_path / "all-fail.sqlite3")
    store.save(run)
    runner = PipelineRunner(store, FailingSearchProvider(), interpreter=FakeGemini())

    asyncio.run(runner.execute(run.run_id))
    result = store.get(run.run_id)

    assert result is not None and result.status == "failed"
    assert result.sources == []
    assert result.provider_errors[-1].code == "all_searches_failed"
    assert result.error == "All live searches failed; no sources were retrieved."


def test_tavily_auth_body_and_secret_are_not_persisted_or_logged(tmp_path, caplog) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid credential {TEST_SECRET}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tavily = TavilyProvider(live_config(), client=client)
    request = ValidationRequest(
        idea="A scheduling tool for small businesses with missed appointments."
    )
    now = datetime.now(UTC)
    run = ValidationRun(
        run_id="tavily-secret-test",
        created_at=now,
        updated_at=now,
        request=request,
        mode="live",
        demo_mode=False,
        simulation_notice=None,
    )
    store = RunStore(tmp_path / "tavily-secret.sqlite3")
    store.save(run)

    try:
        with caplog.at_level(logging.WARNING, logger="app.pipeline.runner"):
            asyncio.run(PipelineRunner(store, tavily, interpreter=FakeGemini()).execute(run.run_id))
    finally:
        asyncio.run(client.aclose())

    result = store.get(run.run_id)
    assert result is not None and result.status == "failed"
    assert TEST_SECRET not in result.model_dump_json()
    assert TEST_SECRET not in caplog.text


def test_successful_empty_search_is_not_misreported_as_provider_failure(tmp_path) -> None:
    class EmptySearch:
        async def search(self, query, category, round_number):
            if round_number == 1 and category == "customer_pain":
                return []
            if round_number == 2:
                raise ProviderFailure(
                    ProviderError(
                        provider="tavily",
                        stage="search",
                        code="network_error",
                        message="Tavily could not be reached.",
                        retryable=True,
                    )
                )
            return []

    request = ValidationRequest(
        idea="A scheduling tool for small businesses with missed appointments."
    )
    now = datetime.now(UTC)
    run = ValidationRun(
        run_id="empty-search-result",
        created_at=now,
        updated_at=now,
        request=request,
        mode="live",
        demo_mode=False,
        simulation_notice=None,
    )
    store = RunStore(tmp_path / "empty-search.sqlite3")
    store.save(run)
    asyncio.run(PipelineRunner(store, EmptySearch(), interpreter=FakeGemini()).execute(run.run_id))
    result = store.get(run.run_id)

    assert result is not None and result.status == "completed"
    assert result.sources == []
    assert result.quality == "degraded"
    assert all(error.code != "all_searches_failed" for error in result.provider_errors)


def test_partial_search_failure_completes_degraded_with_successful_sources(tmp_path) -> None:
    class PartiallyFailingSearch:
        async def search(self, query, category, round_number):
            if category == "market":
                raise ProviderFailure(
                    ProviderError(
                        provider="tavily",
                        stage="search",
                        code="rate_limited",
                        message="Tavily rate limit reached. Try again later.",
                        retryable=True,
                    )
                )
            return [
                SearchResult(
                    url=f"https://example.org/{category}-{round_number}",
                    domain="example.org",
                    title="Research result",
                    query=query,
                    category=category,
                    text="A provider supplied search snippet for this result.",
                    retrieved_at=datetime.now(UTC),
                    simulated=False,
                )
            ]

    request = ValidationRequest(
        idea="A scheduling tool for small businesses with missed appointments."
    )
    now = datetime.now(UTC)
    run = ValidationRun(
        run_id="partial-search-failure",
        created_at=now,
        updated_at=now,
        request=request,
        mode="live",
        demo_mode=False,
        simulation_notice=None,
    )
    store = RunStore(tmp_path / "partial-failure.sqlite3")
    store.save(run)
    runner = PipelineRunner(store, PartiallyFailingSearch(), interpreter=FakeGemini())

    asyncio.run(runner.execute(run.run_id))
    result = store.get(run.run_id)

    assert result is not None and result.status == "completed"
    assert result.quality == "degraded"
    assert result.sources
    assert all(not source.simulated for source in result.sources)
    assert any(error.code == "rate_limited" for error in result.provider_errors)
