"""Async Tavily Search and Extract adapter using the official REST API."""

import asyncio
import email.utils
import random
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.core.config import ApplicationConfig
from app.providers.llm.errors import ProviderFailure
from app.providers.search.base import ExtractionResult, SearchResult
from app.schemas.contracts import ProviderError, ResearchCategory

API_BASE = "https://api.tavily.com"


class TavilyProvider:
    """Tavily search and extraction, with bounded retries and sanitized errors."""

    def __init__(self, config: ApplicationConfig, client: httpx.AsyncClient | None = None) -> None:
        if config.mode != "live" or not config.tavily_api_key:
            raise ValueError("Tavily provider requires LIVE mode and a configured API key.")
        self._config = config
        self._client = client
        self._owns_client = client is None

    async def search(
        self, query: str, category: ResearchCategory, round_number: int
    ) -> list[SearchResult]:
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "max_results": min(5, self._config.max_search_results_per_query),
            "include_raw_content": False,
        }
        response = await self._request("/search", payload, stage="search")
        try:
            entries = response.get("results")
            if not isinstance(entries, list):
                raise ValueError
            now = datetime.now(UTC)
            results: list[SearchResult] = []
            for item in entries[: self._config.max_search_results_per_query]:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    continue
                url = item["url"].strip()
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    continue
                snippet_value = item.get("content")
                snippet = snippet_value if isinstance(snippet_value, str) else ""
                title_value = item.get("title")
                title = title_value.strip() if isinstance(title_value, str) else ""
                score = item.get("score")
                published = _parse_datetime(item.get("published_date"))
                results.append(
                    SearchResult(
                        url=url,
                        domain=parsed.hostname.casefold(),
                        title=title[:500],
                        query=query,
                        category=category,
                        text=snippet[: self._config.max_source_text_chars],
                        snippet=snippet[: self._config.max_source_text_chars],
                        retrieved_at=now,
                        simulated=False,
                        provider_score=score if isinstance(score, (int, float)) else None,
                        published_at=published,
                        provider_metadata=_safe_metadata(item),
                    )
                )
            return results
        except (AttributeError, TypeError, ValueError):
            raise self._failure(
                "malformed_response", "Tavily returned an unreadable search response.", "search"
            ) from None

    async def extract(self, urls: list[str]) -> list[ExtractionResult]:
        selected = list(dict.fromkeys(urls))[: self._config.max_extract_sources_per_run]
        if not selected:
            return []
        payload = {
            "urls": selected,
            "extract_depth": "basic",
            "include_images": False,
        }
        response = await self._request("/extract", payload, stage="extraction")
        try:
            entries = response.get("results", [])
            failures = response.get("failed_results", [])
            if not isinstance(entries, list) or not isinstance(failures, list):
                raise ValueError
            extracted: dict[str, str] = {}
            failed: set[str] = set()
            for item in entries:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    content = item.get("raw_content")
                    if isinstance(content, str) and content.strip():
                        extracted[item["url"]] = content[: self._config.max_source_text_chars]
                    else:
                        failed.add(item["url"])
            for item in failures:
                if isinstance(item, dict) and isinstance(item.get("url"), str):
                    failed.add(item["url"])
            return [
                ExtractionResult(
                    url=url,
                    content=extracted.get(url),
                    failed=url in failed or url not in extracted,
                )
                for url in selected
            ]
        except (AttributeError, TypeError, ValueError):
            raise self._failure(
                "malformed_response",
                "Tavily returned an unreadable extraction response.",
                "extraction",
            ) from None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, path: str, payload: dict[str, Any], stage: str) -> dict[str, Any]:
        client = self._get_client()
        retries = self._config.tavily_max_retries
        headers = {
            "Authorization": f"Bearer {self._config.tavily_api_key.get_secret_value()}"
        }
        for attempt in range(retries + 1):
            try:
                async with asyncio.timeout(self._config.tavily_timeout_seconds):
                    async with client.stream(
                        "POST", f"{API_BASE}{path}", json=payload, headers=headers
                    ) as stream:
                        status_code = stream.status_code
                        response_headers = stream.headers
                        content = (
                            await self._read_bounded(stream, stage)
                            if status_code < 400
                            else b""
                        )
            except (httpx.TimeoutException, TimeoutError):
                if attempt < retries:
                    await self._backoff(attempt, None, stage)
                    continue
                raise self._failure("timeout", "Tavily request timed out.", stage, True) from None
            except httpx.HTTPError:
                if attempt < retries:
                    await self._backoff(attempt, None, stage)
                    continue
                raise self._failure(
                    "network_error", "Tavily could not be reached.", stage, True
                ) from None

            if status_code in {401, 403}:
                raise self._failure(
                    "authentication_failed",
                    "Tavily authentication failed. Check server settings.",
                    stage,
                )
            if status_code == 429 or status_code >= 500:
                retryable = status_code == 429 or status_code >= 500
                if retryable and attempt < retries:
                    await self._backoff(
                        attempt, response_headers.get("Retry-After"), stage
                    )
                    continue
                code = "rate_limited" if status_code == 429 else "provider_unavailable"
                message = (
                    "Tavily rate limit reached. Try again later."
                    if status_code == 429
                    else "Tavily is temporarily unavailable."
                )
                raise self._failure(code, message, stage, retryable)
            if status_code >= 400:
                raise self._failure(
                    "request_rejected", "Tavily rejected the research request.", stage
                )
            try:
                response = httpx.Response(
                    status_code=status_code,
                    headers=response_headers,
                    content=content,
                    request=httpx.Request("POST", f"{API_BASE}{path}"),
                )
                data = response.json()
            except (ValueError, TypeError):
                raise self._failure(
                    "malformed_response", "Tavily returned malformed data.", stage
                ) from None
            if not isinstance(data, dict):
                raise self._failure(
                    "malformed_response", "Tavily returned malformed data.", stage
                )
            return data
        raise AssertionError("bounded retry loop exited unexpectedly")

    async def _read_bounded(self, response: httpx.Response, stage: str) -> bytes:
        limit = self._config.max_provider_response_bytes
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > limit:
                    raise self._failure(
                        "response_too_large",
                        "Tavily response exceeded the configured size limit.",
                        stage,
                    )
            except ValueError:
                raise self._failure(
                    "malformed_response", "Tavily returned invalid response headers.", stage
                ) from None
        chunks = bytearray()
        async for chunk in response.aiter_bytes():
            if len(chunks) + len(chunk) > limit:
                raise self._failure(
                    "response_too_large",
                    "Tavily response exceeded the configured size limit.",
                    stage,
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.tavily_timeout_seconds),
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
        return self._client

    async def _backoff(self, attempt: int, retry_after: str | None, stage: str) -> None:
        delay = _retry_delay(retry_after)
        maximum = self._config.tavily_max_retry_delay_seconds
        if delay is not None:
            if delay > maximum:
                # Do not retry before the provider's requested retry time.
                raise self._failure(
                    "rate_limited", "Tavily asked the service to retry later.", stage, True
                )
            await asyncio.sleep(delay)
            return
        await asyncio.sleep(min(maximum, 0.25 * (2**attempt) + random.uniform(0, 0.1)))

    @staticmethod
    def _failure(code: str, message: str, stage: str, retryable: bool = False) -> ProviderFailure:
        return ProviderFailure(
            ProviderError(
                provider="tavily",
                stage=stage,  # type: ignore[arg-type]
                code=code,
                message=message,
                retryable=retryable,
            )
        )


def _retry_delay(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _safe_metadata(item: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    allowed = (str, int, float, bool, type(None))
    return {
        key: value
        for key, value in item.items()
        if key not in {"url", "title", "content", "raw_content"}
        and isinstance(value, allowed)
    }
