"""Tavily Web Search provider using the official Tavily Search REST API."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"
DEFAULT_TAVILY_MAX_RESULTS = 3
DEFAULT_TAVILY_TIMEOUT_SECONDS = 20.0
ALLOWED_SEARCH_DEPTHS = {"basic", "advanced"}


class TavilyProviderError(Exception):
    """Base exception for Tavily provider failures."""


class TavilyConfigError(TavilyProviderError):
    """Raised when Tavily configuration (such as API key or settings) is missing or invalid."""


class TavilyAuthError(TavilyProviderError):
    """Raised when Tavily rejects authentication credentials."""


class TavilyRateLimitError(TavilyProviderError):
    """Raised when the Tavily API rate limit is exceeded."""


class TavilyTimeoutError(TavilyProviderError):
    """Raised when a Tavily API request exceeds the configured timeout."""


class TavilyUnavailableError(TavilyProviderError):
    """Raised when the Tavily API is unreachable or returns a server/network error."""


class TavilyInvalidResponseError(TavilyProviderError):
    """Raised when Tavily returns malformed JSON or an unexpected response structure."""


class TavilySearchItem(BaseModel):
    """Normalized single search result returned by TavilyProvider."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(default="", description="Page title returned by the search engine.")
    url: str = Field(default="", description="Original page URL returned by the search engine.")
    content: str = Field(default="", description="Retrieved page snippet/content (untrusted data).")
    relevance_score: float | None = Field(
        default=None,
        description="Optional numeric relevance score reported by Tavily.",
    )
    published_date: str | None = Field(
        default=None,
        description="Optional publication date reported by Tavily.",
    )
    domain: str | None = Field(
        default=None,
        description="Normalized source domain parsed from the URL.",
    )


def _extract_domain(url: str) -> str | None:
    """Extract a clean lowercase domain without leading www. from an HTTP(S) URL."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.hostname or ""
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host or None


class TavilyProvider:
    """Thin, safe client for the official Tavily Search API returning normalized results."""

    def __init__(
        self,
        api_key: str | None = None,
        search_depth: str | None = None,
        max_results: int | None = None,
        timeout_seconds: float | None = None,
        endpoint_url: str = TAVILY_SEARCH_URL,
    ) -> None:
        self._api_key_override = api_key
        self._search_depth_override = search_depth
        self._max_results_override = max_results
        self._timeout_override = timeout_seconds
        self._endpoint_url = endpoint_url

    def validate_config(self) -> None:
        """Validate that required Tavily configuration is present and well-formed."""
        self._get_api_key()
        self._get_search_depth()
        self._get_max_results()
        self._get_timeout_seconds()

    def _get_api_key(self) -> str:
        raw_key = (
            self._api_key_override
            if self._api_key_override is not None
            else os.getenv("TAVILY_API_KEY", "")
        )
        api_key = raw_key.strip()
        if not api_key:
            raise TavilyConfigError("TAVILY_API_KEY is not configured.")
        return api_key

    def _get_search_depth(self, override: str | None = None) -> str:
        raw_depth = (
            override
            if override is not None
            else (
                self._search_depth_override
                if self._search_depth_override is not None
                else os.getenv("TAVILY_SEARCH_DEPTH", DEFAULT_TAVILY_SEARCH_DEPTH)
            )
        )
        depth = raw_depth.strip().lower() if raw_depth else DEFAULT_TAVILY_SEARCH_DEPTH
        if depth not in ALLOWED_SEARCH_DEPTHS:
            raise TavilyConfigError(
                f"TAVILY_SEARCH_DEPTH must be one of {sorted(ALLOWED_SEARCH_DEPTHS)}."
            )
        return depth

    def _get_max_results(self, override: int | None = None) -> int:
        if override is not None:
            max_results = override
        elif self._max_results_override is not None:
            max_results = self._max_results_override
        else:
            raw_max = os.getenv("TAVILY_MAX_RESULTS", str(DEFAULT_TAVILY_MAX_RESULTS)).strip()
            try:
                max_results = int(raw_max) if raw_max else DEFAULT_TAVILY_MAX_RESULTS
            except ValueError as exc:
                raise TavilyConfigError("TAVILY_MAX_RESULTS must be a positive integer.") from exc

        if max_results < 1 or max_results > 10:
            raise TavilyConfigError("TAVILY_MAX_RESULTS must be between 1 and 10.")
        return max_results

    def _get_timeout_seconds(self) -> float:
        if self._timeout_override is not None:
            timeout_seconds = self._timeout_override
        else:
            raw_timeout = os.getenv(
                "TAVILY_TIMEOUT_SECONDS", str(DEFAULT_TAVILY_TIMEOUT_SECONDS)
            ).strip()
            try:
                timeout_seconds = (
                    float(raw_timeout) if raw_timeout else DEFAULT_TAVILY_TIMEOUT_SECONDS
                )
            except ValueError as exc:
                raise TavilyConfigError(
                    "TAVILY_TIMEOUT_SECONDS must be a positive number."
                ) from exc

        if timeout_seconds <= 0:
            raise TavilyConfigError("TAVILY_TIMEOUT_SECONDS must be greater than zero.")
        return timeout_seconds

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        search_depth: str | None = None,
    ) -> list[TavilySearchItem]:
        """Execute a Tavily search query and return normalized result items."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query must not be empty.")

        api_key = self._get_api_key()
        resolved_depth = self._get_search_depth(override=search_depth)
        resolved_max_results = self._get_max_results(override=max_results)
        timeout_seconds = self._get_timeout_seconds()

        payload = {
            "api_key": api_key,
            "query": cleaned_query,
            "search_depth": resolved_depth,
            "max_results": resolved_max_results,
        }

        try:
            response = httpx.post(
                self._endpoint_url,
                json=payload,
                timeout=timeout_seconds,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise TavilyTimeoutError("Tavily search request timed out.") from exc
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            raise TavilyUnavailableError("Unable to reach the Tavily search service.") from exc

        status_code = response.status_code
        if status_code in (401, 403):
            raise TavilyAuthError("Tavily API authentication failed.")
        if status_code == 429:
            raise TavilyRateLimitError("Tavily API rate limit exceeded.")
        if status_code in (408, 504):
            raise TavilyTimeoutError("Tavily search request timed out.")
        if status_code >= 500:
            raise TavilyUnavailableError("Tavily search service returned a server error.")
        if status_code >= 400:
            raise TavilyUnavailableError("Tavily search request was rejected.")

        try:
            data = response.json()
        except ValueError as exc:
            raise TavilyInvalidResponseError(
                "Tavily search returned invalid JSON."
            ) from exc

        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise TavilyInvalidResponseError(
                "Tavily search response did not contain a valid results list."
            )

        normalized: list[TavilySearchItem] = []
        for raw_item in data["results"]:
            if not isinstance(raw_item, dict):
                continue

            title = str(raw_item.get("title") or "").strip()
            url = str(raw_item.get("url") or "").strip()
            content = str(raw_item.get("content") or "").strip()

            raw_score = raw_item.get("score")
            relevance_score: float | None = None
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                score_val = float(raw_score)
                if score_val >= 0.0:
                    relevance_score = score_val

            raw_date = raw_item.get("published_date")
            published_date = (
                raw_date.strip() if isinstance(raw_date, str) and raw_date.strip() else None
            )

            normalized.append(
                TavilySearchItem(
                    title=title,
                    url=url,
                    content=content,
                    relevance_score=relevance_score,
                    published_date=published_date,
                    domain=_extract_domain(url) if url else None,
                )
            )

        return normalized
