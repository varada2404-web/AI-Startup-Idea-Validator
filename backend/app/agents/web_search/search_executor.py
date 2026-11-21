"""Sequential Tavily Search Executor for Agent 1."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.providers.tavily import (
    DEFAULT_TAVILY_MAX_RESULTS,
    TavilyAuthError,
    TavilyConfigError,
    TavilyInvalidResponseError,
    TavilyProvider,
    TavilyProviderError,
    TavilyRateLimitError,
    TavilySearchItem,
    TavilyTimeoutError,
    TavilyUnavailableError,
)
from app.schemas.research import (
    ResearchQueryPlan,
    SearchExecutionReport,
    SearchQueryError,
    SourceRecord,
)

DEFAULT_MIN_CONTENT_LENGTH = 30
DEFAULT_MIN_RELEVANCE_SCORE = 0.15

_TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "yclid",
}

_BOILERPLATE_TITLE_MARKERS = (
    "404 not found",
    "page not found",
    "access denied",
    "403 forbidden",
    "just a moment",
    "attention required",
    "robot or human",
)


def canonicalize_url(url: str) -> str:
    """Return a canonical URL key for deduplication while never mutating the original URL."""
    raw = url.strip()
    parsed = urlsplit(raw)

    scheme = parsed.scheme.lower()
    if scheme == "http":
        scheme = "https"

    host = (parsed.hostname or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]

    port = parsed.port
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        netloc = host
    elif port is not None:
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    if path == "/":
        path = ""

    filtered_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_QUERY_PARAMS
    ]
    filtered_params.sort()
    query = urlencode(filtered_params)

    return urlunsplit((scheme, netloc, path, query, ""))


def _is_valid_http_url(url: str) -> bool:
    """Verify the URL is a well-formed HTTP or HTTPS link with a host."""
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def _is_relevant_result(
    item: TavilySearchItem,
    *,
    min_content_length: int,
    min_relevance_score: float | None,
) -> bool:
    """Filter empty, malformed, boilerplate, or low-relevance search hits."""
    if not item.title or not item.url or not item.content:
        return False
    if not _is_valid_http_url(item.url):
        return False
    if len(item.content) < min_content_length:
        return False

    lowered_title = item.title.lower()
    if any(marker in lowered_title for marker in _BOILERPLATE_TITLE_MARKERS):
        return False

    if (
        min_relevance_score is not None
        and item.relevance_score is not None
        and item.relevance_score < min_relevance_score
    ):
        return False

    return True


def _classify_provider_error(exc: TavilyProviderError) -> tuple[str, str]:
    """Map provider exceptions to safe, secret-free error codes and messages."""
    if isinstance(exc, TavilyRateLimitError):
        return ("rate_limit", "Search provider rate limit exceeded for this query.")
    if isinstance(exc, TavilyTimeoutError):
        return ("timeout", "Search request timed out for this query.")
    if isinstance(exc, TavilyInvalidResponseError):
        return ("invalid_response", "Search provider returned a malformed response.")
    if isinstance(exc, TavilyUnavailableError):
        return ("unavailable", "Search provider was unavailable for this query.")
    return ("provider_error", "Search provider error occurred for this query.")


class SearchExecutor:
    """Sequentially executes a ResearchQueryPlan through TavilyProvider."""

    def __init__(
        self,
        provider: TavilyProvider | None = None,
        *,
        max_results_per_query: int = DEFAULT_TAVILY_MAX_RESULTS,
        search_depth: str | None = None,
        min_content_length: int = DEFAULT_MIN_CONTENT_LENGTH,
        min_relevance_score: float | None = DEFAULT_MIN_RELEVANCE_SCORE,
    ) -> None:
        self._provider = provider or TavilyProvider()
        self._max_results_per_query = max_results_per_query
        self._search_depth = search_depth
        self._min_content_length = min_content_length
        self._min_relevance_score = min_relevance_score

    def execute(self, plan: ResearchQueryPlan) -> SearchExecutionReport:
        """Execute all planned queries sequentially and return a validated SearchExecutionReport."""
        # Fail fast if configuration (such as TAVILY_API_KEY) is missing or invalid.
        self._provider.validate_config()

        total_queries = len(plan.queries)
        successful_queries = 0
        failed_queries = 0
        total_results_retrieved = 0
        errors: list[SearchQueryError] = []
        deduplicated_sources: dict[str, SourceRecord] = {}

        for planned_query in plan.queries:
            try:
                raw_items = self._provider.search(
                    planned_query.query,
                    max_results=self._max_results_per_query,
                    search_depth=self._search_depth,
                )
            except (TavilyConfigError, TavilyAuthError):
                # Configuration and authentication errors affect all queries; fail fast cleanly.
                raise
            except TavilyProviderError as exc:
                failed_queries += 1
                error_type, safe_message = _classify_provider_error(exc)
                errors.append(
                    SearchQueryError(
                        query=planned_query.query,
                        category=planned_query.category,
                        error_type=error_type,
                        message=safe_message,
                    )
                )
                continue

            successful_queries += 1
            total_results_retrieved += len(raw_items)

            for item in raw_items:
                if not _is_relevant_result(
                    item,
                    min_content_length=self._min_content_length,
                    min_relevance_score=self._min_relevance_score,
                ):
                    continue

                canonical_key = canonicalize_url(item.url)
                existing = deduplicated_sources.get(canonical_key)

                if existing is None:
                    deduplicated_sources[canonical_key] = SourceRecord(
                        title=item.title,
                        url=item.url,
                        content=item.content,
                        query=planned_query.query,
                        category=planned_query.category,
                        matched_queries=[planned_query.query],
                        matched_categories=[planned_query.category],
                        relevance_score=item.relevance_score,
                        published_date=item.published_date,
                        domain=item.domain,
                    )
                else:
                    # Preserve provenance from additional queries without discarding the source.
                    updated_queries = list(existing.matched_queries)
                    if planned_query.query not in updated_queries:
                        updated_queries.append(planned_query.query)

                    updated_categories = list(existing.matched_categories)
                    if planned_query.category not in updated_categories:
                        updated_categories.append(planned_query.category)

                    best_score = existing.relevance_score
                    if item.relevance_score is not None and (
                        best_score is None or item.relevance_score > best_score
                    ):
                        best_score = item.relevance_score

                    best_content = (
                        item.content
                        if len(item.content) > len(existing.content)
                        else existing.content
                    )
                    best_published_date = existing.published_date or item.published_date

                    deduplicated_sources[canonical_key] = existing.model_copy(
                        update={
                            "matched_queries": updated_queries,
                            "matched_categories": updated_categories,
                            "relevance_score": best_score,
                            "content": best_content,
                            "published_date": best_published_date,
                        }
                    )

        sources = list(deduplicated_sources.values())
        return SearchExecutionReport(
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            total_results_retrieved=total_results_retrieved,
            unique_sources_count=len(sources),
            sources=sources,
            errors=errors,
        )


def execute_research_plan(
    plan: ResearchQueryPlan,
    provider: TavilyProvider | None = None,
    *,
    max_results_per_query: int = DEFAULT_TAVILY_MAX_RESULTS,
) -> SearchExecutionReport:
    """Convenience function to execute a ResearchQueryPlan with SearchExecutor."""
    return SearchExecutor(
        provider=provider,
        max_results_per_query=max_results_per_query,
    ).execute(plan)
