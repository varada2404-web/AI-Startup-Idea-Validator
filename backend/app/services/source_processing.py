"""Canonical deduplication and coverage for Agent 1 source records."""

from app.providers.search.base import SearchResult
from app.schemas.contracts import CategoryCoverage, ResearchCategory, Source
from app.services.deduplication import canonicalize_url, near_duplicate
from app.services.query_planner import CATEGORY_ORDER


def deduplicate_results(results: list[SearchResult]) -> tuple[list[Source], int]:
    sources: list[Source] = []
    by_url: dict[str, Source] = {}
    duplicate_count = 0

    for result in results:
        url_key = canonicalize_url(result.url)
        existing = by_url.get(url_key)
        if existing is not None:
            duplicate_count += 1
            if result.category not in existing.categories:
                existing.categories.append(result.category)
            if result.query not in existing.queries:
                existing.queries.append(result.query)
            continue

        cluster = next(
            (source.cluster_id for source in sources if near_duplicate(source.text, result.text)),
            f"C{len(sources) + 1}",
        )
        source = Source(
            id=f"S{len(sources) + 1}",
            url=result.url,
            domain=result.domain,
            title=result.title,
            query=result.query,
            queries=[result.query],
            categories=[result.category],
            retrieved_at=result.retrieved_at,
            cluster_id=cluster,
            text=result.text,
            source_type="mock" if result.simulated else "tavily",
            simulated=result.simulated,
            snippet=result.snippet if result.snippet is not None else result.text,
            canonical_url=url_key,
            provider_score=result.provider_score,
            published_at=result.published_at,
            provider_metadata=result.provider_metadata,
        )
        sources.append(source)
        by_url[url_key] = source

    return sources, duplicate_count


def select_sources_for_extraction(sources: list[Source], limit: int = 8) -> list[Source]:
    """Select a deterministic, category-diverse set before filling by relevance."""
    if limit <= 0:
        return []
    candidates = [
        (index, source)
        for index, source in enumerate(sources)
        if not source.simulated and source.extraction_status == "not_attempted"
    ]
    selected: list[Source] = []
    selected_ids: set[str] = set()

    # Reserve representation for each category when matching sources exist.
    for category in CATEGORY_ORDER:
        if len(selected) >= limit:
            break
        matching = [
            (index, source)
            for index, source in candidates
            if source.id not in selected_ids and category in source.categories
        ]
        if matching:
            _, best = max(
                matching,
                key=lambda item: (
                    item[1].provider_score if item[1].provider_score is not None else -1.0,
                    -item[0],
                ),
            )
            selected.append(best)
            selected_ids.add(best.id)

    remaining = [item for item in candidates if item[1].id not in selected_ids]
    remaining.sort(
        key=lambda item: (
            item[1].provider_score if item[1].provider_score is not None else -1.0,
            -item[0],
        ),
        reverse=True,
    )
    selected.extend(source for _, source in remaining[: limit - len(selected)])
    return selected


def calculate_coverage(sources: list[Source]) -> list[CategoryCoverage]:
    coverage: list[CategoryCoverage] = []
    for category in CATEGORY_ORDER:
        relevant = [source for source in sources if category in source.categories]
        clusters = {source.cluster_id for source in relevant}
        count = len(relevant)
        status = (
            "covered" if count >= 2 and len(clusters) >= 2 else "partial" if count else "missing"
        )
        coverage.append(
            CategoryCoverage(
                category=category,
                status=status,
                source_count=count,
                independent_clusters=len(clusters),
            )
        )
    return coverage


def categories_needing_follow_up(coverage: list[CategoryCoverage]) -> list[ResearchCategory]:
    return [item.category for item in coverage if item.status in {"partial", "missing"}]
