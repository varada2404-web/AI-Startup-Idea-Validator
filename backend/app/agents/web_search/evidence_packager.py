"""Deterministic Evidence Packager for Stage 5 (no LLM calls, no fabrication)."""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit

from app.agents.web_search.search_executor import canonicalize_url
from app.schemas.evidence import (
    TRACKED_METADATA_FIELDS,
    CategoryCoverageStatus,
    EvidenceCategoryGroup,
    EvidencePackage,
    MetadataQualitySummary,
    PackagedEvidence,
    PackagingError,
)
from app.schemas.research import (
    ResearchCategory,
    SearchExecutionReport,
    SourceRecord,
)


def _clean_optional_str(value: object) -> str | None:
    """Return stripped non-empty string or None without fabricating data."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_domain(explicit_domain: object, url: str) -> str | None:
    """Normalize an explicit domain or derive it deterministically from the URL."""
    cleaned_domain = _clean_optional_str(explicit_domain)
    if cleaned_domain:
        normalized = cleaned_domain.lower()
        if normalized.startswith("www."):
            normalized = normalized[4:]
        return normalized or None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _is_valid_http_url(url: str) -> bool:
    """Return True if url is a well-formed http or https URL with a hostname."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def _make_evidence_id(canonical_url: str) -> str:
    """Produce a deterministic evidence_id from the canonicalized URL."""
    digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:12]
    return f"ev_{digest}"


def _compute_metadata_completeness(
    *,
    title: str | None,
    domain: str | None,
    content_snippet: str | None,
    published_date: str | None,
    relevance_score: float | None,
) -> tuple[float, list[str]]:
    """Compute transparent equal-weight metadata completeness across 5 tracked fields.

    Measures metadata presence only — never factual accuracy or source credibility.
    """
    field_presence = {
        "title": title is not None,
        "domain": domain is not None,
        "content_snippet": content_snippet is not None,
        "published_date": published_date is not None,
        "relevance_score": relevance_score is not None,
    }
    missing = [field for field in TRACKED_METADATA_FIELDS if not field_presence[field]]
    present_count = len(TRACKED_METADATA_FIELDS) - len(missing)
    score = round(present_count / len(TRACKED_METADATA_FIELDS), 2)
    return score, missing


def _coverage_status_for_count(count: int) -> CategoryCoverageStatus:
    """Map evidence count to an objective volume-based coverage status."""
    if count <= 0:
        return CategoryCoverageStatus.NONE
    if count <= 2:
        return CategoryCoverageStatus.LIMITED
    if count <= 5:
        return CategoryCoverageStatus.MODERATE
    return CategoryCoverageStatus.BROAD


def _coerce_category(value: object) -> ResearchCategory | None:
    """Safely convert a ResearchCategory or category value string into ResearchCategory."""
    if isinstance(value, ResearchCategory):
        return value
    if isinstance(value, str):
        try:
            return ResearchCategory(value.strip())
        except ValueError:
            return None
    return None


def _collect_queries(source: SourceRecord) -> list[str]:
    """Collect deduplicated originating queries preserving discovery order."""
    candidates: list[object] = [getattr(source, "query", None)]
    matched = getattr(source, "matched_queries", None)
    if isinstance(matched, list):
        candidates.extend(matched)

    seen: set[str] = set()
    queries: list[str] = []
    for raw in candidates:
        cleaned = _clean_optional_str(raw)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            queries.append(cleaned)
    return queries


def _collect_categories(source: SourceRecord) -> list[ResearchCategory]:
    """Collect deduplicated ResearchCategory values preserving discovery order."""
    candidates: list[object] = [getattr(source, "category", None)]
    matched = getattr(source, "matched_categories", None)
    if isinstance(matched, list):
        candidates.extend(matched)

    seen: set[ResearchCategory] = set()
    categories: list[ResearchCategory] = []
    for raw in candidates:
        coerced = _coerce_category(raw)
        if coerced is not None and coerced not in seen:
            seen.add(coerced)
            categories.append(coerced)
    return categories


def _normalize_relevance_score(raw_score: object) -> float | None:
    """Preserve valid non-negative provider relevance_score without rescaling."""
    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        value = float(raw_score)
        if value >= 0.0:
            return value
    return None


class EvidencePackager:
    """Deterministically normalizes, deduplicates, and groups SearchExecutionReport sources."""

    def package(self, search_report: SearchExecutionReport) -> EvidencePackage:
        """Transform a SearchExecutionReport into a structured EvidencePackage."""
        if not isinstance(search_report, SearchExecutionReport):
            raise TypeError("search_report must be a SearchExecutionReport instance.")

        total_sources_received = len(search_report.sources)
        duplicate_sources = 0
        packaging_errors: list[PackagingError] = []
        merged_by_canonical: dict[str, PackagedEvidence] = {}

        for source in search_report.sources:
            if not isinstance(source, SourceRecord):
                packaging_errors.append(
                    PackagingError(
                        url=None,
                        error_type="invalid_source_record",
                        message="Encountered a non-SourceRecord item during evidence packaging.",
                    )
                )
                continue

            raw_url = _clean_optional_str(getattr(source, "url", None))
            if not raw_url or not _is_valid_http_url(raw_url):
                packaging_errors.append(
                    PackagingError(
                        url=raw_url,
                        error_type="invalid_url",
                        message="Skipped source record with missing or non-HTTP(S) URL.",
                    )
                )
                continue

            categories = _collect_categories(source)
            if not categories:
                packaging_errors.append(
                    PackagingError(
                        url=raw_url,
                        error_type="invalid_category",
                        message="Skipped source record with missing or unrecognized research category.",
                    )
                )
                continue

            canonical_key = canonicalize_url(raw_url)
            evidence_id = _make_evidence_id(canonical_key)
            title = _clean_optional_str(getattr(source, "title", None))
            domain = _normalize_domain(getattr(source, "domain", None), raw_url)
            content_snippet = _clean_optional_str(getattr(source, "content", None))
            published_date = _clean_optional_str(getattr(source, "published_date", None))
            relevance_score = _normalize_relevance_score(
                getattr(source, "relevance_score", None)
            )
            queries = _collect_queries(source)

            existing = merged_by_canonical.get(canonical_key)
            if existing is None:
                completeness, missing = _compute_metadata_completeness(
                    title=title,
                    domain=domain,
                    content_snippet=content_snippet,
                    published_date=published_date,
                    relevance_score=relevance_score,
                )
                merged_by_canonical[canonical_key] = PackagedEvidence(
                    evidence_id=evidence_id,
                    title=title,
                    url=raw_url,
                    domain=domain,
                    content_snippet=content_snippet,
                    research_category=categories[0],
                    originating_queries=queries,
                    matched_categories=categories,
                    relevance_score=relevance_score,
                    published_date=published_date,
                    metadata_completeness=completeness,
                    missing_metadata=missing,
                )
            else:
                duplicate_sources += 1

                merged_queries = list(existing.originating_queries)
                for q in queries:
                    if q not in merged_queries:
                        merged_queries.append(q)

                merged_categories = list(existing.matched_categories)
                for cat in categories:
                    if cat not in merged_categories:
                        merged_categories.append(cat)

                merged_title = existing.title or title
                merged_domain = existing.domain or domain

                if existing.content_snippet is None:
                    merged_snippet = content_snippet
                elif content_snippet is not None and len(content_snippet) > len(existing.content_snippet):
                    merged_snippet = content_snippet
                else:
                    merged_snippet = existing.content_snippet

                merged_published_date = existing.published_date or published_date

                if relevance_score is not None and (
                    existing.relevance_score is None
                    or relevance_score > existing.relevance_score
                ):
                    merged_relevance = relevance_score
                else:
                    merged_relevance = existing.relevance_score

                completeness, missing = _compute_metadata_completeness(
                    title=merged_title,
                    domain=merged_domain,
                    content_snippet=merged_snippet,
                    published_date=merged_published_date,
                    relevance_score=merged_relevance,
                )

                merged_by_canonical[canonical_key] = existing.model_copy(
                    update={
                        "title": merged_title,
                        "domain": merged_domain,
                        "content_snippet": merged_snippet,
                        "originating_queries": merged_queries,
                        "matched_categories": merged_categories,
                        "relevance_score": merged_relevance,
                        "published_date": merged_published_date,
                        "metadata_completeness": completeness,
                        "missing_metadata": missing,
                    }
                )

        unique_items = list(merged_by_canonical.values())

        # Group deterministically across all five fixed research categories.
        category_groups: list[EvidenceCategoryGroup] = []
        for category in ResearchCategory:
            items_for_category = [
                item for item in unique_items if category in item.matched_categories
            ]
            items_for_category.sort(
                key=lambda item: (
                    -(item.relevance_score if item.relevance_score is not None else -1.0),
                    item.evidence_id,
                )
            )
            unique_domains = sorted(
                {item.domain for item in items_for_category if item.domain}
            )
            category_groups.append(
                EvidenceCategoryGroup(
                    category=category,
                    evidence_items=items_for_category,
                    total_evidence=len(items_for_category),
                    unique_domains=unique_domains,
                    coverage_status=_coverage_status_for_count(len(items_for_category)),
                )
            )

        uncovered_categories = [
            group.category for group in category_groups if group.total_evidence == 0
        ]

        # Build aggregate metadata quality summary across unique packaged sources.
        missing_field_counts = {field: 0 for field in TRACKED_METADATA_FIELDS}
        complete_count = 0
        partial_count = 0
        minimal_count = 0

        for item in unique_items:
            for field in item.missing_metadata:
                if field in missing_field_counts:
                    missing_field_counts[field] += 1
            if item.metadata_completeness >= 1.0:
                complete_count += 1
            elif item.metadata_completeness > 0.0:
                partial_count += 1
            else:
                minimal_count += 1

        avg_completeness = (
            round(
                sum(item.metadata_completeness for item in unique_items) / len(unique_items),
                4,
            )
            if unique_items
            else 0.0
        )

        quality_summary = MetadataQualitySummary(
            average_completeness=avg_completeness,
            complete_metadata_count=complete_count,
            partial_metadata_count=partial_count,
            minimal_metadata_count=minimal_count,
            missing_field_counts=missing_field_counts,
        )

        return EvidencePackage(
            total_sources_received=total_sources_received,
            unique_sources=len(unique_items),
            duplicate_sources=duplicate_sources,
            total_categories=len(category_groups),
            categories=category_groups,
            metadata_quality_summary=quality_summary,
            uncovered_categories=uncovered_categories,
            packaging_errors=packaging_errors,
        )


def package_evidence(search_report: SearchExecutionReport) -> EvidencePackage:
    """Convenience function to package a SearchExecutionReport into an EvidencePackage."""
    return EvidencePackager().package(search_report)
