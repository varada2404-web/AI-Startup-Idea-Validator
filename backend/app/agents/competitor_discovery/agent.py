"""
Competitor Discovery and Comparison Agent (Milestone 2)
---------------------------------------------------------
Part of: AI-Based Startup Idea Validator with Market Analysis Assistance
Author: Varada

Discovers existing players in the market for a startup idea, compares their
offerings, and identifies gaps — using the existing SearchProvider interface
so it works in both demo (mock) and live (Tavily) modes.
"""

from datetime import datetime
from urllib.parse import urlsplit

from app.agents.competitor_discovery.schemas import CompetitorProfile, CompetitorReport
from app.providers.search.base import SearchProvider

# ---------------------------------------------------------------------------
# Query templates — no hardcoded years; use {year} placeholder
# ---------------------------------------------------------------------------

COMPETITOR_QUERY_TEMPLATES: list[tuple[str, str]] = [
    ("{idea} direct competitors and alternatives", "direct_competitors"),
    ("{idea} existing products and startups {year}", "direct_competitors"),
    ("{idea} market players comparison pricing", "pricing"),
    ("{idea} similar tools weaknesses and complaints", "negative_evidence"),
    ("{idea} gaps and unmet needs in market", "alternatives"),
]


def _build_queries(idea: str) -> list[tuple[str, str]]:
    """Return (query_string, category) pairs with the current year injected."""
    year = str(datetime.now().year)
    return [
        (template.format(idea=idea, year=year), category)
        for template, category in COMPETITOR_QUERY_TEMPLATES
    ]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CompetitorDiscoveryAgent:
    """
    Discovers competitors and compares their offerings for a given startup idea.

    Uses the existing SearchProvider interface — works with MockSearchProvider
    in demo mode and TavilyProvider in live mode.

    Usage:
        agent = CompetitorDiscoveryAgent(search_provider=provider)
        report = await agent.run("AI resume screener for HR teams")
    """

    def __init__(self, search_provider: SearchProvider) -> None:
        self._provider = search_provider

    async def run(self, idea: str, max_results: int = 5) -> CompetitorReport:  # noqa: ARG002
        """
        Run competitor discovery for the given startup idea.

        Args:
            idea: Plain-text startup idea description.
            max_results: Unused directly — controlled by provider config.

        Returns:
            CompetitorReport with competitors, gap analysis, and summary.
        """
        queries = _build_queries(idea)
        all_results = []
        errors: list[str] = []
        queries_used: list[str] = []

        for query_str, category in queries:
            try:
                results = await self._provider.search(
                    query=query_str,
                    category=category,  # type: ignore[arg-type]
                    round_number=1,
                )
                all_results.extend(results)
                queries_used.append(query_str)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Query failed '{query_str}': {exc}")

        competitor_results = [
            r for r in all_results if r.category == "direct_competitors"
        ]
        negative_results = [
            r for r in all_results if r.category == "negative_evidence"
        ]

        competitors = _build_competitor_profiles(competitor_results)
        gap_analysis = _extract_gaps(negative_results)
        summary = _build_summary(idea, competitors, gap_analysis)

        return CompetitorReport(
            idea=idea,
            competitors=competitors,
            gap_analysis=gap_analysis,
            summary=summary,
            search_queries_used=queries_used,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract hostname from URL, stripping www. prefix."""
    try:
        hostname = urlsplit(url).hostname or ""
        return hostname.removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


def _build_competitor_profiles(
    results: list,
) -> list[CompetitorProfile]:
    """
    Group search results by domain and build one CompetitorProfile per domain.
    Deduplicates multiple results from the same domain into one profile.
    """
    domain_map: dict[str, CompetitorProfile] = {}

    for result in results:
        domain = _extract_domain(result.url)
        if not domain:
            continue

        if domain not in domain_map:
            domain_map[domain] = CompetitorProfile(
                name=result.title or domain,
                url=result.url,
                domain=domain,
                description=result.text or "",
                source_urls=[result.url],
                offerings=_extract_offerings(result.text or ""),
                strengths=_extract_strengths(result.text or ""),
                weaknesses=[],
            )
        else:
            # Merge additional URLs from same domain
            profile = domain_map[domain]
            if result.url not in profile.source_urls:
                profile.source_urls.append(result.url)
            # Accumulate offerings from additional pages
            new_offerings = _extract_offerings(result.text or "")
            existing = set(profile.offerings)
            profile.offerings.extend(
                o for o in new_offerings if o not in existing
            )

    return list(domain_map.values())


def _extract_offerings(text: str) -> list[str]:
    """Extract brief feature/offering sentences from result text."""
    if not text:
        return []
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    return sentences[:3]


def _extract_strengths(text: str) -> list[str]:
    """Extract strength signals from text based on positive keywords."""
    if not text:
        return []
    strength_keywords = ["offers", "provides", "features", "supports", "integrates", "includes"]
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return [
        s for s in sentences
        if any(kw in s.lower() for kw in strength_keywords)
    ][:2]


def _extract_gaps(negative_results: list) -> list[str]:
    """Extract gap signals from negative evidence results."""
    gaps: list[str] = []
    gap_keywords = [
        "expensive", "difficult", "no integration", "missing", "lack",
        "fail", "complaint", "limitation", "not support", "doesn't", "poor",
    ]
    for result in negative_results:
        text = result.text or ""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in gap_keywords):
                gaps.append(sentence)
    return gaps[:5]


def _build_summary(
    idea: str,
    competitors: list[CompetitorProfile],
    gap_analysis: list[str],
) -> str:
    """Build a human-readable summary of the competitive landscape."""
    if not competitors:
        return (
            f"No direct competitors found for '{idea}'. "
            "This may indicate an emerging market or a very niche segment."
        )

    names = ", ".join(c.name for c in competitors[:5])
    gap_text = (
        f" Key gaps identified: {gap_analysis[0]}."
        if gap_analysis
        else " No clear gaps identified from available data."
    )

    return (
        f"Found {len(competitors)} competitor(s) for '{idea}': {names}."
        f"{gap_text}"
    )
