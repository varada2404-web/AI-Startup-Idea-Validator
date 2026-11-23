"""Deterministic offline provider. It never performs network requests."""

import asyncio
from datetime import UTC, datetime

from app.providers.search.base import SearchResult
from app.schemas.contracts import ResearchCategory

PLACEHOLDER_DETAILS = {
    "customer_pain": "friction burden queue rescheduling reminders followthrough workflow",
    "market": "context landscape segment adoption geography category terminology",
    "direct_competitors": "comparison provider catalogue feature positioning product category",
    "alternatives": "substitute workaround spreadsheet manual process incumbent route",
    "pricing": "price packaging billing tier contract monetization model",
    "recent_developments": "timeline release policy shift event dated update",
    "negative_evidence": "failure barrier constraint objection shutdown risk uncertainty",
}


class MockSearchProvider:
    async def search(
        self, query: str, category: ResearchCategory, round_number: int
    ) -> list[SearchResult]:
        # Keep the run visible long enough for the progress UI to show its stages.
        await asyncio.sleep(0.04)

        # An intentional syndicated duplicate demonstrates URL deduplication in the UI.
        source_category: ResearchCategory = (
            "direct_competitors" if category == "alternatives" and round_number == 2 else category
        )
        source_round = 1 if category == "alternatives" and round_number == 2 else round_number
        slug = source_category.replace("_", "-")
        url = f"mock://demo.invalid/{slug}/round-{source_round}"
        category_label = source_category.replace("_", " ")
        text = (
            f"SIMULATED PLACEHOLDER ONLY: {category_label}. "
            f"Scenario markers: {PLACEHOLDER_DETAILS[source_category]}. "
            f"Round {source_round}. Not a real finding; no live research."
        )
        return [
            SearchResult(
                url=url,
                domain="demo.invalid",
                title=f"SIMULATED PLACEHOLDER — {category_label.title()} — round {source_round}",
                query=query,
                category=category,
                text=text,
                retrieved_at=datetime.now(UTC),
            )
        ]
