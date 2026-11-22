"""Bounded seven-category query planning for M1."""

from app.schemas.contracts import IdeaBrief, QueryPlanItem, ResearchCategory

CATEGORY_ORDER: tuple[ResearchCategory, ...] = (
    "customer_pain",
    "market",
    "direct_competitors",
    "alternatives",
    "pricing",
    "recent_developments",
    "negative_evidence",
)

CATEGORY_QUESTIONS = {
    "customer_pain": "customer problem and pain points",
    "market": "market context and relevant industry information",
    "direct_competitors": "direct competitor products and services",
    "alternatives": "workarounds substitutes and alternatives",
    "pricing": "pricing models and willingness to pay signals",
    "recent_developments": "recent developments and changes",
    "negative_evidence": "why similar ideas may fail or face barriers",
}


def plan_initial_queries(brief: IdeaBrief) -> list[QueryPlanItem]:
    topic = " ".join(brief.search_keywords[:4])
    return [
        QueryPlanItem(category=category, query=f"{topic}: {CATEGORY_QUESTIONS[category]}")
        for category in CATEGORY_ORDER
    ]


def plan_follow_up_queries(
    brief: IdeaBrief, missing_categories: list[ResearchCategory], limit: int = 4
) -> list[QueryPlanItem]:
    topic = " ".join(brief.search_keywords[:4])
    return [
        QueryPlanItem(
            category=category,
            query=f"specific evidence about {topic}: {CATEGORY_QUESTIONS[category]}",
            round=2,
        )
        for category in missing_categories[:limit]
    ]
