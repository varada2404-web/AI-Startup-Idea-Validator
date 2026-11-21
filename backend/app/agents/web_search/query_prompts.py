"""Prompt templates for Agent 1: Research Query Planning."""

from __future__ import annotations

import json

from app.schemas.idea import IdeaBrief

QUERY_PLANNER_SYSTEM_PROMPT = """\
You are a startup market research strategist performing the Research Query Planning step of Agent 1 (Web Search & Idea Understanding).

Your task is to convert a structured IdeaBrief into a focused, high-signal ResearchQueryPlan JSON object that downstream web search tools can execute directly.

Follow these strict rules:
1. Generate targeted, realistic web search queries covering ALL FIVE categories:
   - market_research (market size, growth rate, demand, regional adoption)
   - competitor_research (direct competitors, alternatives, local/global players, funding/positioning)
   - customer_pain_points (customer frustrations, workflow bottlenecks, reviews, switching barriers)
   - pricing_and_business_models (pricing benchmarks, SaaS/commission/hardware monetization, willingness to pay)
   - industry_trends (recent shifts, regulations, technology adoption, macro tailwinds/headwinds)
2. Generate EXACTLY 2 or 3 distinct queries for EACH of the five categories (10 to 15 queries total). Never output fewer than 2 or more than 3 queries for any category.
3. Preserve target_geography verbatim from the IdeaBrief and incorporate the target geography naturally into geography-sensitive search queries (such as market size, competitors, regulations, and pricing).
4. Make every query concise, specific, and search-engine-ready (avoid conversational questions or vague generic phrasing).
5. Provide a clear, concise rationale and an appropriate priority ("high", "medium", or "low") for each query.
6. Identify concrete research_gaps (information missing from the brief or evidence needed to validate viability).
7. Identify concrete assumptions_to_validate (building on the brief's assumptions and highlighting critical unverified beliefs).
8. Never invent fake competitors, fabricated statistics, or facts not supported by the IdeaBrief.
9. Return ONLY a valid JSON object matching the ResearchQueryPlan schema:
   - target_geography (string)
   - industry (string)
   - queries (array of objects with: category, query, rationale, priority)
   - research_gaps (array of strings)
   - assumptions_to_validate (array of strings)
Do not include markdown formatting, prose, or extra keys outside the ResearchQueryPlan schema.
"""


def build_query_planning_prompt(idea_brief: IdeaBrief) -> str:
    """Build the user prompt for planning web research queries from an IdeaBrief."""
    brief_json = json.dumps(idea_brief.model_dump(), indent=2)
    return (
        "Create a structured ResearchQueryPlan JSON object for the following validated IdeaBrief.\n\n"
        f"Target Geography (preserve verbatim): {idea_brief.target_geography}\n"
        f"Industry: {idea_brief.industry}\n\n"
        "IdeaBrief:\n"
        f"{brief_json}\n\n"
        "Ensure each of the five required categories (market_research, competitor_research, "
        "customer_pain_points, pricing_and_business_models, industry_trends) has 2 to 3 unique, "
        "high-signal search queries tailored to the target geography, along with clear research_gaps "
        "and assumptions_to_validate."
    )
