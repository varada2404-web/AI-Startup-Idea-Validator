"""
Tavily Web Search Agent
-----------------------
Part of: AI-Based Startup Idea Validator with Market Analysis Assistance
Author: Varada
Role  : Tavily web search agent

This module exposes a TavilySearchAgent that accepts a startup idea
and returns structured market intelligence by running multiple targeted
Tavily searches.
"""

import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


SEARCH_QUERIES_TEMPLATE = [
    "{idea} market size and growth",
    "{idea} competitors and existing startups",
    "{idea} target audience and customer segment",
    "{idea} recent news and trends 2024",
    "{idea} challenges and risks",
]


class TavilySearchAgent:
    """
    Runs structured market research for a startup idea using Tavily Search API.

    Usage:
        agent = TavilySearchAgent()
        results = agent.run("AI-powered resume screener")
    """

    def __init__(self, api_key: str | None = None, search_depth: str = "advanced"):
        self.client = TavilyClient(api_key=api_key or os.environ["TAVILY_API_KEY"])
        self.search_depth = search_depth  # "basic" or "advanced"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, startup_idea: str, max_results: int = 5) -> dict[str, Any]:
        """
        Run all market research queries for the given startup idea.

        Args:
            startup_idea: Plain-text description of the startup idea.
            max_results: Max results per query (Tavily cap is 10).

        Returns:
            A dict with keys: idea, queries, results, summary
        """
        queries = [q.format(idea=startup_idea) for q in SEARCH_QUERIES_TEMPLATE]
        all_results: dict[str, list[dict]] = {}

        for query in queries:
            all_results[query] = self._search(query, max_results)

        return {
            "idea": startup_idea,
            "queries": queries,
            "results": all_results,
            "summary": self._build_summary(startup_idea, all_results),
        }

    def search_single(self, query: str, max_results: int = 5) -> list[dict]:
        """Run a single arbitrary search query and return cleaned results."""
        return self._search(query, max_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int) -> list[dict]:
        """Call Tavily and return a cleaned list of result dicts."""
        try:
            response = self.client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=max_results,
                include_answer=True,
            )
            results = []
            for r in response.get("results", []):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                    }
                )
            return results
        except Exception as exc:
            return [{"error": str(exc), "query": query}]

    def _build_summary(
        self, idea: str, all_results: dict[str, list[dict]]
    ) -> dict[str, str]:
        """
        Build a lightweight text summary per category from search results.
        The LLM extraction endpoint (Prajna's part) will do deeper processing;
        this gives a quick readable digest.
        """
        summary: dict[str, str] = {}
        for query, results in all_results.items():
            category = query.replace(idea, "").strip().title()
            snippets = [r.get("content", "") for r in results if "error" not in r]
            summary[category] = " | ".join(snippets[:2]) if snippets else "No data found."
        return summary
