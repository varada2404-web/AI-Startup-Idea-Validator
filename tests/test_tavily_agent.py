"""
Automated tests for TavilySearchAgent
--------------------------------------
Run with:  pytest tests/ -v
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent.tavily_search_agent import SearchError, TavilySearchAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_TAVILY_RESULT = {
    "results": [
        {
            "title": "AI Market Report 2024",
            "url": "https://example.com/report",
            "content": "The AI market is growing rapidly with a CAGR of 35%.",
            "score": 0.95,
        },
        {
            "title": "Top AI Startups",
            "url": "https://example.com/startups",
            "content": "Several startups are competing in this space.",
            "score": 0.88,
        },
    ]
}


@pytest.fixture
def mock_tavily_client():
    """Return a mock TavilyClient that returns MOCK_TAVILY_RESULT."""
    client = MagicMock()
    client.search.return_value = MOCK_TAVILY_RESULT
    return client


@pytest.fixture
def agent(mock_tavily_client):
    """Return a TavilySearchAgent with a mocked client."""
    a = TavilySearchAgent(api_key="tvly-test-key")
    a._client = mock_tavily_client
    return a


# ---------------------------------------------------------------------------
# TavilySearchAgent — initialization
# ---------------------------------------------------------------------------

class TestAgentInitialization:
    def test_lazy_client_raises_without_api_key(self, monkeypatch):
        """Agent should raise SearchError on first use if key is missing."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        a = TavilySearchAgent(api_key=None)
        with pytest.raises(SearchError, match="TAVILY_API_KEY is not set"):
            _ = a.client

    def test_client_initialized_with_explicit_key(self):
        """Client initializes when api_key is passed explicitly."""
        with patch("agent.tavily_search_agent.TavilyClient") as MockClient:
            a = TavilySearchAgent(api_key="tvly-test-key")
            _ = a.client
            MockClient.assert_called_once_with(api_key="tvly-test-key")

    def test_client_initialized_from_env(self, monkeypatch):
        """Client initializes from TAVILY_API_KEY env var."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
        with patch("agent.tavily_search_agent.TavilyClient") as MockClient:
            a = TavilySearchAgent()
            _ = a.client
            MockClient.assert_called_once_with(api_key="tvly-env-key")

    def test_client_is_lazy(self, monkeypatch):
        """TavilyClient should NOT be created at __init__ time."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with patch("agent.tavily_search_agent.TavilyClient") as MockClient:
            TavilySearchAgent()  # should not raise or call TavilyClient
            MockClient.assert_not_called()


# ---------------------------------------------------------------------------
# TavilySearchAgent — query templates
# ---------------------------------------------------------------------------

class TestQueryTemplates:
    def test_no_hardcoded_year_in_templates(self):
        """Query templates must not contain any hardcoded year (e.g. 2024)."""
        from agent.tavily_search_agent import SEARCH_QUERIES_TEMPLATE
        for template in SEARCH_QUERIES_TEMPLATE:
            assert "2024" not in template, (
                f"Hardcoded year found in template: '{template}'"
            )

    def test_dynamic_year_in_run_output(self, agent):
        """The 'recent news' query must use the current year dynamically."""
        current_year = str(datetime.now().year)
        result = agent.run("test idea", max_results=1)
        trend_queries = [q for q in result["queries"] if "recent news" in q]
        assert trend_queries, "No 'recent news' query found"
        assert current_year in trend_queries[0], (
            f"Expected current year {current_year} in query: {trend_queries[0]}"
        )

    def test_run_generates_five_queries(self, agent):
        """run() should generate exactly 5 search queries."""
        result = agent.run("blockchain startup", max_results=1)
        assert len(result["queries"]) == 5

    def test_queries_contain_idea(self, agent):
        """Every generated query must contain the startup idea."""
        idea = "solar panel marketplace"
        result = agent.run(idea, max_results=1)
        for q in result["queries"]:
            assert idea in q, f"Query does not contain idea: '{q}'"


# ---------------------------------------------------------------------------
# TavilySearchAgent — search results
# ---------------------------------------------------------------------------

class TestSearchResults:
    def test_run_returns_expected_keys(self, agent):
        """run() result must have all required keys."""
        result = agent.run("test idea", max_results=1)
        assert set(result.keys()) >= {"idea", "queries", "results", "summary", "errors"}

    def test_results_cleaned_correctly(self, agent):
        """Each result dict must have title, url, content, score."""
        result = agent.run("test idea", max_results=2)
        for query_results in result["results"].values():
            for r in query_results:
                assert "title" in r
                assert "url" in r
                assert "content" in r
                assert "score" in r

    def test_search_single_returns_list(self, agent):
        """search_single() must return a list of result dicts."""
        results = agent.search_single("AI startup funding", max_results=2)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_empty_response_handled(self, agent, mock_tavily_client):
        """Agent handles empty results from Tavily gracefully."""
        mock_tavily_client.search.return_value = {"results": []}
        results = agent.search_single("obscure query", max_results=1)
        assert results == []


# ---------------------------------------------------------------------------
# TavilySearchAgent — error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_search_error_raised_on_api_failure(self, agent, mock_tavily_client):
        """_search() must raise SearchError when Tavily API throws."""
        mock_tavily_client.search.side_effect = RuntimeError("API timeout")
        with pytest.raises(SearchError, match="Tavily search failed"):
            agent.search_single("some query")

    def test_run_captures_per_query_errors(self, agent, mock_tavily_client):
        """run() must capture per-query errors without aborting remaining queries."""
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure on first query")
            return MOCK_TAVILY_RESULT

        mock_tavily_client.search.side_effect = side_effect
        result = agent.run("test idea", max_results=1)

        assert len(result["errors"]) == 1
        # Remaining 4 queries should still have results
        successful = [r for r in result["results"].values() if r]
        assert len(successful) == 4

    def test_all_queries_fail_gracefully(self, agent, mock_tavily_client):
        """run() must return structured errors if all queries fail."""
        mock_tavily_client.search.side_effect = RuntimeError("Network error")
        result = agent.run("test idea", max_results=1)
        assert len(result["errors"]) == 5
        for query_results in result["results"].values():
            assert query_results == []
