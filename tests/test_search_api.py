"""
Automated tests for Search API endpoints
-----------------------------------------
Run with:  pytest tests/ -v
"""

from unittest.mock import MagicMock

import pytest
from agent.tavily_search_agent import SearchError, TavilySearchAgent
from api.search_router import _get_agent, get_agent
from fastapi.testclient import TestClient

from main import app

MOCK_RUN_RESULT = {
    "idea": "AI resume screener",
    "queries": ["q1", "q2", "q3", "q4", "q5"],
    "results": {"q1": [{"title": "T", "url": "u", "content": "c", "score": 0.9}]},
    "summary": {"Q1": "some summary"},
    "errors": {},
}

MOCK_SINGLE_RESULT = [
    {"title": "Result", "url": "https://example.com", "content": "content", "score": 0.8}
]


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=TavilySearchAgent)
    agent.run.return_value = MOCK_RUN_RESULT
    agent.search_single.return_value = MOCK_SINGLE_RESULT
    return agent


@pytest.fixture
def client(mock_agent):
    """TestClient with the agent dependency overridden."""
    app.dependency_overrides[get_agent] = lambda: mock_agent
    _get_agent.cache_clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    def test_valid_request_returns_success(self, client):
        response = client.post("/api/search", json={"idea": "AI resume screener"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["idea"] == "AI resume screener"

    def test_idea_too_short_returns_422(self, client):
        response = client.post("/api/search", json={"idea": "AI"})
        assert response.status_code == 422

    def test_max_results_out_of_range_returns_422(self, client):
        response = client.post("/api/search", json={"idea": "valid idea", "max_results": 0})
        assert response.status_code == 422

    def test_missing_idea_returns_422(self, client):
        response = client.post("/api/search", json={})
        assert response.status_code == 422

    def test_search_error_returns_503(self, client, mock_agent):
        mock_agent.run.side_effect = SearchError("API key missing")
        response = client.post("/api/search", json={"idea": "test idea"})
        assert response.status_code == 503
        assert "API key missing" in response.json()["detail"]

    def test_unexpected_error_returns_500(self, client, mock_agent):
        mock_agent.run.side_effect = RuntimeError("unexpected crash")
        response = client.post("/api/search", json={"idea": "test idea"})
        assert response.status_code == 500

    def test_partial_errors_include_warning(self, client, mock_agent):
        result_with_errors = {**MOCK_RUN_RESULT, "errors": {"q1": "timeout"}}
        mock_agent.run.return_value = result_with_errors
        response = client.post("/api/search", json={"idea": "AI resume screener"})
        assert response.status_code == 200
        assert "warning" in response.json()["data"]


# ---------------------------------------------------------------------------
# POST /api/search/single
# ---------------------------------------------------------------------------

class TestSingleSearchEndpoint:
    def test_valid_query_returns_results(self, client):
        response = client.post(
            "/api/search/single",
            json={"query": "AI market size 2025"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert isinstance(body["results"], list)

    def test_query_too_short_returns_422(self, client):
        response = client.post("/api/search/single", json={"query": "AI"})
        assert response.status_code == 422

    def test_search_error_returns_503(self, client, mock_agent):
        mock_agent.search_single.side_effect = SearchError("rate limited")
        response = client.post(
            "/api/search/single", json={"query": "valid query here"}
        )
        assert response.status_code == 503
