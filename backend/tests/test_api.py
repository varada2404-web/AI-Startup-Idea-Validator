"""API contract tests using the real FastAPI application and local demo pipeline."""

import time

from fastapi.testclient import TestClient

from app.core.config import ApplicationConfig
from app.main import create_app


def test_health_and_complete_validation_flow(tmp_path) -> None:
    app = create_app(tmp_path / "api-runs.sqlite3")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "IdeaSignal" in page.text
        assert client.get("/css/styles.css").status_code == 200
        assert client.get("/js/app.js").status_code == 200

        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "mode": "DEMO", "live_research": False}

        submitted = client.post(
            "/api/v1/validations",
            json={
                "idea": "A tool that helps local clinics manage appointment reminders.",
                "target_customer": "Independent clinics",
            },
        )
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]

        for _ in range(100):
            result = client.get(f"/api/v1/validations/{run_id}")
            assert result.status_code == 200
            if result.json()["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)

        payload = result.json()
        assert payload["status"] == "completed"
        assert payload["demo_mode"] is True
        assert len(payload["idea_brief"]["hypotheses"]) == 5
        assert len(payload["queries"]) == 11
        assert len(payload["coverage"]) == 7
        assert payload["sources"]
        assert all(source["simulated"] is True for source in payload["sources"])


def test_invalid_request_and_unknown_run_use_structured_errors(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "api-errors.sqlite3")) as client:
        invalid = client.post("/api/v1/validations", json={"idea": "too short"})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_request"

        missing = client.get("/api/v1/validations/not-a-run")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "run_not_found"


def test_live_active_run_limit_rejects_excess_without_provider_calls(tmp_path) -> None:
    config = ApplicationConfig(
        mode="live",
        gemini_api_key="test-placeholder",
        tavily_api_key="test-placeholder",
        gemini_model="test-model",
        max_active_live_runs=1,
    )
    app = create_app(tmp_path / "api-live-limit.sqlite3", config=config)

    class WaitingRunner:
        async def execute(self, _run_id: str) -> None:
            import asyncio

            await asyncio.sleep(0.3)

    app.state.pipeline_runner = WaitingRunner()
    request = {"idea": "A local service for small clinics to reduce missed appointments."}
    with TestClient(app) as client:
        first = client.post("/api/v1/validations", json=request)
        second = client.post("/api/v1/validations", json=request)

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "live_run_limit"
