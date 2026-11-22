# AI Startup Idea Validator

An evidence-first startup idea validation project. The approved architecture defines six sequential agents. M1 implements Agent 1 only. DEMO mode uses deterministic mock providers; explicit LIVE mode uses Gemini for idea interpretation and Tavily for web search and selective extraction.

## Project status

The M1 flow provides idea submission, a structured idea brief, five unassessed hypotheses, deterministic seven-category query planning, bounded two-round research, URL deduplication, selective page extraction, SQLite run storage, progress polling, and a browser results dashboard. LIVE sources preserve search snippets separately from extracted text and include retrieval provenance. Search or extraction errors are reported without fabricated sources. No market sizing, competitor conclusions, SWOT, MVP recommendations, full evidence verification, or verdict are produced.

Revision 3 of the architecture document is the project source of truth. Its original copy is kept at [`docs/architecture/AI-Startup-Idea-Validator-Revision-3.pdf`](docs/architecture/AI-Startup-Idea-Validator-Revision-3.pdf).

## Run locally

Requires Python 3.11 or newer. From PowerShell at the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install "fastapi>=0.115,<1" "google-genai>=1.0,<2" "pydantic>=2,<3" "uvicorn>=0.32,<1" "httpx>=0.27,<1" "pytest>=8,<10" "ruff>=0.11,<1"
```

Start the application in one terminal:

```powershell
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. FastAPI serves the HTML, CSS, and JavaScript frontend from the same local process as the API. The default local database is `backend/data/demo_runs.sqlite3` (ignored by Git). `VALIDATOR_DB_PATH` can override its location. No API keys or external service access are required.

The API is available under `http://127.0.0.1:8000/api/v1`; interactive docs are at `/docs`. DEMO is the default. To use LIVE mode, configure `VALIDATOR_MODE=live`, `GEMINI_API_KEY`, `GEMINI_MODEL`, and `TAVILY_API_KEY` in the server environment. `.env.example` lists provider timeouts, retry policy, query/result/extraction bounds, and source text limits. The application does not load `.env` automatically. LIVE mode makes external provider requests only when a validation is submitted; tests use mocked responses and make no provider requests.

## Tests and lint

```powershell
python -m ruff check .
python -m pytest
```

Tests use the real FastAPI app and deterministic mock provider. The API exposes:

- `GET /api/v1/health`
- `POST /api/v1/validations` (returns `202` with a `run_id`)
- `GET /api/v1/validations/{run_id}` (poll run state and results)

## Scope and decisions

- Preserve the six-agent architecture; Agent 1 is the only agent in M1.
- Use a plain asynchronous Python runner as the default. LangGraph is only a technical-spike candidate.
- Start with HTML, CSS, and JavaScript ES modules. Do not add React or Vite without approval.
- M1's evidence work is limited to source storage, provenance, quote-location utilities, and reference-integrity foundations. Full evidence verification stays in M2 unless the scope is explicitly revised.
- DEMO remains the default and uses mock providers and synthetic startup ideas. Automated tests use injected provider responses; no paid API calls are made.
- The M1 estimate remains unresolved: the architecture lists roughly 10 hours per milestone (per-person versus per-team needs confirmation) and separately estimates 49 hours for M1 core tasks.
- Interrupted runs must not be described as automatically recoverable unless that behavior is implemented and tested.

See [`docs/decisions/`](docs/decisions/) for the initial architecture decision records.
