# AI Startup Idea Validator

An evidence-first startup idea validation project. The approved architecture defines six sequential agents. Milestone 1 is limited to project foundation and Agent 1 work when separately approved; this Phase 1 foundation does not implement application behavior or any agent.

## Project status

This repository currently contains project setup, documentation, and directory skeletons only. No backend, frontend behavior, provider integration, API, or automated application tests have been implemented.

Revision 3 of the architecture document is the project source of truth. Its original copy is kept at [`docs/architecture/AI-Startup-Idea-Validator-Revision-3.pdf`](docs/architecture/AI-Startup-Idea-Validator-Revision-3.pdf).

## Development setup

- Python 3.11 or newer
- Ruff and Pytest for local static and test checks

No provider credentials or runtime packages are needed for the current foundation. Copy `.env.example` to `.env` only when local settings are introduced; keep real values out of source control.

## Checks

Run from the repository root:

```powershell
python -m ruff check .
python -m pytest
```

The current skeleton has no application tests or functionality. These commands may report that the tools are unavailable or that no tests are collected until the project adds them.

## Scope and decisions

- Preserve the six-agent architecture; Agent 1 is the only agent in M1.
- Use a plain asynchronous Python runner as the default. LangGraph is only a technical-spike candidate.
- Start with HTML, CSS, and JavaScript ES modules. Do not add React or Vite without approval.
- M1's evidence work is limited to source storage, provenance, quote-location utilities, and reference-integrity foundations. Full evidence verification stays in M2 unless the scope is explicitly revised.
- Development should use mock providers and synthetic startup ideas. No paid API calls are approved.
- The M1 estimate remains unresolved: the architecture lists roughly 10 hours per milestone (per-person versus per-team needs confirmation) and separately estimates 49 hours for M1 core tasks.
- Interrupted runs must not be described as automatically recoverable unless that behavior is implemented and tested.

See [`docs/decisions/`](docs/decisions/) for the initial architecture decision records.
