# ADR 0005: Development providers and data

- Status: Accepted, updated 2026-11-20

## Decision

Initial development uses mock providers and synthetic startup ideas. The Gemini Agent 1 adapter may be implemented behind explicit LIVE configuration. This implementation approval does not authorize real provider calls during development or tests; all automated checks use injected mock clients and synthetic ideas. Tavily integration remains a separate phase.

## Consequences

Deterministic fixtures should be used for development checks. Before sending founder ideas to a live provider, review the provider's current retention and data-use terms and obtain the required project approval. Never include credentials in source control, logs, API responses, or run records.
