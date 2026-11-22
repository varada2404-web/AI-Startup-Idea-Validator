# ADR 0007: Tavily Search and Extraction for Agent 1

## Status

Accepted for the M1 LIVE provider implementation.

## Decision

Use Tavily's documented HTTPS Search and Extract API endpoints through the existing asynchronous HTTPX client. Keep search and extraction as separate typed operations behind the Agent 1 provider interface. DEMO remains the default and continues to use the deterministic mock provider. LIVE mode explicitly selects Gemini and Tavily and never falls back to mocks.

The deterministic query planner remains authoritative. A run starts with seven category queries, can add at most four follow-ups, accepts at most five results per query, limits concurrent provider calls to four, and selects at most eight unique URLs for extraction. Snippets and extracted page content remain distinct and are both bounded before persistence.

Transient network, timeout, server, and rate-limit errors receive a configurable bounded retry. The adapter respects `Retry-After` and will not retry earlier than requested. Authentication and provider failures are sanitized into typed errors. Partial search failures preserve successful sources and degrade the run; if no search succeeds, the research stage fails. Extraction failures preserve the search record and snippet.

## Scope

This does not implement Agents 2–6 or full evidence verification. Retrieval and extraction are not evidence verification and do not establish that a claim is true. Retrieved content remains untrusted data.

## Consequences

- HTTPX becomes a runtime dependency because the live provider performs asynchronous requests.
- LIVE configuration requires Gemini and Tavily credentials plus the Gemini model identifier.
- Provider response mapping can be tested with mocked HTTP responses without network access or credentials.
- The existing endpoints and response models remain stable; live runs populate the existing mode, provider-error, source-provenance, snippet, and extraction fields.
