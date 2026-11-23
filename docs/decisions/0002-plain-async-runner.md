# ADR 0002: Plain asynchronous runner

- Status: Accepted
- Date: 2026-11-20

## Decision

Use a plain asynchronous Python runner as the default orchestration approach. LangGraph may be evaluated in a bounded technical spike, but it is not a project dependency unless a later decision approves it.

## Consequences

Agent interfaces and shared schemas should remain independent of any orchestration framework.
