# ADR 0006: M1 estimate and interrupted runs

- Status: Open
- Date: 2026-11-20

## Context

Revision 3 mentions roughly 10 hours per milestone and marks whether this is per person or per team as `[CONFIRM]`. Its M1 task breakdown separately totals 49 hours. The estimates do not currently reconcile.

## Decision

Do not commit to an implementation schedule until the milestone-hour basis and 49-hour estimate are reconciled. Report both figures when discussing planning.

Interrupted-run handling must be described only to the level implemented and tested. The architecture's proposed interrupted status does not imply automatic recovery or resumption.

## Consequences

Any run persistence or restart behavior requires explicit implementation and tests before stronger recovery claims are made.
