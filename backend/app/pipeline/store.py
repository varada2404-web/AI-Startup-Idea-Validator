"""Small SQLite run store for the local M1 prototype."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.contracts import ValidationRun


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS validation_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, run: ValidationRun) -> None:
        payload = json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO validation_runs(run_id, status, created_at, payload)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     status=excluded.status, payload=excluded.payload""",
                (run.run_id, run.status, run.created_at.isoformat(), payload),
            )

    def get(self, run_id: str) -> ValidationRun | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM validation_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ValidationRun.model_validate_json(row["payload"])

    def mark_orphaned_runs_interrupted(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM validation_runs WHERE status IN ('queued', 'running')"
            ).fetchall()
        interrupted = 0
        for row in rows:
            run = ValidationRun.model_validate_json(row["payload"])
            run.status = "interrupted"
            run.updated_at = datetime.now(UTC)
            run.error = (
                "The server restarted before this run completed. "
                "Start a new validation run to continue."
            )
            for stage in run.progress:
                if stage.status == "running":
                    stage.status = "interrupted"
            self.save(run)
            interrupted += 1
        return interrupted
