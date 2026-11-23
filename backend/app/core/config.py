"""Validated application and provider configuration."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

ApplicationMode = Literal["demo", "live"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ApplicationConfig(BaseModel):
    """Settings shared by demo and live provider wiring.

    Demo remains the safe default. Secret fields are redacted by Pydantic when represented.
    """

    model_config = ConfigDict(frozen=True)

    mode: ApplicationMode = "demo"
    gemini_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    gemini_model: str | None = None
    gemini_timeout_seconds: float = Field(default=30, gt=0, le=120)
    gemini_max_output_tokens: int = Field(default=2048, ge=256, le=4096)
    tavily_timeout_seconds: float = Field(default=20, gt=0, le=120)
    tavily_max_retries: int = Field(default=1, ge=0, le=2)
    tavily_max_retry_delay_seconds: float = Field(default=10, ge=0, le=30)
    max_provider_response_bytes: int = Field(default=2_000_000, ge=1_024, le=16_000_000)
    max_search_results_per_query: int = Field(default=5, ge=1, le=5)
    max_follow_up_queries: int = Field(default=4, ge=0, le=4)
    max_extract_sources_per_run: int = Field(default=8, ge=0, le=8)
    max_concurrent_provider_calls: int = Field(default=4, ge=1, le=4)
    max_active_live_runs: int = Field(default=1, ge=1, le=8)
    max_source_text_chars: int = Field(default=12_000, ge=500, le=20_000)
    cors_allowed_origins: tuple[str, ...] = ()
    database_path: Path = PROJECT_ROOT / "backend" / "data" / "demo_runs.sqlite3"

    @model_validator(mode="after")
    def live_mode_requires_credentials(self) -> "ApplicationConfig":
        if self.mode == "live":
            missing = [
                name
                for name, value in (
                    ("GEMINI_API_KEY", self.gemini_api_key),
                    ("TAVILY_API_KEY", self.tavily_api_key),
                )
                if value is None or not value.get_secret_value().strip()
            ]
            if self.gemini_model is None or not self.gemini_model.strip():
                missing.append("GEMINI_MODEL")
            if missing:
                names = ", ".join(missing)
                raise ValueError(f"LIVE mode requires configuration values: {names}.")
        return self


def load_config(environ: dict[str, str] | None = None) -> ApplicationConfig:
    """Load supported environment settings without logging credential values."""

    values = os.environ if environ is None else environ
    raw_mode = values.get("VALIDATOR_MODE", "demo").strip().lower()
    raw_origins = values.get("CORS_ALLOWED_ORIGINS", "")
    raw_database_path = values.get("VALIDATOR_DB_PATH")
    return ApplicationConfig(
        mode=raw_mode,  # type: ignore[arg-type]
        gemini_api_key=values.get("GEMINI_API_KEY") or None,
        tavily_api_key=values.get("TAVILY_API_KEY") or None,
        gemini_model=values.get("GEMINI_MODEL") or None,
        gemini_timeout_seconds=float(
            values.get("GEMINI_TIMEOUT_SECONDS", values.get("PROVIDER_TIMEOUT_SECONDS", "30"))
        ),
        gemini_max_output_tokens=int(values.get("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
        tavily_timeout_seconds=float(values.get("TAVILY_TIMEOUT_SECONDS", "20")),
        tavily_max_retries=int(values.get("TAVILY_MAX_RETRIES", "1")),
        tavily_max_retry_delay_seconds=float(values.get("TAVILY_MAX_RETRY_DELAY_SECONDS", "10")),
        max_provider_response_bytes=int(values.get("MAX_PROVIDER_RESPONSE_BYTES", "2000000")),
        max_search_results_per_query=int(values.get("MAX_SEARCH_RESULTS_PER_QUERY", "5")),
        max_follow_up_queries=int(values.get("MAX_FOLLOW_UP_QUERIES", "4")),
        max_extract_sources_per_run=int(values.get("MAX_EXTRACT_SOURCES_PER_RUN", "8")),
        max_concurrent_provider_calls=int(values.get("MAX_CONCURRENT_PROVIDER_CALLS", "4")),
        max_active_live_runs=int(values.get("MAX_ACTIVE_LIVE_RUNS", "1")),
        max_source_text_chars=int(values.get("MAX_SOURCE_TEXT_CHARS", "12000")),
        cors_allowed_origins=tuple(
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ),
        database_path=Path(raw_database_path) if raw_database_path else PROJECT_ROOT
        / "backend"
        / "data"
        / "demo_runs.sqlite3",
    )


def database_path() -> Path:
    """Backward-compatible path helper for current app wiring."""

    return load_config().database_path
