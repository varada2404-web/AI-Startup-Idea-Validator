"""FastAPI application factory and local demo entry point."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import ApplicationConfig, load_config
from app.core.errors import http_error_handler, validation_error_handler
from app.pipeline.runner import PipelineRunner
from app.pipeline.store import RunStore
from app.providers.llm.factory import create_idea_understanding_provider
from app.providers.search.factory import create_search_provider


def create_app(
    db_path: Path | None = None, config: ApplicationConfig | None = None
) -> FastAPI:
    settings = config or load_config()
    store = RunStore(db_path or settings.database_path)
    store.mark_orphaned_runs_interrupted()
    runner = PipelineRunner(
        store=store,
        provider=create_search_provider(settings),
        interpreter=create_idea_understanding_provider(settings),
        max_concurrent_calls=settings.max_concurrent_provider_calls,
        max_follow_up_queries=settings.max_follow_up_queries,
        max_extract_sources=settings.max_extract_sources_per_run,
        max_source_text_chars=settings.max_source_text_chars,
        provider_semaphore=asyncio.Semaphore(settings.max_concurrent_provider_calls),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.tasks = {}
        yield
        tasks = list(app.state.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for provider in (runner.provider, runner.interpreter):
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()

    app = FastAPI(
        title="AI Startup Idea Validator — Agent 1",
        version="0.1.0",
        description="M1 Agent 1 startup idea understanding and research workflow.",
        lifespan=lifespan,
    )
    app.state.run_store = store
    app.state.pipeline_runner = runner
    app.state.config = settings
    app.state.tasks = {}
    app.state.live_submission_lock = asyncio.Lock()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            *settings.cors_allowed_origins,
            "http://127.0.0.1:5500",
            "http://localhost:5500",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.include_router(router)
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


app = create_app()
