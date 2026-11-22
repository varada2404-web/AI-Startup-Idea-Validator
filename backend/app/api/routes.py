"""M1 validation and health endpoints."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.pipeline.runner import PIPELINE_STAGES, PipelineRunner
from app.pipeline.store import RunStore
from app.schemas.contracts import (
    ProgressItem,
    ValidationAccepted,
    ValidationRequest,
    ValidationRun,
)

router = APIRouter(prefix="/api/v1")


@router.post(
    "/validations",
    response_model=ValidationAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_validation(payload: ValidationRequest, request: Request) -> ValidationAccepted:
    store: RunStore = request.app.state.run_store
    runner: PipelineRunner = request.app.state.pipeline_runner
    mode = request.app.state.config.mode
    if mode == "live":
        async with request.app.state.live_submission_lock:
            active = sum(not task.done() for task in request.app.state.tasks.values())
            if active >= request.app.state.config.max_active_live_runs:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "code": "live_run_limit",
                        "message": (
                            "The maximum number of active LIVE validations has been reached. "
                            "Wait for the active run to finish, then try again."
                        ),
                    },
                )
            return _enqueue_validation(payload, request, store, runner)
    return _enqueue_validation(payload, request, store, runner)


def _enqueue_validation(
    payload: ValidationRequest,
    request: Request,
    store: RunStore,
    runner: PipelineRunner,
) -> ValidationAccepted:
    mode = request.app.state.config.mode
    now = datetime.now(UTC)
    run_id = str(uuid4())
    run = ValidationRun(
        run_id=run_id,
        created_at=now,
        updated_at=now,
        request=payload,
        mode=request.app.state.config.mode,
        demo_mode=request.app.state.config.mode == "demo",
        simulation_notice=(
            "DEMO MODE: results are deterministic synthetic placeholders. "
            "No live search or market research was performed."
            if request.app.state.config.mode == "demo"
            else None
        ),
        progress=[
            ProgressItem(stage=key, label=label)
            for key, label in PIPELINE_STAGES
            if request.app.state.config.mode == "live" or key != "source_extraction"
        ],
    )
    store.save(run)
    task = asyncio.create_task(runner.execute(run_id))
    request.app.state.tasks[run_id] = task
    task.add_done_callback(lambda _task: request.app.state.tasks.pop(run_id, None))
    return ValidationAccepted(run_id=run_id, status="queued", mode=mode, demo_mode=mode == "demo")


@router.get("/validations/{run_id}", response_model=ValidationRun)
async def get_validation(run_id: str, request: Request) -> ValidationRun:
    run = request.app.state.run_store.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "run_not_found", "message": "No run exists with that ID."},
        )
    return run


@router.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    mode = request.app.state.config.mode
    return {"status": "ok", "mode": mode.upper(), "live_research": mode == "live"}
