"""Structured API error responses."""

from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    del request
    errors = exc.errors()
    message = errors[0].get("msg", "Request validation failed") if errors else "Invalid request"
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "message": message, "stage": "request"}},
    )


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    del request
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": str(detail.get("code", "http_error")),
                "message": str(detail.get("message", "The request could not be completed.")),
                "stage": "api",
            }
        },
    )
