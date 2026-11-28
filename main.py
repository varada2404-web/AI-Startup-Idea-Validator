"""
Entry point — Startup Idea Validator: Tavily Search Agent Service
-----------------------------------------------------------------
Run with:  uvicorn main:app --reload --port 8001
"""

from fastapi import FastAPI

from api import router

app = FastAPI(
    title="Startup Idea Validator — Tavily Search Agent",
    description=(
        "Performs real-time web research on a startup idea using Tavily Search API. "
        "Returns market size, competitors, target audience, trends, and risks."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Tavily Search Agent",
        "docs": "/docs",
        "health": "/api/health",
    }
