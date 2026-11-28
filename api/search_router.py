"""
FastAPI router — Tavily Search Agent endpoint
----------------------------------------------
Exposes POST /api/search so other team members (Prajna's LLM extraction
endpoint, Harshita's architecture layer) can call it over HTTP.
"""

from functools import lru_cache

from agent import TavilySearchAgent
from agent.tavily_search_agent import SearchError
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["search"])


# ---------------------------------------------------------------------------
# Dependency injection — agent initialized per-request, not at import time
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_agent() -> TavilySearchAgent:
    """
    Return a shared TavilySearchAgent instance.
    Initialized on first request, not at module load time.
    If TAVILY_API_KEY is missing the error surfaces at request time,
    not at application startup.
    """
    return TavilySearchAgent()


def get_agent() -> TavilySearchAgent:
    return _get_agent()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    idea: str = Field(..., min_length=3, description="Startup idea to research")
    max_results: int = Field(default=5, ge=1, le=10)


class SingleSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Arbitrary search query")
    max_results: int = Field(default=5, ge=1, le=10)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/search")
def run_market_research(
    payload: SearchRequest,
    agent: TavilySearchAgent = Depends(get_agent),  # noqa: B008
):
    """
    Run structured market research for a startup idea.
    Returns market size, competitors, target audience, trends, and risks.
    """
    try:
        result = agent.run(payload.idea, payload.max_results)
    except SearchError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    # Surface per-query errors as a warning but still return partial results
    if result.get("errors"):
        result["warning"] = "Some queries failed. See 'errors' field for details."

    return {"status": "success", "data": result}


@router.post("/search/single")
def run_single_search(
    payload: SingleSearchRequest,
    agent: TavilySearchAgent = Depends(get_agent),  # noqa: B008
):
    """
    Run a single arbitrary Tavily search query.
    Useful for Prajna's LLM endpoint to do ad-hoc follow-up queries.
    """
    try:
        results = agent.search_single(payload.query, payload.max_results)
    except SearchError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    return {"status": "success", "query": payload.query, "results": results}


@router.get("/health")
def health_check():
    return {"status": "ok", "agent": "TavilySearchAgent"}
