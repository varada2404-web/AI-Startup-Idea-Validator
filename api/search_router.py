"""
FastAPI router — Tavily Search Agent endpoint
----------------------------------------------
Exposes POST /api/search so other team members (Prajna's LLM extraction
endpoint, Harshita's architecture layer) can call it over HTTP.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent import TavilySearchAgent

router = APIRouter(prefix="/api", tags=["search"])

# Single shared agent instance (re-uses the Tavily client connection)
_agent = TavilySearchAgent()


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
def run_market_research(payload: SearchRequest):
    """
    Run structured market research for a startup idea.
    Returns market size, competitors, target audience, trends, and risks.
    """
    try:
        result = _agent.run(payload.idea, payload.max_results)
        return {"status": "success", "data": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/search/single")
def run_single_search(payload: SingleSearchRequest):
    """
    Run a single arbitrary Tavily search query.
    Useful for Prajna's LLM endpoint to do ad-hoc follow-up queries.
    """
    try:
        results = _agent.search_single(payload.query, payload.max_results)
        return {"status": "success", "query": payload.query, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
def health_check():
    return {"status": "ok", "agent": "TavilySearchAgent"}
