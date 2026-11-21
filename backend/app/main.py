"""FastAPI application entry point for the AI Startup Idea Validator."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.agents.web_search.agent1_runner import (
    Agent1Runner,
    Agent1SearchFailedError,
)
from app.agents.web_search.evidence_packager import EvidencePackager
from app.agents.web_search.idea_analyzer import (
    IdeaAnalyzer,
    IdeaValidationError,
    InvalidModelOutputError,
)
from app.agents.web_search.query_planner import (
    InvalidQueryPlanOutputError,
    QueryPlanner,
    QueryPlanValidationError,
)
from app.agents.web_search.search_executor import SearchExecutor
from app.providers.gemini import (
    GeminiConfigError,
    GeminiInvalidResponseError,
    GeminiProvider,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from app.providers.groq import GroqProvider
from app.providers.llm_router import (
    LLMAllProvidersFailedError,
    LLMExecutionMetadata,
    LLMFallbackService,
)
from app.providers.tavily import (
    TavilyAuthError,
    TavilyConfigError,
    TavilyProvider,
    TavilyProviderError,
)
from app.schemas.agent1 import Agent1RunResponse
from app.schemas.evidence import EvidencePackage
from app.schemas.idea import IdeaBrief, StartupIdeaInput, StartupIdeaResponse
from app.schemas.research import (
    ResearchPlanResponse,
    ResearchQueryPlan,
    SearchExecutionReport,
)

# Load environment variables from a local .env file if present (never overriding real env vars).
_BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_BACKEND_ENV_PATH, override=False)
load_dotenv(override=False)

app = FastAPI(
    title="AI Startup Idea Validator",
    description="Backend API for the startup validation system",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    """Response model for the root health check endpoint."""

    message: str = Field(..., description="API availability status message.")


def _apply_llm_metadata_headers(
    response: Response,
    metadata: LLMExecutionMetadata | None,
) -> None:
    """Attach safe, secret-free LLM execution metadata to HTTP response headers."""
    if metadata is None:
        return
    response.headers["X-LLM-Provider"] = metadata.provider
    response.headers["X-LLM-Model"] = metadata.model
    response.headers["X-LLM-Fallback-Used"] = str(metadata.fallback_used).lower()


def get_gemini_provider() -> GeminiProvider:
    """Dependency provider for the shared GeminiProvider instance."""
    return GeminiProvider()


def get_groq_provider() -> GroqProvider:
    """Dependency provider for the shared GroqProvider fallback instance."""
    return GroqProvider()


def get_llm_service(
    gemini_provider: GeminiProvider = Depends(get_gemini_provider),
    groq_provider: GroqProvider = Depends(get_groq_provider),
) -> LLMFallbackService:
    """Dependency provider for the shared Gemini -> Groq LLMFallbackService."""
    return LLMFallbackService(
        primary_provider=gemini_provider,
        fallback_provider=groq_provider,
        enable_fallback=True,
    )


def get_idea_analyzer(
    llm_service: LLMFallbackService = Depends(get_llm_service),
) -> IdeaAnalyzer:
    """Dependency provider for the IdeaAnalyzer instance."""
    return IdeaAnalyzer(llm_service=llm_service)


def get_query_planner(
    llm_service: LLMFallbackService = Depends(get_llm_service),
) -> QueryPlanner:
    """Dependency provider for the QueryPlanner instance."""
    return QueryPlanner(llm_service=llm_service)


def get_tavily_provider() -> TavilyProvider:
    """Dependency provider for the TavilyProvider instance."""
    return TavilyProvider()


def get_search_executor(
    provider: TavilyProvider = Depends(get_tavily_provider),
) -> SearchExecutor:
    """Dependency provider for the SearchExecutor instance."""
    return SearchExecutor(provider=provider)


def get_evidence_packager() -> EvidencePackager:
    """Dependency provider for the deterministic EvidencePackager instance."""
    return EvidencePackager()


def get_agent1_runner(
    idea_analyzer: IdeaAnalyzer = Depends(get_idea_analyzer),
    query_planner: QueryPlanner = Depends(get_query_planner),
    search_executor: SearchExecutor = Depends(get_search_executor),
    evidence_packager: EvidencePackager = Depends(get_evidence_packager),
) -> Agent1Runner:
    """Dependency provider for the per-request Agent1Runner orchestrator."""
    return Agent1Runner(
        idea_analyzer=idea_analyzer,
        query_planner=query_planner,
        search_executor=search_executor,
        evidence_packager=evidence_packager,
    )


@app.get("/", response_model=HealthResponse, tags=["Health"])
def read_root() -> HealthResponse:
    """Return a basic health check message confirming the API is running."""
    return HealthResponse(message="AI Startup Idea Validator API is running")


@app.post("/api/v1/ideas", response_model=StartupIdeaResponse, tags=["Ideas"])
def submit_startup_idea(payload: StartupIdeaInput) -> StartupIdeaResponse:
    """Accept and validate a startup idea submission."""
    return StartupIdeaResponse(
        message="Startup idea received successfully",
        idea=payload.idea,
        target_geography=payload.target_geography,
    )


@app.post("/api/v1/ideas/analyze", response_model=IdeaBrief, tags=["Ideas"])
def analyze_startup_idea_endpoint(
    payload: StartupIdeaInput,
    response: Response,
    analyzer: IdeaAnalyzer = Depends(get_idea_analyzer),
) -> IdeaBrief:
    """Analyze a submitted startup idea using Gemini (with Groq fallback) and return IdeaBrief."""
    try:
        brief = analyzer.analyze(payload)
        _apply_llm_metadata_headers(response, analyzer.last_execution_metadata)
        return brief
    except LLMAllProvidersFailedError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={
                "error": "all_llm_providers_failed",
                "message": "Both primary and fallback AI analysis providers failed.",
                "attempts": exc.attempts,
            },
        ) from exc
    except GeminiConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis service is not configured. Ensure GEMINI_API_KEY is set.",
        ) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI analysis request timed out. Please try again.",
        ) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI analysis service is currently unavailable. Please try again later.",
        ) from exc
    except (GeminiInvalidResponseError, InvalidModelOutputError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI analysis service returned an unreadable response.",
        ) from exc
    except IdeaValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI analysis response did not match the expected IdeaBrief schema.",
        ) from exc


@app.post("/api/v1/ideas/plan-research", response_model=ResearchPlanResponse, tags=["Ideas"])
def plan_research_endpoint(
    payload: StartupIdeaInput,
    response: Response,
    analyzer: IdeaAnalyzer = Depends(get_idea_analyzer),
    planner: QueryPlanner = Depends(get_query_planner),
) -> ResearchPlanResponse:
    """Analyze a startup idea and generate a structured ResearchQueryPlan for Agent 1."""
    try:
        idea_brief = analyzer.analyze(payload)
        research_plan = planner.plan(idea_brief)
        _apply_llm_metadata_headers(response, planner.last_execution_metadata)
        return ResearchPlanResponse(
            idea_brief=idea_brief,
            research_plan=research_plan,
        )
    except LLMAllProvidersFailedError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={
                "error": "all_llm_providers_failed",
                "message": "Both primary and fallback AI research planning providers failed.",
                "attempts": exc.attempts,
            },
        ) from exc
    except GeminiConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI research planning service is not configured. Ensure GEMINI_API_KEY is set.",
        ) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The AI research planning request timed out. Please try again.",
        ) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI research planning service is currently unavailable. Please try again later.",
        ) from exc
    except (
        GeminiInvalidResponseError,
        InvalidModelOutputError,
        InvalidQueryPlanOutputError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI research planning service returned an unreadable response.",
        ) from exc
    except (IdeaValidationError, QueryPlanValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI research planning response did not match the expected schema.",
        ) from exc


@app.post("/api/v1/ideas/search", response_model=SearchExecutionReport, tags=["Ideas"])
def execute_research_search_endpoint(
    payload: ResearchQueryPlan,
    executor: SearchExecutor = Depends(get_search_executor),
) -> SearchExecutionReport:
    """Execute a validated ResearchQueryPlan via Tavily and return deduplicated sources."""
    try:
        return executor.execute(payload)
    except TavilyConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search service is not configured. Ensure TAVILY_API_KEY is set.",
        ) from exc
    except TavilyAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Web search service authentication failed.",
        ) from exc
    except TavilyProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search service is currently unavailable. Please try again later.",
        ) from exc


@app.post("/api/v1/ideas/package-evidence", response_model=EvidencePackage, tags=["Ideas"])
def package_evidence_endpoint(
    payload: SearchExecutionReport,
    packager: EvidencePackager = Depends(get_evidence_packager),
) -> EvidencePackage:
    """Deterministically normalize, deduplicate, and group SearchExecutionReport sources."""
    try:
        return packager.package(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "evidence_packaging_failed",
                "message": "An unexpected internal error occurred while packaging evidence.",
            },
        ) from exc


@app.post("/api/v1/agent1/run", response_model=Agent1RunResponse, tags=["Agent 1"])
def run_agent1_endpoint(
    payload: StartupIdeaInput,
    runner: Agent1Runner = Depends(get_agent1_runner),
) -> Agent1RunResponse:
    """Execute the complete Agent 1 pipeline: Idea Analysis -> Query Planning -> Web Search."""
    try:
        return runner.run(payload)
    except LLMAllProvidersFailedError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={
                "error": "all_llm_providers_failed",
                "message": "Both primary and fallback AI providers failed during Agent 1 execution.",
                "attempts": exc.attempts,
            },
        ) from exc
    except GeminiConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llm_not_configured",
                "message": "AI service is not configured. Ensure GEMINI_API_KEY is set.",
            },
        ) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": "llm_timeout",
                "message": "The AI request timed out during Agent 1 execution. Please try again.",
            },
        ) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "llm_unavailable",
                "message": "The AI service is currently unavailable. Please try again later.",
            },
        ) from exc
    except (
        GeminiInvalidResponseError,
        InvalidModelOutputError,
        InvalidQueryPlanOutputError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "invalid_llm_output",
                "message": "The AI service returned an unreadable response during Agent 1 execution.",
            },
        ) from exc
    except (IdeaValidationError, QueryPlanValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "llm_schema_validation_failed",
                "message": "The AI response did not match the required Agent 1 schema.",
            },
        ) from exc
    except TavilyConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "search_not_configured",
                "message": "Web search service is not configured. Ensure TAVILY_API_KEY is set.",
            },
        ) from exc
    except TavilyAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "search_auth_failed",
                "message": "Web search service authentication failed.",
            },
        ) from exc
    except Agent1SearchFailedError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={
                "error": "web_search_failed",
                "message": "All planned web search queries failed during Agent 1 execution.",
                "errors": [err.model_dump(mode="json") for err in exc.errors],
            },
        ) from exc
    except TavilyProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "search_unavailable",
                "message": "Web search service is currently unavailable. Please try again later.",
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected internal error occurred while running Agent 1.",
            },
        ) from exc
