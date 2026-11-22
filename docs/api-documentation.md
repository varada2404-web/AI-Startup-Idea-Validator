# API Documentation

## AI-Based Startup Idea Validator

### 1. Overview

This document describes the REST API endpoints, request models, response models, and error handling for the AI-Based Startup Idea Validator backend.

The API is built using FastAPI and uses AI providers for startup idea analysis and Tavily for web research.

### 2. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/v1/ideas` | Submit a startup idea |
| POST | `/api/v1/ideas/analyze` | Analyze a startup idea |
| POST | `/api/v1/ideas/plan-research` | Generate a research plan |
| POST | `/api/v1/ideas/search` | Execute web searches |
| POST | `/api/v1/ideas/package-evidence` | Package and organize evidence |
| POST | `/api/v1/agent1/run` | Execute the complete Agent 1 pipeline |

### 3. Request Models

#### 3.1 StartupIdeaInput

Used by the idea submission, idea analysis, and complete Agent 1 endpoints.

| Field | Type | Validation |
|---|---|---|
| `idea` | string | Required, minimum 20 characters |
| `target_geography` | string | Required, minimum 2 characters |

#### 3.2 IdeaBrief

Structured representation of a startup idea.

| Field | Type |
|---|---|
| `problem` | string |
| `solution` | string |
| `target_customers` | list of strings |
| `industry` | string |
| `business_model` | optional string |
| `target_geography` | string |
| `missing_information` | list of strings |
| `assumptions` | list of strings |

#### 3.3 ResearchQueryPlan

Used to organize research queries across five categories.

| Field | Type |
|---|---|
| `target_geography` | string |
| `industry` | string |
| `queries` | list of ResearchQuery |
| `research_gaps` | list of strings |
| `assumptions_to_validate` | list of strings |

Each ResearchQuery contains:

- `category`
- `query`
- `rationale`
- `priority`

Research categories:

- `market_research`
- `competitor_research`
- `customer_pain_points`
- `pricing_and_business_models`
- `industry_trends`

Each category requires 2–3 unique queries.

Query priorities are `high`, `medium`, and `low`.

#### 3.4 SearchExecutionReport

Response model for web search execution.

| Field | Type |
|---|---|
| `total_queries` | integer |
| `successful_queries` | integer |
| `failed_queries` | integer |
| `total_results_retrieved` | integer |
| `unique_sources_count` | integer |
| `sources` | list of SourceRecord |
| `errors` | list of SearchQueryError |

Each SourceRecord includes title, URL, content, query, category, matched queries, matched categories, relevance score, publication date, and domain.

Each SearchQueryError includes query, category, error type, and a sanitized error message.

#### 3.5 EvidencePackage

Response model for organized research evidence.

| Field | Type |
|---|---|
| `total_sources_received` | integer |
| `unique_sources` | integer |
| `duplicate_sources` | integer |
| `total_categories` | integer |
| `categories` | list of EvidenceCategoryGroup |
| `metadata_quality_summary` | MetadataQualitySummary |
| `uncovered_categories` | list of ResearchCategory |
| `packaging_errors` | list of PackagingError |

Each PackagedEvidence record includes:

- `evidence_id`
- `title`
- `url`
- `domain`
- `content_snippet`
- `research_category`
- `originating_queries`
- `matched_categories`
- `relevance_score`
- `published_date`
- `metadata_completeness`
- `missing_metadata`

Metadata completeness measures the presence of tracked metadata. It does not measure factual accuracy or source credibility.

### 4. Complete Agent 1 API

**Endpoint:** `POST /api/v1/agent1/run`

**Request model:** `StartupIdeaInput`

**Response model:** `Agent1RunResponse`

The complete pipeline response contains five sections:

| Field | Description |
|---|---|
| `idea_brief` | Structured startup idea analysis |
| `research_plan` | Multi-category research query plan |
| `search_report` | Web search results and query errors |
| `evidence_package` | Deduplicated and categorized evidence |
| `execution_metadata` | Execution statistics, provider information, and status |

Execution metadata includes provider and model information, fallback usage, query counts, source counts, metadata completeness, uncovered categories, overall status, and execution errors.

Possible overall statuses:

- `success`
- `partial_success`
- `no_sources`

### 5. API Error Handling

The backend returns structured HTTP errors when an operation fails.

| Error Code | HTTP Status | Description |
|---|---:|---|
| `llm_not_configured` | 503 | AI provider is not configured |
| `llm_timeout` | 504 | AI request timed out |
| `llm_unavailable` | 503 | AI provider unavailable |
| `invalid_llm_output` | 502 | AI returned an unreadable response |
| `llm_schema_validation_failed` | 502 | AI output did not match the required schema |
| `all_llm_providers_failed` | Provider-dependent | Primary and fallback AI providers failed |
| `search_not_configured` | 503 | Tavily API is not configured |
| `search_auth_failed` | 502 | Tavily authentication failed |
| `web_search_failed` | Provider-dependent | All planned search queries failed |
| `search_unavailable` | 503 | Web search provider unavailable |
| `internal_server_error` | 500 | Unexpected internal error |

### 6. External Service Configuration

The backend integrates with:

- Gemini — primary AI provider.
- Groq — fallback AI provider.
- Tavily — web search provider.

Required environment variable names:

- `GEMINI_API_KEY`
- `TAVILY_API_KEY`

API keys must be stored securely in environment configuration and must never be exposed in API responses or source control.

### 7. Data Validation

The API uses Pydantic models for request and response validation.

Important validation rules include:

- Startup idea must contain at least 20 characters.
- Target geography must contain at least 2 characters.
- Research queries must cover all five research categories.
- Each category must contain 2–3 unique queries.
- Successful and failed search counts must equal total queries.
- Unique source counts must match the returned source records.
- Evidence package source accounting must remain consistent.

### 8. Complete Agent 1 Workflow

1. Receive startup idea and target geography.
2. Analyze the idea and generate an IdeaBrief.
3. Generate a structured research query plan.
4. Execute planned web searches using Tavily.
5. Normalize and deduplicate search evidence.
6. Group evidence by research category.
7. Return the complete Agent 1 response with execution metadata.

### 9. Notes

- Retrieved web content is untrusted external information.
- Relevance scores are not credibility ratings.
- Metadata completeness measures metadata presence, not factual accuracy.
- API errors should not expose secrets or stack traces.