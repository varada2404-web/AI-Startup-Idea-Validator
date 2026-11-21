# Tavily Web Search Agent
### Module: AI-Based Startup Idea Validator with Market Analysis Assistance
**Author:** Varada | **Role:** Tavily web search agent

---

## What This Does

Given a startup idea (e.g., *"AI resume screener for HR teams"*), this agent runs **5 targeted Tavily searches** and returns structured market intelligence:

| Category | Query Pattern |
|---|---|
| Market Size | `{idea} market size and growth` |
| Competitors | `{idea} competitors and existing startups` |
| Target Audience | `{idea} target audience and customer segment` |
| Trends | `{idea} recent news and trends 2024` |
| Risks | `{idea} challenges and risks` |

---

## Project Structure

```
WEB-TAVALIY/
├── agent/
│   ├── __init__.py
│   └── tavily_search_agent.py   ← core agent logic
├── api/
│   ├── __init__.py
│   └── search_router.py         ← FastAPI endpoints
├── main.py                      ← app entry point
├── test_agent.py                ← quick local test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API keys
copy .env.example .env
# Then edit .env and add your keys:
#   TAVILY_API_KEY=tvly-...
#   OPENAI_API_KEY=sk-...  (needed only if LLM layer is added)
```

Get a free Tavily API key at: https://app.tavily.com

---

## Run

```bash
# Start the API server (port 8001 to avoid conflict with other team services)
uvicorn main:app --reload --port 8001

# Quick local test (no server needed)
python test_agent.py
```

API docs available at: `http://localhost:8001/docs`

---

## API Endpoints

### `POST /api/search`
Run full market research for a startup idea.

**Request:**
```json
{
  "idea": "AI-powered resume screener for HR teams",
  "max_results": 5
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "idea": "AI-powered resume screener for HR teams",
    "queries": ["...", "..."],
    "results": { "query": [{ "title": "", "url": "", "content": "", "score": 0.9 }] },
    "summary": { "Market Size And Growth": "...", "Competitors ...": "..." }
  }
}
```

### `POST /api/search/single`
Run a single custom query (used by Prajna's LLM extraction endpoint).

```json
{ "query": "ChatGPT for recruitment market share 2024", "max_results": 3 }
```

### `GET /api/health`
Health check — returns `{ "status": "ok" }`.

---

## Integration with Other Modules

| Team Member | Module | How It Connects |
|---|---|---|
| Prajna | LLM extraction endpoint | Calls `POST /api/search` to get raw web data, then extracts structured insights with LLM |
| Harshita | Architecture doc | Documents this service in the system design |
| Deepak | Repo setup / PR review | Reviews this module on GitHub |

---

## Notes

- `search_depth` is set to `"advanced"` by default (better results, slightly slower).  
  Change to `"basic"` in `TavilySearchAgent(search_depth="basic")` for faster/cheaper runs.
- Results are paginated — `max_results` caps at 10 per Tavily's limit.
- The agent does **not** call an LLM — it only retrieves raw web data. Deeper summarization is handled by Prajna's endpoint.
