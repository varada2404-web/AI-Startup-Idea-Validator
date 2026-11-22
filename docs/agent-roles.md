# AI-Based Startup Idea Validator
## Agent Roles Documentation

### 1. Idea Analyzer
- Understands the startup idea submitted by the user.
- Extracts important information from the idea.
- Creates a structured idea brief for further research.

### 2. Query Planner
- Uses the analyzed startup idea to plan market research.
- Generates relevant search queries.
- Organizes queries into a structured research plan.

### 3. Search Executor
- Executes research queries using the Tavily search API.
- Collects relevant information from the web.
- Handles search results and search failures.

### 4. Evidence Packager
- Organizes the collected search results.
- Normalizes and removes duplicate sources.
- Prepares structured evidence for further analysis.

### 5. Agent 1 Runner
- Coordinates the complete Agent 1 workflow.
- Connects the Idea Analyzer, Query Planner, Search Executor, and Evidence Packager.
- Manages the execution process.
- Returns the final Agent 1 response.

### 6. LLM Providers
- Gemini acts as the primary LLM provider.
- Groq is available as a fallback provider.
- LLM Router manages provider selection and fallback.

### Conclusion
Each component has a specific responsibility in the startup idea validation process. Together, these components help analyze startup ideas, plan research, collect web evidence, and generate structured results.