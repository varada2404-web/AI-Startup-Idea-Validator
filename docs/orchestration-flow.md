# AI-Based Startup Idea Validator

## Agent 1 Orchestration Flow Documentation

### 1. Overview

Agent 1 is responsible for analyzing a startup idea and collecting relevant market research evidence from the web. It coordinates multiple components to complete the research workflow.

### 2. Workflow Steps

#### Step 1: User Input
- The user submits a startup idea through the application.
- The idea is received by the backend API.
- The input is validated before processing.

#### Step 2: Idea Analyzer
- Understands the submitted startup idea.
- Extracts important details.
- Creates a structured idea brief.

#### Step 3: Query Planner
- Receives the structured idea brief.
- Generates relevant search queries.
- Prepares a research plan for market analysis.

#### Step 4: Search Executor
- Executes the planned queries using the Tavily Search API.
- Collects relevant web information.
- Handles search results and possible errors.

#### Step 5: Evidence Packager
- Organizes the collected search results.
- Removes duplicate sources.
- Prepares structured evidence for further analysis.

#### Step 6: Agent 1 Runner
- Coordinates all the workflow components.
- Manages the execution sequence.
- Collects the final output from the workflow.

#### Step 7: Response
- The backend returns the structured Agent 1 response.
- The frontend displays the result to the user.

### 3. LLM Provider Flow

- Gemini is used as the primary LLM provider.
- Groq is available as a fallback provider.
- The LLM Router manages provider selection and fallback.

### 4. Orchestration Sequence

1. User submits a startup idea.
2. Backend validates the input.
3. Idea Analyzer creates an idea brief.
4. Query Planner generates search queries.
5. Search Executor collects web information using Tavily.
6. Evidence Packager organizes the research evidence.
7. Agent 1 Runner coordinates the workflow.
8. Backend returns the final structured response.

### 5. Conclusion

The Agent 1 orchestration flow connects multiple components into a structured research pipeline. Each component performs a specific task, helping the system analyze startup ideas and collect relevant market research evidence.