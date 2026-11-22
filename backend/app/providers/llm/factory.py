"""Explicit provider selection; LIVE never falls back to the deterministic mock."""

from app.core.config import ApplicationConfig
from app.providers.llm.base import IdeaUnderstandingProvider
from app.providers.llm.gemini import GeminiIdeaUnderstandingProvider
from app.providers.llm.mock import MockIdeaUnderstandingProvider


def create_idea_understanding_provider(
    config: ApplicationConfig,
) -> IdeaUnderstandingProvider:
    if config.mode == "demo":
        return MockIdeaUnderstandingProvider()
    return GeminiIdeaUnderstandingProvider(config)
