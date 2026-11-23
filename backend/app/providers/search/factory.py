"""Explicit search provider selection. LIVE mode never falls back to mock data."""

from app.core.config import ApplicationConfig
from app.providers.search.base import SearchProvider
from app.providers.search.mock import MockSearchProvider
from app.providers.search.tavily import TavilyProvider


def create_search_provider(config: ApplicationConfig) -> SearchProvider:
    if config.mode == "demo":
        return MockSearchProvider()
    return TavilyProvider(config)
