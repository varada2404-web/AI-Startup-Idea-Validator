"""Typed interface shared by local and future search providers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.schemas.contracts import ResearchCategory


@dataclass(frozen=True)
class SearchResult:
    url: str
    domain: str
    title: str
    query: str
    category: ResearchCategory
    text: str
    retrieved_at: datetime
    simulated: bool = True
    snippet: str | None = None
    provider_score: float | None = None
    published_at: datetime | None = None
    provider_metadata: dict[str, str | int | float | bool | None] | None = None


@dataclass(frozen=True)
class ExtractionResult:
    url: str
    content: str | None
    failed: bool = False


class SearchProvider(Protocol):
    async def search(
        self, query: str, category: ResearchCategory, round_number: int
    ) -> list[SearchResult]: ...


class SourceExtractionProvider(Protocol):
    async def extract(self, urls: list[str]) -> list[ExtractionResult]: ...
