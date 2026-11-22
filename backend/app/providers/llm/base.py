"""Typed interface for idea understanding providers."""

from typing import Protocol

from app.schemas.contracts import IdeaBrief, ProviderRunProvenance, ValidationRequest


class IdeaUnderstandingProvider(Protocol):
    @property
    def provenance(self) -> ProviderRunProvenance: ...

    async def extract_idea_brief(self, request: ValidationRequest) -> IdeaBrief: ...
