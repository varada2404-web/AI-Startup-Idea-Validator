"""Deterministic local substitute for an LLM idea-understanding provider."""

from app.schemas.contracts import IdeaBrief, ProviderRunProvenance, ValidationRequest
from app.services.idea_understanding import build_idea_brief


class MockIdeaUnderstandingProvider:
    @property
    def provenance(self) -> ProviderRunProvenance:
        return ProviderRunProvenance(
            provider="mock",
            model_identifier="deterministic-heuristic",
            prompt_version="mock-v1",
        )

    async def extract_idea_brief(self, request: ValidationRequest) -> IdeaBrief:
        return build_idea_brief(request)
