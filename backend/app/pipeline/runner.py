"""Bounded asynchronous Agent 1 pipeline for demo and live providers."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import cast

from app.pipeline.store import RunStore
from app.providers.llm.base import IdeaUnderstandingProvider
from app.providers.llm.errors import ProviderFailure
from app.providers.llm.mock import MockIdeaUnderstandingProvider
from app.providers.search.base import (
    ExtractionResult,
    SearchProvider,
    SearchResult,
    SourceExtractionProvider,
)
from app.schemas.contracts import ProgressItem, ProviderError, QueryPlanItem, ValidationRun
from app.services.deduplication import canonicalize_url, check_source_references
from app.services.query_planner import plan_follow_up_queries, plan_initial_queries
from app.services.source_processing import (
    calculate_coverage,
    categories_needing_follow_up,
    deduplicate_results,
    select_sources_for_extraction,
)

logger = logging.getLogger(__name__)

PIPELINE_STAGES = (
    ("extraction", "Understand idea"),
    ("query_planning", "Plan research queries"),
    ("search_round_1", "Research search — round 1"),
    ("deduplication", "Categorize and deduplicate sources"),
    ("follow_up_planning", "Plan bounded follow-up"),
    ("search_round_2", "Research search — round 2"),
    ("source_extraction", "Extract selected source pages"),
    ("storage", "Save run results"),
)


class PipelineRunner:
    def __init__(
        self,
        store: RunStore,
        provider: SearchProvider,
        interpreter: IdeaUnderstandingProvider | None = None,
        max_concurrent_calls: int = 4,
        max_follow_up_queries: int = 4,
        max_extract_sources: int = 8,
        max_source_text_chars: int = 12_000,
        provider_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.interpreter = interpreter or MockIdeaUnderstandingProvider()
        self.max_concurrent_calls = max(1, min(4, max_concurrent_calls))
        self.max_follow_up_queries = max(0, min(4, max_follow_up_queries))
        self.max_extract_sources = max(0, min(8, max_extract_sources))
        self.max_source_text_chars = max(500, min(20_000, max_source_text_chars))
        self.provider_semaphore = provider_semaphore or asyncio.Semaphore(
            self.max_concurrent_calls
        )

    async def execute(self, run_id: str) -> None:
        run = self.store.get(run_id)
        if run is None:
            return
        run.status = "running"
        is_live = self.interpreter.provenance.provider == "gemini"
        run.mode = "live" if is_live else "demo"
        run.demo_mode = not is_live
        run.simulation_notice = None if is_live else run.simulation_notice
        self._save(run)
        try:
            await self._stage(
                run,
                "extraction",
                "Generating the structured idea brief with the configured Agent 1 provider.",
            )
            run.manifest.idea_understanding = self.interpreter.provenance
            self._save(run)
            async with self.provider_semaphore:
                run.idea_brief = await self.interpreter.extract_idea_brief(run.request)

            await self._stage(
                run, "query_planning", "Planning one query for each of seven categories."
            )
            initial = plan_initial_queries(run.idea_brief)
            run.queries.extend(initial)

            await self._stage(
                run,
                "search_round_1",
                f"Executing {len(initial)} bounded research searches.",
            )
            results, errors, successful_searches = await self._search(initial)
            run.provider_errors.extend(errors)
            run.raw_result_count += len(results)

            await self._stage(
                run,
                "deduplication",
                "Canonical URL deduplication and category coverage calculation.",
            )
            interim_sources, _ = deduplicate_results(results)
            coverage = calculate_coverage(interim_sources)

            await self._stage(
                run,
                "follow_up_planning",
                "Planning at most four follow-up queries for partial or missing categories.",
            )
            follow_ups = plan_follow_up_queries(
                run.idea_brief,
                categories_needing_follow_up(coverage),
                limit=self.max_follow_up_queries,
            )
            run.queries.extend(follow_ups)

            await self._stage(
                run,
                "search_round_2",
                f"Executing {len(follow_ups)} bounded follow-up searches.",
            )
            follow_results, errors, successful_follow_ups = await self._search(follow_ups)
            run.provider_errors.extend(errors)
            run.raw_result_count += len(follow_results)
            all_results = results + follow_results

            if is_live and not (successful_searches + successful_follow_ups):
                raise ProviderFailure(
                    ProviderError(
                        provider="tavily",
                        stage="search",
                        code="all_searches_failed",
                        message="All live searches failed; no sources were retrieved.",
                    )
                )

            run.sources, duplicates = deduplicate_results(all_results)
            run.deduplicated_count = duplicates
            if is_live:
                await self._extract_selected(run)

            await self._stage(run, "storage", "Persisting the run results to local SQLite.")
            reference_problems = check_source_references(run.sources)
            if reference_problems:
                raise RuntimeError("Source reference integrity failed.")
            run.coverage = calculate_coverage(run.sources)
            if run.provider_errors:
                run.quality = "degraded"
            run.status = "completed"
            run.updated_at = datetime.now(UTC)
            for stage in run.progress:
                if stage.status == "running":
                    stage.status = "completed"
            self._save(run)
        except ProviderFailure as exc:
            logger.warning(
                "Provider failure run_id=%s provider=%s stage=%s code=%s",
                run_id,
                exc.error.provider,
                exc.error.stage,
                exc.error.code,
            )
            run.status = "failed"
            run.provider_errors.append(exc.error)
            run.error = exc.error.message
            run.updated_at = datetime.now(UTC)
            for stage in run.progress:
                if stage.status == "running":
                    stage.status = "failed"
            self._save(run)
        except Exception:
            logger.exception("Agent 1 run %s failed unexpectedly", run_id)
            run.status = "failed"
            run.error = (
                "The Agent 1 pipeline encountered an unexpected error. Check the server log."
            )
            run.updated_at = datetime.now(UTC)
            for stage in run.progress:
                if stage.status == "running":
                    stage.status = "failed"
            self._save(run)

    async def _search(
        self, queries: list[QueryPlanItem]
    ) -> tuple[list[SearchResult], list[ProviderError], int]:
        async def bounded_search(query: QueryPlanItem) -> list[SearchResult]:
            async with self.provider_semaphore:
                return await self.provider.search(query.query, query.category, query.round)

        responses = await asyncio.gather(
            *(bounded_search(query) for query in queries), return_exceptions=True
        )
        results: list[SearchResult] = []
        errors: list[ProviderError] = []
        successes = 0
        for response in responses:
            if isinstance(response, ProviderFailure):
                errors.append(response.error)
            elif isinstance(response, BaseException):
                errors.append(
                    ProviderError(
                        provider="tavily",
                        stage="search",
                        code="provider_error",
                        message="A live search request failed.",
                        retryable=True,
                    )
                )
            else:
                results.extend(response)
                successes += 1
        return results, errors, successes

    async def _extract_selected(self, run: ValidationRun) -> None:
        extract = getattr(self.provider, "extract", None)
        if not callable(extract):
            return
        selected = select_sources_for_extraction(run.sources, self.max_extract_sources)
        if not selected:
            return
        await self._stage(
            run,
            "source_extraction",
            f"Extracting up to {len(selected)} selected unique source pages.",
        )
        try:
            async with self.provider_semaphore:
                extractions = cast(SourceExtractionProvider, self.provider).extract(
                    [source.url for source in selected]
                )
                results: list[ExtractionResult] = await extractions
        except ProviderFailure as exc:
            run.provider_errors.append(exc.error)
            run.quality = "degraded"
            for source in selected:
                source.extraction_status = "failed"
                source.extraction_error = exc.error
            return
        except Exception:
            error = ProviderError(
                provider="tavily",
                stage="extraction",
                code="provider_error",
                message=(
                    "Tavily could not extract the selected pages; "
                    "search snippets are retained."
                ),
                retryable=True,
            )
            run.provider_errors.append(error)
            run.quality = "degraded"
            for source in selected:
                source.extraction_status = "failed"
                source.extraction_error = error
            return

        by_url = {canonicalize_url(item.url): item for item in results}
        for source in selected:
            result = by_url.get(canonicalize_url(source.url))
            if result is not None and not result.failed and result.content:
                source.extracted_text = result.content[: self.max_source_text_chars]
                source.extraction_status = "succeeded"
            else:
                error = ProviderError(
                    provider="tavily",
                    stage="extraction",
                    code="extraction_failed",
                    message="This page could not be extracted; its search snippet is retained.",
                    retryable=True,
                )
                source.extraction_status = "failed"
                source.extraction_error = error
                run.provider_errors.append(error)
                run.quality = "degraded"

    async def _stage(self, run: ValidationRun, name: str, detail: str) -> None:
        now = datetime.now(UTC)
        for stage in run.progress:
            if stage.status == "running":
                stage.status = "completed"
        item = next((stage for stage in run.progress if stage.stage == name), None)
        if item is None:
            label = next(label for key, label in PIPELINE_STAGES if key == name)
            item = ProgressItem(stage=name, label=label)
            run.progress.append(item)
        item.status = "running"
        item.detail = detail
        run.updated_at = now
        self._save(run)
        await asyncio.sleep(0.12)

    def _save(self, run: ValidationRun) -> None:
        for source in run.sources:
            source.text = source.text[: self.max_source_text_chars]
            if source.snippet is not None:
                source.snippet = source.snippet[: self.max_source_text_chars]
            if source.extracted_text is not None:
                source.extracted_text = source.extracted_text[: self.max_source_text_chars]
        self.store.save(run)
