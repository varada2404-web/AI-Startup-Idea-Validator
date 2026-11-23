"""Deterministic tests for the offline M1 research path."""

import asyncio
from datetime import UTC, datetime

from app.pipeline.runner import PipelineRunner
from app.pipeline.store import RunStore
from app.providers.search.mock import MockSearchProvider
from app.schemas.contracts import ProgressItem, ValidationRequest, ValidationRun
from app.services.deduplication import canonicalize_url, check_source_references, near_duplicate
from app.services.idea_understanding import build_idea_brief
from app.services.query_planner import CATEGORY_ORDER, plan_initial_queries
from app.services.quote_location import locate_quote


def make_run(request: ValidationRequest) -> ValidationRun:
    now = datetime.now(UTC)
    return ValidationRun(
        run_id="test-run",
        created_at=now,
        updated_at=now,
        request=request,
    )


def test_idea_brief_keeps_user_wording_and_five_unknown_hypotheses() -> None:
    request = ValidationRequest(
        idea="An appointment reminder tool for independent clinics.",
        target_customer="Independent clinics",
    )

    brief = build_idea_brief(request)

    assert brief.original_text == request.idea
    assert brief.target_customer.origin == "user"
    assert brief.geography.value == "India"
    assert brief.geography.origin == "default"
    assert len(brief.hypotheses) == 5
    assert [item.status for item in brief.hypotheses] == ["unknown"] * 5
    assert [item.critical for item in brief.hypotheses] == [True, True, False, False, False]


def test_validation_request_removes_control_characters() -> None:
    request = ValidationRequest(idea="A useful tool for small local businesses.\x00")
    assert "\x00" not in request.idea


def test_query_plan_covers_all_seven_categories() -> None:
    brief = build_idea_brief(ValidationRequest(idea="A useful tool for small local businesses."))
    queries = plan_initial_queries(brief)

    assert len(queries) == 7
    assert tuple(item.category for item in queries) == CATEGORY_ORDER
    assert {item.round for item in queries} == {1}


def test_demo_extracts_a_caveated_problem_and_inferred_customer() -> None:
    brief = build_idea_brief(
        ValidationRequest(idea="A tool that helps local clinics manage appointment reminders.")
    )

    assert brief.target_customer.value == "local clinics"
    assert brief.target_customer.origin == "inferred"
    assert brief.problem == "Potential users may need a better way to manage appointment reminders."
    assert any("unverified demo inference" in note for note in brief.missing_info)


def test_url_canonicalization_and_near_duplicate_detection() -> None:
    assert canonicalize_url("https://Example.com/article/?utm_source=demo#section") == (
        "https://example.com/article"
    )
    body = "This is a sufficiently long synthetic record with several words to compare."
    assert near_duplicate(body, body)
    assert not near_duplicate(
        body, "Another unrelated group of words has almost nothing in common."
    )


def test_quote_location_finds_text_without_verifying_claim_support() -> None:
    source = "A founder said:  Users need simpler appointment scheduling. End of note."
    location = locate_quote("users need simpler appointment scheduling.", source)

    assert location is not None
    assert source[location.start : location.end] == location.excerpt
    assert "Users need simpler appointment scheduling." == location.excerpt


def test_pipeline_runs_bounded_search_dedupes_and_labels_simulation(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.sqlite3")
    request = ValidationRequest(idea="A tool that helps local clinics manage appointments.")
    run = make_run(request)
    store.save(run)
    runner = PipelineRunner(store, MockSearchProvider())

    asyncio.run(runner.execute(run.run_id))
    result = store.get(run.run_id)

    assert result is not None
    assert result.status == "completed"
    assert result.demo_mode is True
    assert result.manifest.idea_understanding is not None
    assert result.manifest.idea_understanding.provider == "mock"
    assert result.raw_result_count == 11  # Seven initial, at most four follow-up results.
    assert result.deduplicated_count == 1
    assert len(result.sources) == 10
    assert {source.category for source in result.queries if source.round == 1} == set(
        CATEGORY_ORDER
    )
    assert all(source.simulated is True for source in result.sources)
    assert any(len(source.categories) == 2 for source in result.sources)
    assert sum(item.status == "covered" for item in result.coverage) == 4
    assert check_source_references(result.sources) == []
    assert check_source_references(result.sources, ["missing-source"])
    assert [item.stage for item in result.progress][-1] == "storage"
    assert all(item.status == "completed" for item in result.progress)


def test_orphaned_queued_run_is_marked_interrupted_not_resumed(tmp_path) -> None:
    db_path = tmp_path / "runs.sqlite3"
    store = RunStore(db_path)
    run = make_run(ValidationRequest(idea="A useful tool for small local businesses."))
    run.progress = [ProgressItem(stage="search", label="Search", status="running")]
    store.save(run)

    reopened = RunStore(db_path)
    assert reopened.mark_orphaned_runs_interrupted() == 1
    result = reopened.get(run.run_id)

    assert result is not None
    assert result.status == "interrupted"
    assert "Start a new validation run" in (result.error or "")
    assert result.progress[0].status == "interrupted"
