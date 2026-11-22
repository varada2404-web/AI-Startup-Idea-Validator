"""Deterministic interpretation used only for the offline demo."""

import re

from app.schemas.contracts import Hypothesis, IdeaBrief, Provenance, ValidationRequest

HYPOTHESIS_TEMPLATES = (
    (
        "demand",
        "Potential target users experience the stated problem often enough to seek a solution.",
        True,
    ),
    (
        "willingness_to_pay",
        "A reachable buyer is willing to pay for a solution to the stated problem.",
        True,
    ),
    (
        "competition",
        "Existing alternatives leave an unmet need for the intended target users.",
        False,
    ),
    (
        "feasibility",
        "A useful first version can be built and operated within the founder's constraints.",
        False,
    ),
    (
        "timing",
        "Current user and market conditions make adoption plausible now.",
        False,
    ),
)


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text.casefold())
    stop = {
        "about",
        "after",
        "and",
        "app",
        "for",
        "from",
        "help",
        "helps",
        "manage",
        "platform",
        "service",
        "system",
        "that",
        "the",
        "their",
        "tool",
        "with",
    }
    selected: list[str] = []
    for word in words:
        if len(word) < 3 or word in stop or word in selected:
            continue
        selected.append(word)
        if len(selected) == limit:
            break
    return selected or ["startup idea"]


def build_idea_brief(request: ValidationRequest) -> IdeaBrief:
    idea = request.idea.strip()
    inferred = re.search(
        r"\b(?:helps?|enables?|allows?)\s+(.+?)\s+(?:to\s+)?"
        r"(manage|reduce|track|find|book|share|sell|buy|schedule|monitor|organize|organise|access|compare)\s+(.+)",
        idea,
        flags=re.IGNORECASE,
    )
    inferred_target = inferred.group(1).strip(" .,!?") if inferred else None
    inferred_action = inferred.group(2).casefold() if inferred else None
    inferred_object = inferred.group(3).strip(" .,!?") if inferred else None

    target = request.target_customer or inferred_target
    geography = request.geography or "India"
    missing = []
    if not request.target_customer and inferred_target:
        missing.append(
            "Target customer is a deterministic demo inference; confirm it with the founder."
        )
    elif target is None:
        missing.append("Target customer was not specified by the user.")
    if request.geography is None:
        missing.append("Geography defaults to India; confirm or edit it for this idea.")
    if inferred_action and inferred_object:
        problem = f"Potential users may need a better way to {inferred_action} {inferred_object}."
        missing.append(
            "Problem interpretation is an unverified demo inference; confirm it with users."
        )
    else:
        problem = "Not separately identified from the submitted wording."
        missing.append("Clarify who experiences the problem and what they need to do.")

    hypotheses = [
        Hypothesis(id=f"H{index}", kind=kind, statement=statement, critical=critical)
        for index, (kind, statement, critical) in enumerate(HYPOTHESIS_TEMPLATES, start=1)
    ]

    return IdeaBrief(
        original_text=idea,
        problem=problem,
        solution=f"Founder-submitted solution description: {idea}",
        target_customer=Provenance(
            value=target,
            origin="user" if request.target_customer else "inferred" if target else "unknown",
        ),
        geography=Provenance(value=geography, origin="user" if request.geography else "default"),
        industry="Not classified in demo mode",
        hypotheses=hypotheses,
        missing_info=missing,
        search_keywords=extract_keywords(idea),
    )
