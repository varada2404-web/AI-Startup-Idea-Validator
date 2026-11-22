"""URL and text deduplication for retrieved demo records."""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.schemas.contracts import Source

TRACKING_KEYS = {"fbclid", "gclid", "ref", "source"}


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.casefold()
    host = (parts.hostname or "").casefold()
    port = f":{parts.port}" if parts.port else ""
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_") and key.casefold() not in TRACKING_KEYS
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, f"{host}{port}", path, query, ""))


def word_five_grams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[index : index + 5]) for index in range(max(0, len(words) - 4))}


def near_duplicate(left: str, right: str, threshold: float = 0.6) -> bool:
    left_grams = word_five_grams(left)
    right_grams = word_five_grams(right)
    if not left_grams or not right_grams:
        return False
    overlap = len(left_grams & right_grams) / len(left_grams | right_grams)
    return overlap >= threshold


def check_source_references(
    sources: list[Source], referenced_source_ids: list[str] | None = None
) -> list[str]:
    problems: list[str] = []
    ids = [source.id for source in sources]
    if len(ids) != len(set(ids)):
        problems.append("duplicate source id")
    known = set(ids)
    for source_id in referenced_source_ids or []:
        if source_id not in known:
            problems.append(f"reference to missing source: {source_id}")
    return problems
