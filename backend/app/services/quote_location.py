"""Quote-location utility foundation; this does not verify claim support."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuoteLocation:
    start: int
    end: int
    excerpt: str


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    previous_space = False
    for offset, original in enumerate(text):
        folded = original.casefold()
        if original.isspace():
            if chars and not previous_space:
                chars.append(" ")
                offsets.append(offset)
            previous_space = True
            continue
        for char in folded:
            chars.append(char)
            offsets.append(offset)
        previous_space = False
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def locate_quote(excerpt: str, source_text: str) -> QuoteLocation | None:
    normalized_excerpt, _ = _normalized_with_offsets(excerpt.strip())
    normalized_source, offsets = _normalized_with_offsets(source_text)
    if not normalized_excerpt:
        return None
    start = normalized_source.find(normalized_excerpt)
    if start < 0:
        return None
    end = start + len(normalized_excerpt)
    original_start = offsets[start]
    original_end = offsets[end - 1] + 1
    return QuoteLocation(
        start=original_start,
        end=original_end,
        excerpt=source_text[original_start:original_end],
    )
