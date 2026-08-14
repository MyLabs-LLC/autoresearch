"""Turning a judge's quoted evidence into character offsets.

Language models are unreliable at reporting character offsets and reliable at quoting
text verbatim. So the judge is never asked for offsets: it returns
``{label, text}`` evidence, and this module locates each quote in the document
deterministically. Offsets are therefore always correct by construction -- a quote
either matches the document or is discarded, and there is no silent third case where a
span points at the wrong characters.

The output shape mirrors nvidia/Nemotron-PII exactly:

* ``spans``: ``[{"start": int, "end": int, "text": str, "label": str}]``
* ``text_tagged``: the document with each span rewritten as ``[surface]label``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

OccurrenceMode = Literal["all", "first"]

# Wrapping characters a model routinely adds around a quotation.
_WRAP = " \t\n\r\"'()[]{}<>"
# Sentence punctuation that a quote may pick up from the surrounding text.
_TRAILING = ".,;:!?"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text, "label": self.label}

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Evidence:
    """One PII mention as reported by the judge."""

    label: str
    text: str


@dataclass
class Resolution:
    spans: list[Span]
    unresolved: list[dict] = field(default_factory=list)
    """Evidence that could not be located, with a ``reason``. These are kept rather
    than dropped silently: a high unresolved rate means the judge is paraphrasing, and
    that is a quality signal worth surfacing."""
    dropped_overlaps: list[dict] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        seen: dict[str, None] = {}
        for span in self.spans:
            seen.setdefault(span.label, None)
        return list(seen)

    def count_by_label(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for span in self.spans:
            counts[span.label] = counts.get(span.label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _find_exact(text: str, needle: str, mode: OccurrenceMode) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    start = text.find(needle)
    while start != -1:
        hits.append((start, start + len(needle)))
        if mode == "first":
            break
        start = text.find(needle, start + 1)
    return hits


def _candidate_quotes(quote: str) -> list[str]:
    """Quote variants to try, in priority order.

    The trimmed form is tried **before** the raw quote. A quote that ends in sentence
    punctuation almost always picked it up from the surrounding text -- an entity does
    not end in a comma -- and matching the raw form would produce a span whose offsets
    are real but whose boundary is one character too wide, which then fails to line up
    with gold spans. The cost is that a genuine trailing period (``Acme Inc.``) is
    trimmed too; that trade is worth it because boundary agreement is what span-level
    scoring measures.
    """
    candidates: list[str] = []
    unwrapped = quote.strip(_WRAP)
    trimmed = unwrapped.rstrip(_TRAILING).rstrip()
    for candidate in (trimmed, unwrapped, quote):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _find_flexible(text: str, needle: str, mode: OccurrenceMode) -> list[tuple[int, int]]:
    """Match with runs of whitespace treated as interchangeable.

    A quote copied out of a document often normalises a line break or a double space;
    that should not cost us the span.
    """
    tokens = [re.escape(tok) for tok in needle.split()]
    if not tokens:
        return []
    pattern = re.compile(r"\s+".join(tokens))
    hits = [(m.start(), m.end()) for m in pattern.finditer(text)]
    return hits[:1] if mode == "first" and hits else hits


def resolve(
    text: str,
    evidence: Iterable[Evidence],
    *,
    allowed_labels: set[str] | None = None,
    mode: OccurrenceMode = "all",
    min_length: int = 1,
) -> Resolution:
    """Locate each piece of evidence in ``text``.

    ``mode="all"`` labels every occurrence of a quote, matching Nemotron-PII's
    convention that a name repeated five times yields five spans. ``mode="first"``
    labels only the first.
    """
    spans: list[Span] = []
    unresolved: list[dict] = []

    for item in evidence:
        quote = item.text
        if allowed_labels is not None and item.label not in allowed_labels:
            unresolved.append({"label": item.label, "text": quote, "reason": "label_not_in_taxonomy"})
            continue
        if not quote or len(quote.strip()) < min_length:
            unresolved.append({"label": item.label, "text": quote, "reason": "empty_quote"})
            continue

        hits: list[tuple[int, int]] = []
        for candidate in _candidate_quotes(quote):
            hits = _find_exact(text, candidate, mode)
            if hits:
                break
        if not hits:
            # Last resort: tolerate whitespace differences (a quote copied across a
            # line break normalises the newline to a space).
            for candidate in _candidate_quotes(quote):
                hits = _find_flexible(text, candidate, mode)
                if hits:
                    break
        if not hits:
            unresolved.append({"label": item.label, "text": item.text, "reason": "not_found_in_text"})
            continue

        for start, end in hits:
            spans.append(Span(start=start, end=end, text=text[start:end], label=item.label))

    spans, dropped = _dedupe_and_resolve_overlaps(spans)
    return Resolution(spans=spans, unresolved=unresolved, dropped_overlaps=dropped)


def dedupe_overlaps(spans: list[Span]) -> tuple[list[Span], list[dict]]:
    """Public wrapper: drop duplicates and resolve overlaps, longest span winning."""
    return _dedupe_and_resolve_overlaps(spans)


def _dedupe_and_resolve_overlaps(spans: list[Span]) -> tuple[list[Span], list[dict]]:
    """Drop exact duplicates, then resolve overlaps by keeping the longest span.

    Overlaps are common and benign: a judge that reports both ``Jane Doe`` and ``Doe``
    should yield one span, not two contradictory ones. Longest-wins is chosen because
    the wider span carries the more complete surface form.
    """
    unique = {(s.start, s.end, s.label): s for s in spans}
    ordered = sorted(unique.values(), key=lambda s: (-(s.end - s.start), s.start, s.label))

    kept: list[Span] = []
    dropped: list[dict] = []
    for span in ordered:
        clash = next((k for k in kept if k.overlaps(span)), None)
        if clash is None:
            kept.append(span)
        else:
            dropped.append({"dropped": span.to_dict(), "kept": clash.to_dict()})

    kept.sort(key=lambda s: (s.start, s.end))
    return kept, dropped


def tag_text(text: str, spans: list[Span]) -> str:
    """Render ``text_tagged`` in Nemotron-PII's inline format: ``[surface]label``.

    Spans must not overlap; :func:`resolve` guarantees that.
    """
    out: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < cursor:
            raise ValueError("overlapping spans cannot be rendered as inline tags")
        out.append(text[cursor : span.start])
        out.append(f"[{text[span.start:span.end]}]{span.label}")
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


def spans_from_dicts(raw: Iterable[dict]) -> list[Span]:
    """Rebuild spans from a manifest or from a Nemotron-PII row."""
    return [
        Span(start=int(d["start"]), end=int(d["end"]), text=d.get("text", ""), label=d["label"])
        for d in raw
    ]


def realign(old_text: str, new_text: str, spans: list[Span]) -> tuple[list[Span], list[Span]]:
    """Re-point spans from ``old_text`` at ``new_text``.

    Used when a rendering round trip is not byte-exact. Returns ``(realigned, lost)``.
    A span is realigned only if its exact surface form occurs in the new text; ties are
    broken by proximity to the original offset, so a repeated name keeps its position.
    """
    realigned: list[Span] = []
    lost: list[Span] = []
    for span in spans:
        surface = span.text or old_text[span.start : span.end]
        hits = _find_exact(new_text, surface, "all")
        if not hits:
            lost.append(span)
            continue
        start, end = min(hits, key=lambda hit: abs(hit[0] - span.start))
        realigned.append(Span(start=start, end=end, text=new_text[start:end], label=span.label))
    deduped, _ = _dedupe_and_resolve_overlaps(realigned)
    return deduped, lost
