"""Span resolution: the layer that converts a judge's quotes into offsets.

The central guarantee under test is that a span is either correct or absent -- there is
no path that produces an offset pointing at the wrong characters.
"""

from __future__ import annotations

import pytest

from datax.spans import Evidence, Span, realign, resolve, tag_text

TEXT = "Patient: Jane Doe\nMRN: 88213-A\nSSN: 123-45-6789\nContact jane@example.com."


def test_resolves_exact_quotes():
    resolution = resolve(
        TEXT,
        [
            Evidence("first_name", "Jane"),
            Evidence("last_name", "Doe"),
            Evidence("ssn", "123-45-6789"),
        ],
    )
    assert not resolution.unresolved
    for span in resolution.spans:
        assert TEXT[span.start : span.end] == span.text


def test_hallucinated_quote_is_recorded_not_invented():
    resolution = resolve(TEXT, [Evidence("first_name", "Bartholomew")])
    assert resolution.spans == []
    assert resolution.unresolved[0]["reason"] == "not_found_in_text"


def test_label_outside_taxonomy_is_rejected():
    resolution = resolve(
        TEXT, [Evidence("social_security_number", "123-45-6789")], allowed_labels={"ssn"}
    )
    assert resolution.spans == []
    assert resolution.unresolved[0]["reason"] == "label_not_in_taxonomy"


def test_all_occurrences_by_default():
    text = "Jane met Jane."
    resolution = resolve(text, [Evidence("first_name", "Jane")])
    assert len(resolution.spans) == 2
    resolution_first = resolve(text, [Evidence("first_name", "Jane")], mode="first")
    assert len(resolution_first.spans) == 1


def test_whitespace_flexible_match():
    text = "Address: 12 Main\nStreet, Springfield"
    resolution = resolve(text, [Evidence("street_address", "12 Main Street")])
    assert len(resolution.spans) == 1
    span = resolution.spans[0]
    assert text[span.start : span.end] == "12 Main\nStreet"


def test_trailing_punctuation_is_trimmed():
    resolution = resolve(TEXT, [Evidence("email", "jane@example.com.")])
    assert len(resolution.spans) == 1
    assert resolution.spans[0].text == "jane@example.com"


def test_overlapping_spans_resolve_to_the_longest():
    text = "Contact Jane Doe today."
    resolution = resolve(
        text, [Evidence("first_name", "Jane Doe"), Evidence("last_name", "Doe")]
    )
    assert len(resolution.spans) == 1
    assert resolution.spans[0].text == "Jane Doe"
    assert resolution.dropped_overlaps


def test_duplicate_evidence_yields_one_span():
    resolution = resolve(TEXT, [Evidence("ssn", "123-45-6789"), Evidence("ssn", "123-45-6789")])
    assert len(resolution.spans) == 1


def test_spans_are_sorted_by_position():
    resolution = resolve(
        TEXT, [Evidence("ssn", "123-45-6789"), Evidence("first_name", "Jane")]
    )
    starts = [s.start for s in resolution.spans]
    assert starts == sorted(starts)


def test_tag_text_matches_nemotron_format():
    text = "I, Jason, am applying."
    spans = [Span(start=3, end=8, text="Jason", label="first_name")]
    assert tag_text(text, spans) == "I, [Jason]first_name, am applying."


def test_tag_text_rejects_overlaps():
    spans = [
        Span(start=0, end=8, text="Jane Doe", label="first_name"),
        Span(start=5, end=8, text="Doe", label="last_name"),
    ]
    with pytest.raises(ValueError):
        tag_text("Jane Doe here", spans)


def test_count_by_label_is_descending():
    text = "Jane and Jane and Bob"
    resolution = resolve(text, [Evidence("first_name", "Jane"), Evidence("first_name", "Bob")])
    counts = resolution.count_by_label()
    assert counts == {"first_name": 3}


def test_realign_after_text_shift():
    old = "Header\nPatient: Jane Doe"
    new = "Patient: Jane Doe"
    spans = [Span(start=old.index("Jane"), end=old.index("Jane") + 4, text="Jane", label="first_name")]
    realigned, lost = realign(old, new, spans)
    assert not lost
    assert new[realigned[0].start : realigned[0].end] == "Jane"


def test_realign_reports_loss():
    spans = [Span(start=0, end=4, text="Jane", label="first_name")]
    realigned, lost = realign("Jane Doe", "no name here", spans)
    assert realigned == []
    assert len(lost) == 1


def test_realign_prefers_the_nearest_occurrence():
    old = "x" * 50 + "Jane" + "y" * 50 + "Jane"
    new = "Jane" + "y" * 50 + "Jane"
    span = Span(start=104, end=108, text="Jane", label="first_name")
    realigned, _ = realign(old, new, [span])
    assert realigned[0].start == 54
