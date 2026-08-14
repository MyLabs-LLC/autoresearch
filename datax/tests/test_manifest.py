"""Manifest construction, validation, and Nemotron schema compatibility."""

from __future__ import annotations

import json

from datax.manifest import (
    FileInfo,
    IndustryLabel,
    JudgeInfo,
    Record,
    SourceInfo,
    build_record,
    read_manifest,
    validate_record,
    write_manifest,
)
from datax.spans import Span
from datax.taxonomy import default_taxonomy

TEXT = "Patient: Jane Doe\nMRN: 88213-A"

# The nine fields a consumer of nvidia/Nemotron-PII expects to find.
NEMOTRON_FIELDS = {
    "uid",
    "domain",
    "document_type",
    "document_description",
    "document_format",
    "locale",
    "text",
    "spans",
    "text_tagged",
}


def make_record(spans=None, **overrides):
    taxonomy = default_taxonomy()
    spans = spans if spans is not None else [
        Span(start=9, end=13, text="Jane", label="first_name"),
        Span(start=23, end=30, text="88213-A", label="medical_record_number"),
    ]
    kwargs = dict(
        taxonomy=taxonomy,
        uid="test-uid",
        text=TEXT,
        file=FileInfo(path="/tmp/doc.docx", sha256="a" * 64, size_bytes=100, word_count=6),
        source=SourceInfo(provider="test", reference="unit-test", synthetic=True),
        industry=IndustryLabel(
            id="healthcare",
            category="clinical_records",
            subcategory="medical_record",
            confidence=0.9,
        ),
        spans=spans,
        document_type="Medical Record",
        document_format="unstructured",
        locale="us",
    )
    kwargs.update(overrides)
    return build_record(**kwargs)


def test_record_carries_the_nemotron_fields():
    record = make_record()
    assert NEMOTRON_FIELDS <= set(record.to_dict())


def test_span_dicts_match_nemotron_shape():
    record = make_record()
    assert set(record.spans[0]) == {"start", "end", "text", "label"}


def test_text_tagged_matches_nemotron_inline_format():
    record = make_record()
    assert record.text_tagged.startswith("Patient: [Jane]first_name")


def test_domain_uses_the_nemotron_display_label():
    assert make_record().domain == "Healthcare"


def test_pii_summary_is_derived_not_trusted():
    record = make_record()
    assert record.pii.has_pii
    assert record.pii.labels == ["first_name", "medical_record_number"]
    assert record.pii.max_sensitivity == "critical"  # medical_record_number
    assert record.pii.contains_phi
    assert record.pii.contains_special_category  # medical_record_number is special-category


def test_document_without_pii():
    record = make_record(spans=[])
    assert not record.pii.has_pii
    assert record.pii.labels == []
    assert record.pii.max_sensitivity == "none"
    assert record.text_tagged == TEXT


def test_valid_record_has_no_problems():
    assert validate_record(make_record(), default_taxonomy()) == []


def test_validation_catches_offset_drift():
    """The failure mode that matters: offsets that no longer point at their text."""
    record = make_record()
    record.spans[0]["start"] = 0
    record.spans[0]["end"] = 4
    problems = validate_record(record, default_taxonomy())
    assert any("text mismatch" in p for p in problems)


def test_validation_catches_out_of_range_span():
    record = make_record()
    record.spans[0]["end"] = 9999
    problems = validate_record(record, default_taxonomy())
    assert any("outside text" in p for p in problems)


def test_validation_catches_unknown_pii_label():
    record = make_record()
    record.spans[0]["label"] = "social_security_number"
    problems = validate_record(record, default_taxonomy())
    assert any("not in PII taxonomy" in p for p in problems)


def test_validation_catches_subcategory_from_wrong_industry():
    record = make_record(
        industry=IndustryLabel(
            id="healthcare", category="clinical_records", subcategory="balance_sheet"
        )
    )
    problems = validate_record(record, default_taxonomy())
    assert any("does not exist under industry" in p for p in problems)


def test_validation_catches_category_subcategory_mismatch():
    record = make_record(
        industry=IndustryLabel(
            id="healthcare", category="coverage_and_billing", subcategory="medical_record"
        )
    )
    problems = validate_record(record, default_taxonomy())
    assert any("belongs to category" in p for p in problems)


def test_validation_catches_pii_summary_drift():
    record = make_record()
    record.pii.labels = ["ssn"]
    problems = validate_record(record, default_taxonomy())
    assert any("disagrees with span labels" in p for p in problems)


def test_other_industry_must_not_carry_subcategory():
    record = make_record(
        industry=IndustryLabel(id="other", category=None, subcategory="medical_record")
    )
    problems = validate_record(record, default_taxonomy())
    assert any("must not carry a subcategory" in p for p in problems)


def test_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "manifest.jsonl"
    original = make_record()
    write_manifest(path, [original])
    (loaded,) = list(read_manifest(path))
    assert loaded.to_dict() == original.to_dict()


def test_manifest_lines_are_stable_json(tmp_path):
    """Sorted keys keep the file diffable across runs."""
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_record()])
    line = path.read_text(encoding="utf-8").strip()
    assert json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False) == line


def test_validate_manifest_detects_duplicate_hashes(tmp_path):
    from datax.manifest import validate_manifest

    path = tmp_path / "manifest.jsonl"
    a = make_record()
    b = Record.from_dict({**a.to_dict(), "uid": "other-uid"})
    write_manifest(path, [a, b])
    report = validate_manifest(path, default_taxonomy())
    assert report.duplicate_sha256
    assert not report.ok
