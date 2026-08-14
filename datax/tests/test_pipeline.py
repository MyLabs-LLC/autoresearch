"""End-to-end pipeline test with a stubbed judge.

Covers the one path unit tests otherwise miss: document -> judge -> manifest ->
validate -> evaluate. The API client is stubbed, so this runs offline and in CI, but
every other component is the real one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from datax.backends import AnthropicBackend
from datax.evaluate import compare
from datax.extract import extract_docx
from datax.judge import JudgeConfig, judge_document
from datax.manifest import (
    FileInfo,
    IndustryLabel,
    JudgeInfo,
    SourceInfo,
    build_record,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from datax.spans import Span
from datax.taxonomy import default_taxonomy

pytest.importorskip("docx", reason="writing .docx requires python-docx")

from datax.docxio import write_text_docx  # noqa: E402

TEXT = (
    "Discharge Summary\n"
    "Patient: Jane Doe\n"
    "MRN: 88213-A\n"
    "Date of birth: 1984-02-11\n"
    "Discharged home in stable condition."
)

JUDGE_PAYLOAD = {
    "subcategory_path": "healthcare/admission_and_discharge/discharge_summary",
    "industry_confidence": 0.95,
    "industry_rationale": "Summary issued on discharge from hospital.",
    "document_type": "Discharge Summary",
    "document_description": "Discharge summary for a single inpatient episode.",
    "document_format": "unstructured",
    "locale": "us",
    "pii_evidence": [
        {"label": "first_name", "text": "Jane"},
        {"label": "last_name", "text": "Doe"},
        {"label": "medical_record_number", "text": "88213-A"},
        {"label": "date_of_birth", "text": "1984-02-11"},
        # A hallucinated quote: must not become a span.
        {"label": "ssn", "text": "123-45-6789"},
    ],
}


class StubClient:
    """Minimal stand-in for anthropic.Anthropic."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, **_kwargs):
        return self

    def _create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self._payload))],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=90,
                cache_read_input_tokens=6400,
                cache_creation_input_tokens=0,
            ),
        )


@pytest.fixture()
def document(tmp_path):
    path = tmp_path / "discharge.docx"
    write_text_docx(path, TEXT)
    return extract_docx(path)


def test_end_to_end_judge_to_validated_manifest(tmp_path, document):
    taxonomy = default_taxonomy()
    client = StubClient(JUDGE_PAYLOAD)
    source = SourceInfo(provider="test", reference="pipeline-test", synthetic=True)

    outcome = judge_document(AnthropicBackend(client), document, source, taxonomy=taxonomy, config=JudgeConfig())
    assert outcome.ok

    record = outcome.record
    # Real quotes resolved; the hallucinated SSN did not.
    assert set(record.pii.labels) == {
        "first_name",
        "last_name",
        "medical_record_number",
        "date_of_birth",
    }
    assert "ssn" not in record.pii.labels
    assert any(e["reason"] == "not_found_in_text" for e in record.pii.unresolved_evidence)

    # Offsets point at the real characters in the real file.
    for span in record.spans:
        assert document.text[span["start"] : span["end"]] == span["text"]

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [record])
    report = validate_manifest(manifest, taxonomy)
    assert report.ok, report.problems


def test_evaluate_scores_judge_against_gold(tmp_path, document):
    taxonomy = default_taxonomy()

    def span_for(surface, label):
        start = TEXT.index(surface)
        return Span(start=start, end=start + len(surface), text=surface, label=label)

    gold = build_record(
        taxonomy=taxonomy,
        uid="gold-1",
        text=document.text,
        file=FileInfo(
            path=str(document.path),
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            word_count=document.word_count,
        ),
        source=SourceInfo(provider="nemotron", reference="gold", synthetic=True),
        industry=IndustryLabel(
            id="healthcare",
            category="admission_and_discharge",
            subcategory="discharge_summary",
            confidence=1.0,
        ),
        spans=[
            span_for("Jane", "first_name"),
            span_for("Doe", "last_name"),
            span_for("88213-A", "medical_record_number"),
            span_for("1984-02-11", "date_of_birth"),
        ],
        document_type="Discharge Summary",
        judge=JudgeInfo(label_source="gold"),
    )

    client = StubClient(JUDGE_PAYLOAD)
    predicted = judge_document(
        AnthropicBackend(client),
        document,
        SourceInfo(provider="test", reference="p", synthetic=True),
        taxonomy=taxonomy,
    ).record

    report = compare([gold], [predicted])
    assert report.matched == 1
    assert report.industry_accuracy == 1.0
    assert report.subcategory_accuracy == 1.0
    assert report.doc_level.f1 == 1.0
    assert report.span_level.f1 == 1.0


def test_evaluate_penalises_a_wrong_industry(tmp_path, document):
    taxonomy = default_taxonomy()
    common = dict(
        taxonomy=taxonomy,
        text=document.text,
        file=FileInfo(
            path=str(document.path),
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            word_count=document.word_count,
        ),
        source=SourceInfo(provider="t", reference="r", synthetic=True),
        spans=[],
        document_type="X",
    )
    gold = build_record(
        uid="g",
        industry=IndustryLabel(id="healthcare", category="clinical_records", subcategory="medical_record"),
        **common,
    )
    predicted = build_record(
        uid="p",
        industry=IndustryLabel(id="finance", category="financial_reporting", subcategory="annual_report"),
        **common,
    )
    report = compare([gold], [predicted])
    assert report.industry_accuracy == 0.0
    assert report.industry_confusion["healthcare"]["finance"] == 1


def test_records_are_joined_on_content_hash_not_uid(tmp_path, document):
    """Judged records get a fresh uid; the join must survive that."""
    taxonomy = default_taxonomy()
    client = StubClient(JUDGE_PAYLOAD)
    predicted = judge_document(
        AnthropicBackend(client), document, SourceInfo(provider="t", reference="r"),
        taxonomy=taxonomy,
    ).record
    gold = build_record(
        taxonomy=taxonomy,
        uid="a-completely-different-uid",
        text=document.text,
        file=FileInfo(
            path=str(document.path),
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            word_count=document.word_count,
        ),
        source=SourceInfo(provider="nemotron", reference="g", synthetic=True),
        industry=IndustryLabel(
            id="healthcare", category="admission_and_discharge", subcategory="discharge_summary"
        ),
        spans=[],
        document_type="Discharge Summary",
        judge=JudgeInfo(label_source="gold"),
    )
    assert gold.uid != predicted.uid
    assert compare([gold], [predicted]).matched == 1


def test_manifest_survives_a_write_read_validate_cycle(tmp_path, document):
    taxonomy = default_taxonomy()
    client = StubClient(JUDGE_PAYLOAD)
    record = judge_document(
        AnthropicBackend(client), document, SourceInfo(provider="t", reference="r"),
        taxonomy=taxonomy,
    ).record
    path = tmp_path / "m.jsonl"
    write_manifest(path, [record])
    reloaded = list(read_manifest(path))
    assert reloaded[0].to_dict() == record.to_dict()
    assert validate_manifest(path, taxonomy).ok
