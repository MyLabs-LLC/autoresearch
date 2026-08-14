"""Judge prompt, schema, and response handling -- all without calling the API.

The API call itself is the one part that cannot be unit tested; everything around it
(schema construction, cache placement, refusal handling, payload -> record) can, and
those are where the bugs actually live.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from datax.extract import ExtractedDocx
from datax.judge import (
    JudgeConfig,
    build_request_params,
    build_schema,
    build_system_blocks,
    build_user_content,
    leaf_paths,
    outcome_from_message,
    record_from_payload,
)
from datax.manifest import SourceInfo, validate_record
from datax.taxonomy import default_taxonomy
from pathlib import Path

TEXT = "Discharge Summary\nPatient: Jane Doe\nMRN: 88213-A\nDischarged 2024-03-01."


def make_extracted(text: str = TEXT) -> ExtractedDocx:
    return ExtractedDocx(
        path=Path("/tmp/doc.docx"),
        text=text,
        sha256="b" * 64,
        size_bytes=1234,
        paragraph_count=4,
        table_count=0,
        word_count=len(text.split()),
        has_headers_or_footers=False,
    )


SOURCE = SourceInfo(provider="test", reference="unit-test", synthetic=True)

PAYLOAD = {
    "subcategory_path": "healthcare/admission_and_discharge/discharge_summary",
    "industry_confidence": 0.93,
    "industry_rationale": "Summary issued at discharge.",
    "document_type": "Discharge Summary",
    "document_description": "Hospital discharge summary for a single patient.",
    "document_format": "unstructured",
    "locale": "us",
    "pii_evidence": [
        {"label": "first_name", "text": "Jane"},
        {"label": "last_name", "text": "Doe"},
        {"label": "medical_record_number", "text": "88213-A"},
        {"label": "date", "text": "2024-03-01"},
    ],
}


# -- schema ----------------------------------------------------------------


def test_schema_constrains_every_level_with_one_enum():
    schema = build_schema(default_taxonomy())
    paths = schema["properties"]["subcategory_path"]["enum"]
    assert "healthcare/admission_and_discharge/discharge_summary" in paths
    assert "other" in paths
    assert len(paths) == len(set(paths))


def test_schema_obeys_structured_output_restrictions():
    """Structured outputs reject numeric/length constraints and open objects."""
    schema = build_schema(default_taxonomy())

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for banned in ("minimum", "maximum", "minLength", "maxLength", "pattern", "minItems"):
                assert banned not in node, banned
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_schema_pii_enum_is_the_taxonomy():
    schema = build_schema(default_taxonomy())
    enum = schema["properties"]["pii_evidence"]["items"]["properties"]["label"]["enum"]
    assert enum == default_taxonomy().pii_ids


def test_leaf_paths_are_sorted_for_byte_stability():
    paths = leaf_paths(default_taxonomy())
    assert paths == sorted(paths)


# -- prompt ----------------------------------------------------------------


def test_cache_breakpoint_sits_on_the_last_system_block():
    blocks = build_system_blocks(default_taxonomy())
    assert "cache_control" not in blocks[0]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_system_prompt_is_byte_identical_across_calls():
    """A single changed byte in the prefix throws away the cache."""
    assert build_system_blocks(default_taxonomy()) == build_system_blocks(default_taxonomy())


def test_document_goes_after_the_cache_breakpoint():
    params, _ = build_request_params(TEXT, default_taxonomy(), JudgeConfig())
    assert TEXT in params["messages"][0]["content"]
    assert all(TEXT not in block["text"] for block in params["system"])


def test_request_omits_sampling_parameters():
    """temperature/top_p/top_k are rejected by this model family."""
    params, _ = build_request_params(TEXT, default_taxonomy(), JudgeConfig())
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in params


def test_long_documents_are_truncated_and_flagged():
    config = JudgeConfig(max_document_chars=50)
    content, truncated = build_user_content("x" * 500, config)
    assert truncated
    assert "truncated" in content


# -- payload -> record -----------------------------------------------------


def test_record_from_payload_produces_a_valid_record():
    record = record_from_payload(
        PAYLOAD,
        extracted=make_extracted(),
        source=SOURCE,
        taxonomy=default_taxonomy(),
        config=JudgeConfig(),
        truncated=False,
    )
    assert validate_record(record, default_taxonomy()) == []
    assert record.industry.id == "healthcare"
    assert record.industry.category == "admission_and_discharge"
    assert record.industry.subcategory == "discharge_summary"
    assert record.pii.contains_phi


def test_other_path_clears_category_and_subcategory():
    record = record_from_payload(
        {**PAYLOAD, "subcategory_path": "other"},
        extracted=make_extracted(),
        source=SOURCE,
        taxonomy=default_taxonomy(),
        config=JudgeConfig(),
        truncated=False,
    )
    assert record.industry.id == "other"
    assert record.industry.subcategory is None
    assert validate_record(record, default_taxonomy()) == []


def test_hallucinated_evidence_never_becomes_a_span():
    payload = {**PAYLOAD, "pii_evidence": [{"label": "ssn", "text": "999-99-9999"}]}
    record = record_from_payload(
        payload,
        extracted=make_extracted(),
        source=SOURCE,
        taxonomy=default_taxonomy(),
        config=JudgeConfig(),
        truncated=False,
    )
    assert record.spans == []
    assert record.pii.unresolved_evidence[0]["reason"] == "not_found_in_text"


def test_confidence_is_clamped():
    record = record_from_payload(
        {**PAYLOAD, "industry_confidence": 7.5},
        extracted=make_extracted(),
        source=SOURCE,
        taxonomy=default_taxonomy(),
        config=JudgeConfig(),
        truncated=False,
    )
    assert record.industry.confidence == 1.0


def test_truncation_is_recorded_on_the_record():
    record = record_from_payload(
        PAYLOAD,
        extracted=make_extracted(),
        source=SOURCE,
        taxonomy=default_taxonomy(),
        config=JudgeConfig(max_document_chars=10),
        truncated=True,
    )
    reasons = {e.get("reason") for e in record.pii.unresolved_evidence}
    assert "document_truncated_for_judge" in reasons


# -- response handling -----------------------------------------------------


def fake_message(text: str, stop_reason: str = "end_turn", **extra):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=0,
        ),
        **extra,
    )


def _outcome(message):
    return outcome_from_message(
        message,
        extracted=make_extracted(),
        source=SOURCE,
        taxonomy=default_taxonomy(),
        config=JudgeConfig(),
        truncated=False,
    )


def test_successful_response_yields_a_record():
    outcome = _outcome(fake_message(json.dumps(PAYLOAD)))
    assert outcome.ok
    assert outcome.usage["cache_read_input_tokens"] == 900


def test_refusal_is_handled_before_reading_content():
    """A refusal can carry empty content; indexing it blindly would raise."""
    message = SimpleNamespace(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
        usage=SimpleNamespace(input_tokens=10, output_tokens=0, cache_read_input_tokens=0),
    )
    outcome = _outcome(message)
    assert not outcome.ok
    assert outcome.refusal == "cyber"


def test_max_tokens_is_reported_not_parsed():
    outcome = _outcome(fake_message('{"subcategory_path": "heal', stop_reason="max_tokens"))
    assert not outcome.ok
    assert "max_tokens" in outcome.error


def test_unparseable_output_is_reported():
    outcome = _outcome(fake_message("not json at all"))
    assert not outcome.ok
    assert "could not parse" in outcome.error
