"""Backend behaviour, exercised without touching the network or spawning a CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from datax.backends import (
    AnthropicBackend,
    BackendError,
    ClaudeCodeBackend,
    _claude_code_usage,
    make_backend,
)
from datax.judge import JudgeConfig
from datax.taxonomy import default_taxonomy

PAYLOAD = {
    "subcategory_path": "finance/financial_reporting/annual_report",
    "industry_confidence": 0.9,
    "industry_rationale": "Annual report.",
    "document_type": "Annual Report",
    "document_description": "d",
    "document_format": "unstructured",
    "locale": "us",
    "pii_evidence": [],
}


class StubClient:
    def __init__(self, message):
        self._message = message
        self.messages = SimpleNamespace(create=lambda **params: self._message)

    def with_options(self, **_kwargs):
        return self


def anthropic_message(text, stop_reason="end_turn", **extra):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=6000,
            cache_creation_input_tokens=0,
        ),
        **extra,
    )


# -- registry --------------------------------------------------------------


def test_make_backend_resolves_both_names():
    assert isinstance(make_backend("claude-code"), ClaudeCodeBackend)
    assert make_backend("claude_code").name == "claude-code"


def test_make_backend_rejects_unknown():
    with pytest.raises(BackendError, match="unknown backend"):
        make_backend("gpt")


# -- anthropic backend -----------------------------------------------------


def test_anthropic_backend_returns_payload():
    backend = AnthropicBackend(StubClient(anthropic_message(json.dumps(PAYLOAD))))
    response = backend.classify("some text", default_taxonomy(), JudgeConfig())
    assert response.ok
    assert response.payload["document_type"] == "Annual Report"
    assert response.usage["cache_read_input_tokens"] == 6000


def test_anthropic_backend_reports_refusal_without_touching_content():
    message = SimpleNamespace(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
        usage=SimpleNamespace(input_tokens=1, output_tokens=0, cache_read_input_tokens=0),
    )
    response = AnthropicBackend(StubClient(message)).classify(
        "t", default_taxonomy(), JudgeConfig()
    )
    assert response.refusal == "cyber"
    assert not response.ok


def test_anthropic_backend_reports_truncated_json():
    backend = AnthropicBackend(StubClient(anthropic_message('{"a":', stop_reason="max_tokens")))
    response = backend.classify("t", default_taxonomy(), JudgeConfig())
    assert "max_tokens" in response.error


# -- claude-code backend ---------------------------------------------------


def test_claude_code_options_strip_the_agent_down():
    pytest.importorskip("claude_agent_sdk")
    options = ClaudeCodeBackend().build_options(default_taxonomy(), JudgeConfig())
    # No tools, no project settings, one turn: this is a classifier, not an agent.
    assert options.tools == []
    assert options.setting_sources == []
    assert options.max_turns == 1
    assert options.model == "claude-opus-5"
    # The schema is enforced, not merely requested.
    assert options.output_format["type"] == "json_schema"
    enum = options.output_format["schema"]["properties"]["subcategory_path"]["enum"]
    assert "finance/financial_reporting/annual_report" in enum


def test_claude_code_replaces_rather_than_appends_the_system_prompt():
    """Appending would keep Claude Code's ~38K-token coding prompt in every call."""
    pytest.importorskip("claude_agent_sdk")
    options = ClaudeCodeBackend().build_options(default_taxonomy(), JudgeConfig())
    assert isinstance(options.system_prompt, str)
    assert "PII / PHI TAXONOMY" in options.system_prompt


def test_claude_code_system_prompt_is_byte_stable():
    """Cache reuse across separate CLI invocations depends on this exactly."""
    pytest.importorskip("claude_agent_sdk")
    taxonomy, config = default_taxonomy(), JudgeConfig()
    a = ClaudeCodeBackend().build_options(taxonomy, config).system_prompt
    b = ClaudeCodeBackend().build_options(taxonomy, config).system_prompt
    assert a == b


def test_claude_code_budget_is_passed_through_when_set():
    pytest.importorskip("claude_agent_sdk")
    options = ClaudeCodeBackend().build_options(
        default_taxonomy(), JudgeConfig(max_cost_usd=0.25)
    )
    assert options.max_budget_usd == 0.25


def test_claude_code_usage_normalises_missing_fields():
    assert _claude_code_usage({}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert _claude_code_usage({"output_tokens": 7})["output_tokens"] == 7


# -- backend-agnostic outcome handling -------------------------------------


def test_outcome_from_backend_handles_each_response_kind(tmp_path):
    from datax.backends import BackendResponse
    from datax.extract import ExtractedDocx
    from datax.judge import outcome_from_backend
    from datax.manifest import SourceInfo
    from pathlib import Path

    extracted = ExtractedDocx(
        path=Path("/tmp/x.docx"),
        text="Annual Report for the year.",
        sha256="c" * 64,
        size_bytes=10,
        paragraph_count=1,
        table_count=0,
        word_count=5,
        has_headers_or_footers=False,
    )
    kwargs = dict(
        extracted=extracted,
        source=SourceInfo(provider="t", reference="r"),
        taxonomy=default_taxonomy(),
        config=JudgeConfig(),
    )

    ok = outcome_from_backend(BackendResponse(payload=PAYLOAD, cost_usd=0.02), **kwargs)
    assert ok.ok and ok.cost_usd == 0.02

    refused = outcome_from_backend(BackendResponse(refusal="cyber"), **kwargs)
    assert not refused.ok and refused.refusal == "cyber"

    failed = outcome_from_backend(BackendResponse(error="boom"), **kwargs)
    assert not failed.ok and failed.error == "boom"

    malformed = outcome_from_backend(BackendResponse(payload={"nonsense": 1}), **kwargs)
    assert not malformed.ok and "invalid judge payload" in malformed.error
