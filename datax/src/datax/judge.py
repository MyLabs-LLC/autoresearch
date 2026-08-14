"""LLM-as-judge classification of a document against the datax taxonomies.

Design decisions worth knowing before changing anything here:

**The judge never reports character offsets.** It reports a label plus the verbatim
text it saw, and :mod:`datax.spans` locates that quote in the document. Offsets are
then correct by construction; a hallucinated quote simply fails to resolve and is
recorded as unresolved evidence rather than becoming a wrong span.

**The subcategory is one enum, not three fields.** A leaf is identified by its full
``industry/category/subcategory`` path, so a single ``enum`` constrains all three
levels at once. JSON Schema conditionals are not supported by structured outputs, so
this is the only way to make an industry-dependent vocabulary machine-enforced.

**The taxonomy is rendered once and cached.** It is the large, stable part of the
prompt and sits before the cache breakpoint; the document is volatile and sits after
it. Rendering is deterministic (sorted, no timestamps) because a single changed byte
in the prefix costs the whole cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable, Sequence

from .extract import ExtractedDocx
from .manifest import (
    FileInfo,
    IndustryLabel,
    JudgeInfo,
    Record,
    SourceInfo,
    build_record,
    new_uid,
)
from .spans import Evidence, resolve
from .taxonomy import OTHER_INDUSTRY, Taxonomy, default_taxonomy

DEFAULT_MODEL = "claude-opus-5"
OTHER_PATH = "other"


class JudgeError(RuntimeError):
    pass


@dataclass
class JudgeConfig:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    """`low` and `medium` are unusually strong on this model and this is a
    high-volume classification task; raise to `high` if evaluation shows it pays."""
    max_tokens: int = 16000
    max_document_chars: int = 60_000
    """Documents longer than this are truncated for the judge only. Spans still
    resolve against the full text, but PII past the cut is not seen -- so truncation is
    recorded on the record rather than being silent."""
    occurrence_mode: str = "all"
    cache_taxonomy: bool = True
    max_retries: int = 3
    max_cost_usd: float = 0.0
    """Per-document spend ceiling, enforced by the claude-code backend. 0 disables."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------


def leaf_paths(taxonomy: Taxonomy) -> list[str]:
    """Every ``industry/category/subcategory`` path, sorted for byte-stability."""
    return sorted(sub.path for sub in taxonomy.subcategories())


@lru_cache(maxsize=4)
def _render_taxonomy_block(industry_version: str, pii_version: str) -> str:
    """Render the taxonomy reference. Cached on version so repeated calls return the
    identical string and the prompt cache keeps hitting."""
    taxonomy = default_taxonomy()
    lines: list[str] = []

    lines.append(f"# INDUSTRY TAXONOMY (version {industry_version})")
    lines.append("")
    lines.append(
        "Choose exactly one leaf path. The path encodes industry, category and "
        "subcategory as `industry/category/subcategory`."
    )
    lines.append("")
    for industry in taxonomy.industries:
        lines.append(f"## {industry.id} -- {industry.label}")
        lines.append(industry.description)
        for category in industry.categories:
            lines.append(f"  ### {category.id} -- {category.label}")
            for sub in category.subcategories:
                lines.append(f"    - {sub.path}: {sub.description}")
        lines.append("")
    lines.append(
        f"## {OTHER_PATH}\n"
        "    Use this when the document does not belong to healthcare, finance or "
        "government, or is too fragmentary to place. Do not force a fit."
    )
    lines.append("")

    lines.append(f"# PII / PHI TAXONOMY (version {pii_version})")
    lines.append("")
    lines.append("Label every entity mention that matches one of these categories.")
    lines.append("")
    current_group = None
    for label in taxonomy.pii_labels:
        if label.group != current_group:
            current_group = label.group
            lines.append(f"## {current_group}")
        flags = []
        if label.phi:
            flags.append("PHI")
        if label.special_category:
            flags.append("special-category")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"  - {label.id}{suffix}: {label.description}")
    return "\n".join(lines)


SYSTEM_INSTRUCTIONS = """\
You classify documents for a machine-learning dataset. You are given the full text of \
one document extracted from a .docx file. Return a single structured classification.

## Industry
Pick the one leaf path that best describes what the document *is*, not what it \
mentions. A hospital's procurement contract is a government/finance-style procurement \
document only if it was produced by a public body; from a hospital it is healthcare. \
When the document does not fit healthcare, finance or government, return `other` and \
leave the rationale short.

## Document type and description
`document_type` is a short free-text name for the document as a practitioner would \
say it ("Discharge Summary", "Statement of Work"). `document_description` is one or \
two sentences on what the document contains and who it is for.

## Format and locale
`document_format` is `structured` for forms, tables, invoices and field/value \
layouts; `unstructured` for prose, letters, notes and reports. `locale` is `us` when \
the conventions are United States (SSN, ZIP, state abbreviations, MM/DD/YYYY, USD) \
and `intl` otherwise.

## PII and PHI
Report every mention of personal information as an evidence item: the label, plus the \
exact substring from the document.

Rules that matter:
- Quote **verbatim**. Copy the characters exactly as they appear. Do not normalise \
case, spacing, or punctuation, and do not paraphrase. A quote that does not appear \
in the document is discarded.
- Quote the **entity only**, not the surrounding sentence or the field label. For \
"Patient Name: Jane Doe", the first_name quote is "Jane" and the last_name quote is \
"Doe".
- Report each distinct mention once. Repeated occurrences of the same string are \
handled automatically; you do not need to list them again.
- Label what the value *is* in context, not what it looks like. A number labelled \
"Member ID" on an insurance form is `health_plan_beneficiary_number`, not \
`unique_id`.
- Placeholders, examples and blank form fields are not PII. "Name: ____________", \
"e.g. john@example.com" and "XXX-XX-XXXX" must not be reported.
- Organisation names are `company_name`. Do not report a person's name as \
`company_name` or vice versa.
- If the document contains no personal information at all, return an empty evidence \
list. An empty list is a valid and common answer; do not invent entities to fill it.

Judge only what is in the document. Do not infer facts that are not written down."""


def build_system_blocks(taxonomy: Taxonomy, *, cache: bool = True) -> list[dict[str, Any]]:
    """System prompt as content blocks, with the cache breakpoint on the last one.

    Instructions and taxonomy are both stable, so a single breakpoint at the end
    caches the entire prefix.
    """
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
        {
            "type": "text",
            "text": _render_taxonomy_block(taxonomy.industry_version, taxonomy.pii_version),
        },
    ]
    if cache:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_schema(taxonomy: Taxonomy) -> dict[str, Any]:
    """JSON Schema for the judge's response.

    Constrained to the subset structured outputs supports: no numeric or length
    constraints, every object closed with ``additionalProperties: false``, and every
    property required.
    """
    return {
        "type": "object",
        "properties": {
            "subcategory_path": {
                "type": "string",
                "description": "Full industry/category/subcategory path, or 'other'.",
                "enum": leaf_paths(taxonomy) + [OTHER_PATH],
            },
            "industry_confidence": {
                "type": "number",
                "description": "Confidence in the chosen path, between 0 and 1.",
            },
            "industry_rationale": {
                "type": "string",
                "description": "One sentence justifying the chosen path.",
            },
            "document_type": {
                "type": "string",
                "description": "Short practitioner-facing name for this kind of document.",
            },
            "document_description": {
                "type": "string",
                "description": "One or two sentences describing the document's contents.",
            },
            "document_format": {"type": "string", "enum": ["structured", "unstructured"]},
            "locale": {"type": "string", "enum": ["us", "intl"]},
            "pii_evidence": {
                "type": "array",
                "description": "One entry per distinct personal-information mention.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": taxonomy.pii_ids},
                        "text": {
                            "type": "string",
                            "description": "Exact substring copied from the document.",
                        },
                    },
                    "required": ["label", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "subcategory_path",
            "industry_confidence",
            "industry_rationale",
            "document_type",
            "document_description",
            "document_format",
            "locale",
            "pii_evidence",
        ],
        "additionalProperties": False,
    }


def build_user_content(text: str, config: JudgeConfig) -> tuple[str, bool]:
    """Render the user turn. Returns the content and whether the text was truncated."""
    truncated = len(text) > config.max_document_chars
    body = text[: config.max_document_chars] if truncated else text
    note = (
        "\n\n[The document was truncated for length; classify what is shown.]"
        if truncated
        else ""
    )
    return f"<document>\n{body}\n</document>{note}", truncated


def build_request_params(
    text: str, taxonomy: Taxonomy, config: JudgeConfig
) -> tuple[dict[str, Any], bool]:
    content, truncated = build_user_content(text, config)
    params: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": build_system_blocks(taxonomy, cache=config.cache_taxonomy),
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": build_schema(taxonomy)},
        },
    }
    return params, truncated


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


@dataclass
class JudgeOutcome:
    record: Record | None
    error: str | None = None
    refusal: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0

    @property
    def ok(self) -> bool:
        return self.record is not None


def _first_text(message: Any) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise JudgeError("response contained no text block")


def _split_path(path: str) -> tuple[str, str | None, str | None]:
    if path == OTHER_PATH:
        return OTHER_INDUSTRY, None, None
    parts = path.split("/")
    if len(parts) != 3:
        raise JudgeError(f"malformed subcategory path {path!r}")
    return parts[0], parts[1], parts[2]


def record_from_payload(
    payload: dict[str, Any],
    *,
    extracted: ExtractedDocx,
    source: SourceInfo,
    taxonomy: Taxonomy,
    config: JudgeConfig,
    truncated: bool,
    usage: dict[str, int] | None = None,
    uid: str | None = None,
) -> Record:
    """Turn a validated judge payload into a manifest record."""
    industry_id, category, subcategory = _split_path(payload["subcategory_path"])

    evidence = [
        Evidence(label=item["label"], text=item["text"]) for item in payload.get("pii_evidence", [])
    ]
    resolution = resolve(
        extracted.text,
        evidence,
        allowed_labels=set(taxonomy.pii_ids),
        mode="first" if config.occurrence_mode == "first" else "all",
    )

    unresolved = list(resolution.unresolved)
    if truncated:
        unresolved.append(
            {
                "reason": "document_truncated_for_judge",
                "judged_chars": config.max_document_chars,
                "total_chars": len(extracted.text),
            }
        )

    confidence = float(payload.get("industry_confidence", 0.0))
    confidence = min(1.0, max(0.0, confidence))

    usage = usage or {}
    return build_record(
        taxonomy=taxonomy,
        uid=uid or new_uid(),
        text=extracted.text,
        file=FileInfo(
            path=str(extracted.path),
            sha256=extracted.sha256,
            size_bytes=extracted.size_bytes,
            word_count=extracted.word_count,
            paragraph_count=extracted.paragraph_count,
            table_count=extracted.table_count,
            has_headers_or_footers=extracted.has_headers_or_footers,
        ),
        source=source,
        industry=IndustryLabel(
            id=industry_id,
            category=category,
            subcategory=subcategory,
            confidence=confidence,
            rationale=payload.get("industry_rationale", ""),
        ),
        spans=resolution.spans,
        document_type=payload.get("document_type", ""),
        document_description=payload.get("document_description", ""),
        document_format=payload.get("document_format", "unstructured"),
        locale=payload.get("locale", "us"),
        judge=JudgeInfo(
            model=config.model,
            industry_taxonomy_version=taxonomy.industry_version,
            pii_taxonomy_version=taxonomy.pii_version,
            effort=config.effort,
            judged_at=_now(),
            label_source="llm_judge",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        ),
        unresolved_evidence=unresolved,
    )


def _usage_dict(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def outcome_from_message(
    message: Any,
    *,
    extracted: ExtractedDocx,
    source: SourceInfo,
    taxonomy: Taxonomy,
    config: JudgeConfig,
    truncated: bool,
) -> JudgeOutcome:
    """Interpret one API response. Handles refusal and truncation before parsing."""
    usage = _usage_dict(message)

    # Check stop_reason before touching content: on a refusal the content list is
    # empty or partial, and indexing it blindly raises instead of reporting.
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        return JudgeOutcome(record=None, refusal=category or "refusal", usage=usage)
    if stop_reason == "max_tokens":
        return JudgeOutcome(
            record=None,
            error="response hit max_tokens; JSON is incomplete. Raise max_tokens or "
            "lower max_document_chars.",
            usage=usage,
        )

    try:
        payload = json.loads(_first_text(message))
    except (JudgeError, json.JSONDecodeError) as exc:
        return JudgeOutcome(record=None, error=f"could not parse judge output: {exc}", usage=usage)

    try:
        record = record_from_payload(
            payload,
            extracted=extracted,
            source=source,
            taxonomy=taxonomy,
            config=config,
            truncated=truncated,
            usage=usage,
        )
    except (JudgeError, KeyError) as exc:
        return JudgeOutcome(record=None, error=f"invalid judge payload: {exc}", usage=usage)

    return JudgeOutcome(record=record, usage=usage)


# ---------------------------------------------------------------------------
# Synchronous judging
# ---------------------------------------------------------------------------


def outcome_from_backend(
    response,
    *,
    extracted: ExtractedDocx,
    source: SourceInfo,
    taxonomy: Taxonomy,
    config: JudgeConfig,
) -> JudgeOutcome:
    """Turn a :class:`datax.backends.BackendResponse` into a judge outcome."""
    if response.refusal:
        return JudgeOutcome(
            record=None, refusal=response.refusal, usage=response.usage,
            cost_usd=response.cost_usd,
        )
    if response.error or response.payload is None:
        return JudgeOutcome(
            record=None, error=response.error or "backend returned no payload",
            usage=response.usage, cost_usd=response.cost_usd,
        )
    try:
        record = record_from_payload(
            response.payload,
            extracted=extracted,
            source=source,
            taxonomy=taxonomy,
            config=config,
            truncated=response.truncated,
            usage=response.usage,
        )
    except (JudgeError, KeyError, TypeError, ValueError) as exc:
        return JudgeOutcome(
            record=None, error=f"invalid judge payload: {exc}",
            usage=response.usage, cost_usd=response.cost_usd,
        )
    return JudgeOutcome(record=record, usage=response.usage, cost_usd=response.cost_usd)


def judge_document(
    backend,
    extracted: ExtractedDocx,
    source: SourceInfo,
    *,
    taxonomy: Taxonomy | None = None,
    config: JudgeConfig | None = None,
) -> JudgeOutcome:
    """Classify one document using the given backend."""
    taxonomy = taxonomy or default_taxonomy()
    config = config or JudgeConfig()

    if extracted.is_empty:
        return JudgeOutcome(record=None, error="document has no extractable text")

    response = backend.classify(extracted.text, taxonomy, config)
    return outcome_from_backend(
        response, extracted=extracted, source=source, taxonomy=taxonomy, config=config
    )


# ---------------------------------------------------------------------------
# Batch judging
# ---------------------------------------------------------------------------


@dataclass
class BatchItem:
    custom_id: str
    extracted: ExtractedDocx
    source: SourceInfo
    truncated: bool = False


def submit_batch(
    client: Any,
    items: Sequence[BatchItem],
    *,
    taxonomy: Taxonomy | None = None,
    config: JudgeConfig | None = None,
) -> str:
    """Submit documents to the Batches API and return the batch id.

    Batch runs at half price and is the right choice for building a large corpus,
    where nothing is latency-sensitive. All requests share one cached system prefix.

    **Anthropic backend only** -- ``client`` is an ``anthropic.Anthropic``. The
    Batches API is a Messages API feature; Claude Code has no equivalent, so the
    claude-code backend judges documents one at a time.
    """
    taxonomy = taxonomy or default_taxonomy()
    config = config or JudgeConfig()

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = []
    for item in items:
        params, truncated = build_request_params(item.extracted.text, taxonomy, config)
        item.truncated = truncated
        requests.append(
            Request(custom_id=item.custom_id, params=MessageCreateParamsNonStreaming(**params))
        )

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def collect_batch(
    client: Any,
    batch_id: str,
    items: Iterable[BatchItem],
    *,
    taxonomy: Taxonomy | None = None,
    config: JudgeConfig | None = None,
) -> dict[str, JudgeOutcome]:
    """Collect a finished batch. Results arrive in arbitrary order, so they are keyed
    by ``custom_id`` and never by position."""
    taxonomy = taxonomy or default_taxonomy()
    config = config or JudgeConfig()
    by_id = {item.custom_id: item for item in items}
    outcomes: dict[str, JudgeOutcome] = {}

    for result in client.messages.batches.results(batch_id):
        item = by_id.get(result.custom_id)
        if item is None:
            continue
        kind = result.result.type
        if kind == "succeeded":
            outcomes[result.custom_id] = outcome_from_message(
                result.result.message,
                extracted=item.extracted,
                source=item.source,
                taxonomy=taxonomy,
                config=config,
                truncated=item.truncated,
            )
        elif kind == "errored":
            error = result.result.error
            outcomes[result.custom_id] = JudgeOutcome(
                record=None, error=f"batch error: {getattr(error, 'type', 'unknown')}"
            )
        else:
            outcomes[result.custom_id] = JudgeOutcome(record=None, error=f"batch result: {kind}")

    return outcomes
