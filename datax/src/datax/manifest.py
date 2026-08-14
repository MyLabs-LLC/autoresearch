"""The manifest: one JSONL record per .docx file.

Record layout is Nemotron-PII's schema plus four extension objects. The nine
Nemotron fields keep their exact names and value conventions so a manifest can be
concatenated with Nemotron-PII rows and consumed by the same training code:

    uid, domain, document_type, document_description, document_format, locale,
    text, spans, text_tagged

Everything datax adds lives under a namespaced object -- ``file``, ``source``,
``industry``, ``pii``, ``judge`` -- so the two vocabularies can never collide.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .spans import Span, spans_from_dicts, tag_text
from .taxonomy import OTHER_INDUSTRY, Taxonomy

MANIFEST_VERSION = "1.0.0"

DOCUMENT_FORMATS = ("structured", "unstructured")
LOCALES = ("us", "intl")
SENSITIVITIES = ("none", "low", "medium", "high", "critical")


class ManifestError(ValueError):
    pass


@dataclass
class FileInfo:
    path: str
    sha256: str
    size_bytes: int
    word_count: int
    paragraph_count: int = 0
    table_count: int = 0
    has_headers_or_footers: bool = False


@dataclass
class SourceInfo:
    provider: str
    """Which fetcher produced the file: ``nemotron``, ``web``, or ``hf``."""
    reference: str
    """Provider-specific origin: a dataset row uid, a URL, or a repo path."""
    license: str = "unknown"
    retrieved_at: str = ""
    synthetic: bool = False
    """True when the document's content was generated rather than collected. Synthetic
    documents carry no real personal data; collected ones may."""


@dataclass
class IndustryLabel:
    id: str
    category: str | None = None
    subcategory: str | None = None
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class PiiLabels:
    has_pii: bool = False
    labels: list[str] = field(default_factory=list)
    count_by_label: dict[str, int] = field(default_factory=dict)
    max_sensitivity: str = "none"
    contains_phi: bool = False
    contains_special_category: bool = False
    unresolved_evidence: list[dict] = field(default_factory=list)


@dataclass
class JudgeInfo:
    model: str = ""
    industry_taxonomy_version: str = ""
    pii_taxonomy_version: str = ""
    effort: str = ""
    judged_at: str = ""
    label_source: str = "llm_judge"
    """``llm_judge`` or ``gold`` -- gold records come from Nemotron-PII's own
    annotations and are what the judge is scored against."""
    refusal: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class Record:
    # -- Nemotron-PII compatible fields --
    uid: str
    domain: str
    document_type: str
    document_description: str
    document_format: str
    locale: str
    text: str
    spans: list[dict]
    text_tagged: str
    # -- datax extensions --
    file: FileInfo
    source: SourceInfo
    industry: IndustryLabel
    pii: PiiLabels
    judge: JudgeInfo
    manifest_version: str = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        # sort_keys keeps the file diffable and byte-stable across runs.
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Record":
        try:
            return cls(
                uid=raw["uid"],
                domain=raw["domain"],
                document_type=raw["document_type"],
                document_description=raw.get("document_description", ""),
                document_format=raw["document_format"],
                locale=raw["locale"],
                text=raw["text"],
                spans=list(raw.get("spans", [])),
                text_tagged=raw.get("text_tagged", ""),
                file=FileInfo(**raw["file"]),
                source=SourceInfo(**raw["source"]),
                industry=IndustryLabel(**raw["industry"]),
                pii=PiiLabels(**raw["pii"]),
                judge=JudgeInfo(**raw["judge"]),
                manifest_version=raw.get("manifest_version", MANIFEST_VERSION),
            )
        except KeyError as exc:
            raise ManifestError(f"record missing required field {exc.args[0]!r}") from None
        except TypeError as exc:
            raise ManifestError(f"record has unexpected fields: {exc}") from None


def new_uid() -> str:
    return str(uuid.uuid4())


def build_record(
    *,
    taxonomy: Taxonomy,
    uid: str,
    text: str,
    file: FileInfo,
    source: SourceInfo,
    industry: IndustryLabel,
    spans: list[Span],
    document_type: str,
    document_description: str = "",
    document_format: str = "unstructured",
    locale: str = "us",
    judge: JudgeInfo | None = None,
    unresolved_evidence: list[dict] | None = None,
) -> Record:
    """Assemble a record, deriving every field that can be derived.

    ``pii``, ``text_tagged`` and ``domain`` are computed from the taxonomy and the
    resolved spans rather than taken on trust, so a manifest is internally consistent
    by construction.
    """
    label_ids = []
    for span in spans:
        if span.label not in label_ids:
            label_ids.append(span.label)

    counts: dict[str, int] = {}
    for span in spans:
        counts[span.label] = counts.get(span.label, 0) + 1

    pii = PiiLabels(
        has_pii=bool(label_ids),
        labels=sorted(label_ids),
        count_by_label=dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        max_sensitivity=taxonomy.max_sensitivity(label_ids),
        contains_phi=any(taxonomy.pii(lid).phi for lid in label_ids),
        contains_special_category=any(taxonomy.pii(lid).special_category for lid in label_ids),
        unresolved_evidence=unresolved_evidence or [],
    )

    # `domain` carries the industry's display label so the field lines up with
    # Nemotron-PII's own vocabulary ("Healthcare", "Finance", "Government").
    domain = (
        taxonomy.industry(industry.id).label if industry.id != OTHER_INDUSTRY else "Other"
    )

    return Record(
        uid=uid,
        domain=domain,
        document_type=document_type,
        document_description=document_description,
        document_format=document_format,
        locale=locale,
        text=text,
        spans=[s.to_dict() for s in spans],
        text_tagged=tag_text(text, spans),
        file=file,
        source=source,
        industry=industry,
        pii=pii,
        judge=judge or JudgeInfo(),
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_manifest(path: str | Path, records: Iterable[Record], *, append: bool = False) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("a" if append else "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json() + "\n")
            count += 1
    return count


def read_manifest(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"line {lineno}: invalid JSON: {exc}") from None
            yield Record.from_dict(raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_record(record: Record, taxonomy: Taxonomy) -> list[str]:
    """Return a list of problems with a record; empty means valid.

    This is the check that catches the failure mode that matters most in a labelled
    dataset: offsets that no longer point at the text they claim to.
    """
    problems: list[str] = []

    if record.document_format not in DOCUMENT_FORMATS:
        problems.append(f"document_format {record.document_format!r} not in {DOCUMENT_FORMATS}")
    if record.locale not in LOCALES:
        problems.append(f"locale {record.locale!r} not in {LOCALES}")
    if not record.text:
        problems.append("text is empty")

    # Industry / subcategory must exist in the taxonomy.
    if record.industry.id == OTHER_INDUSTRY:
        if record.industry.subcategory:
            problems.append("industry 'other' must not carry a subcategory")
    elif record.industry.id not in [i.id for i in taxonomy.industries]:
        problems.append(f"unknown industry {record.industry.id!r}")
    elif record.industry.subcategory:
        sub = taxonomy.find_subcategory(record.industry.id, record.industry.subcategory)
        if sub is None:
            problems.append(
                f"subcategory {record.industry.subcategory!r} does not exist under "
                f"industry {record.industry.id!r}"
            )
        elif record.industry.category and sub.category != record.industry.category:
            problems.append(
                f"subcategory {record.industry.subcategory!r} belongs to category "
                f"{sub.category!r}, not {record.industry.category!r}"
            )

    if not 0.0 <= record.industry.confidence <= 1.0:
        problems.append(f"industry.confidence {record.industry.confidence} outside [0, 1]")

    # Spans: label known, offsets in range, surface form matching the text.
    known = set(taxonomy.pii_ids)
    spans = spans_from_dicts(record.spans)
    for span in spans:
        if span.label not in known:
            problems.append(f"span label {span.label!r} not in PII taxonomy")
        if not 0 <= span.start < span.end <= len(record.text):
            problems.append(f"span [{span.start}, {span.end}) outside text of length {len(record.text)}")
            continue
        actual = record.text[span.start : span.end]
        if span.text and span.text != actual:
            problems.append(
                f"span [{span.start}, {span.end}) text mismatch: "
                f"recorded {span.text!r} but document has {actual!r}"
            )

    ordered = sorted(spans, key=lambda s: s.start)
    for left, right in zip(ordered, ordered[1:]):
        if left.overlaps(right):
            problems.append(f"spans overlap: {left.to_dict()} and {right.to_dict()}")

    # Derived PII summary must agree with the spans.
    span_labels = sorted({s.label for s in spans})
    if record.pii.labels != span_labels:
        problems.append(f"pii.labels {record.pii.labels} disagrees with span labels {span_labels}")
    if record.pii.has_pii != bool(spans):
        problems.append(f"pii.has_pii={record.pii.has_pii} but record has {len(spans)} spans")
    if record.pii.max_sensitivity not in SENSITIVITIES:
        problems.append(f"pii.max_sensitivity {record.pii.max_sensitivity!r} not in {SENSITIVITIES}")

    # text_tagged must be reproducible from text + spans.
    if record.text_tagged:
        try:
            expected = tag_text(record.text, spans)
        except ValueError as exc:
            problems.append(f"text_tagged cannot be rebuilt: {exc}")
        else:
            if expected != record.text_tagged:
                problems.append("text_tagged does not match text + spans")

    return problems


@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    problems: dict[str, list[str]] = field(default_factory=dict)
    duplicate_uids: list[str] = field(default_factory=list)
    duplicate_sha256: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.duplicate_uids and not self.duplicate_sha256


def validate_manifest(path: str | Path, taxonomy: Taxonomy) -> ValidationReport:
    report = ValidationReport()
    seen_uids: set[str] = set()
    seen_hashes: dict[str, str] = {}

    for record in read_manifest(path):
        report.total += 1
        if record.uid in seen_uids:
            report.duplicate_uids.append(record.uid)
        seen_uids.add(record.uid)

        # Identical bytes labelled twice inflates every metric computed downstream.
        if record.file.sha256 in seen_hashes:
            report.duplicate_sha256.append(record.file.sha256)
        else:
            seen_hashes[record.file.sha256] = record.uid

        issues = validate_record(record, taxonomy)
        if issues:
            report.problems[record.uid] = issues
        else:
            report.valid += 1

    return report
