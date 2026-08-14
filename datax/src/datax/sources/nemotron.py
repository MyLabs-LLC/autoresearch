"""Render nvidia/Nemotron-PII rows into .docx files with gold labels.

This is the backbone of the dataset. Nemotron-PII rows are synthetic, CC-BY-4.0, and
already carry span-level PII annotations, so rendering them into Word documents yields
a labelled .docx corpus with **no real personal data in it** -- which is the only
responsible way to build a training set full of SSNs and medical record numbers.

The gold records this produces are also what the judge is scored against
(:mod:`datax.evaluate`), so the pipeline can report its own accuracy instead of
asserting it.

Round-trip integrity is enforced, not assumed. After writing each document the text is
re-extracted and compared with the source string; when they differ the spans are
realigned against the extracted text and any span that cannot be relocated is dropped
with a note. A record therefore never claims an offset that does not hold.
"""

from __future__ import annotations

import ast
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from ..docxio import write_text_docx
from ..extract import extract_docx
from ..manifest import FileInfo, IndustryLabel, JudgeInfo, Record, SourceInfo, build_record
from ..spans import Span, dedupe_overlaps, realign
from ..taxonomy import Taxonomy, default_taxonomy
from . import FetchReport

DATASET = "nvidia/Nemotron-PII"
LICENSE = "cc-by-4.0"
PARQUET_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# Only the three domains whose industry mapping is unambiguous. Nemotron's `Insurance`
# and `Pharmaceuticals` domains straddle healthcare and finance, and a gold label that
# needs a judgement call is not gold.
TARGET_DOMAINS = {
    "Healthcare": "healthcare",
    "Finance": "finance",
    "Government": "government",
}

NEEDED_COLUMNS = [
    "uid",
    "domain",
    "document_type",
    "document_description",
    "document_format",
    "locale",
    "text",
    "spans",
]


@dataclass
class NemotronRow:
    uid: str
    domain: str
    document_type: str
    document_description: str
    document_format: str
    locale: str
    text: str
    spans: list[dict]

    @property
    def key(self) -> str:
        """A genuinely unique key for the row.

        Nemotron's ``uid`` is **not** unique: the train split has 100,000 rows over
        50,000 uids, because each document appears once per locale and the two
        variants share a uid (their text differs -- only one pair of the 50,000 is
        identical). Keying on uid alone would give the manifest duplicate uids and,
        worse, make the two variants overwrite each other on disk.
        """
        return f"{self.uid}-{self.locale}"


def _parse_spans(raw: str | list) -> list[dict]:
    """Nemotron stores `spans` as a Python-repr string, not JSON (single quotes)."""
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def download_parquet(split: str, cache_dir: str | Path) -> Path:
    """Fetch a split's parquet, preferring huggingface_hub (auth, caching, resume)."""
    if split not in PARQUET_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(PARQUET_FILES)}")
    rel = PARQUET_FILES[split]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{split}.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=DATASET,
            filename=rel,
            repo_type="dataset",
            cache_dir=str(cache / "hf"),
            token=os.environ.get("HF_TOKEN"),
        )
        return Path(path)
    except ImportError:
        url = RESOLVE_URL.format(repo=DATASET, path=rel)
        tmp = target.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        tmp.replace(target)
        return target


def iter_rows(
    parquet_path: str | Path,
    *,
    domains: Sequence[str] | None = None,
    limit: int | None = None,
) -> Iterator[NemotronRow]:
    """Stream matching rows. Reads only the columns needed, so a 150 MB file does not
    become a 1 GB working set."""
    import pyarrow.parquet as pq

    wanted = set(domains or TARGET_DOMAINS)
    table = pq.read_table(parquet_path, columns=NEEDED_COLUMNS)
    columns = {name: table.column(name).to_pylist() for name in NEEDED_COLUMNS}

    emitted = 0
    for index in range(table.num_rows):
        if columns["domain"][index] not in wanted:
            continue
        yield NemotronRow(
            uid=columns["uid"][index],
            domain=columns["domain"][index],
            document_type=columns["document_type"][index],
            document_description=columns["document_description"][index] or "",
            document_format=columns["document_format"][index],
            locale=columns["locale"][index],
            text=columns["text"][index],
            spans=_parse_spans(columns["spans"][index]),
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def _balanced(rows: Iterator[NemotronRow], per_domain: int) -> list[NemotronRow]:
    """Take up to ``per_domain`` rows from each target domain.

    Without this the mix follows Nemotron's own domain frequencies, and a classifier
    trained on it learns the prior instead of the document.
    """
    buckets: dict[str, list[NemotronRow]] = {domain: [] for domain in TARGET_DOMAINS}
    remaining = set(buckets)
    for row in rows:
        bucket = buckets.get(row.domain)
        if bucket is None or len(bucket) >= per_domain:
            if row.domain in remaining and len(buckets[row.domain]) >= per_domain:
                remaining.discard(row.domain)
                if not remaining:
                    break
            continue
        bucket.append(row)
    return [row for bucket in buckets.values() for row in bucket]


def gold_spans(row: NemotronRow, extracted_text: str) -> tuple[list[Span], list[dict]]:
    """Convert a row's annotations into verified spans against the extracted text.

    Nemotron's own ``spans`` are not perfectly self-consistent: measured over the
    train split, **1.53% of spans (12,594 of 825,456) have ``span['text']`` that does
    not equal ``text[start:end]``**. Two causes, both benign:

    * the annotation carries a normalised value rather than the surface form
      (``'spanish'`` where the document says ``'Spanish'``);
    * numeric labels (``age``, ``cvv``, ``pin``) store an ``int``, so ``44 != '44'``.

    In both cases the **offsets are right** and only the text field differs, so those
    spans are repaired to the actual substring rather than discarded -- throwing them
    away would silently delete real gold labels. A span is only dropped when its
    offsets do not fit the text at all.
    """
    notes: list[dict] = []
    raw: list[Span] = []

    for entry in row.spans:
        try:
            start, end = int(entry["start"]), int(entry["end"])
            label = entry["label"]
        except (KeyError, TypeError, ValueError):
            notes.append({"reason": "malformed_gold_span", "span": str(entry)})
            continue
        # Coerce: numeric labels arrive as int, not str.
        raw.append(Span(start=start, end=end, text=str(entry.get("text", "")), label=label))

    if extracted_text == row.text:
        candidates = raw
    else:
        candidates, lost = realign(row.text, extracted_text, raw)
        notes.extend(
            {"label": s.label, "text": s.text, "reason": "lost_in_docx_round_trip"} for s in lost
        )

    repaired: list[Span] = []
    for span in candidates:
        if not 0 <= span.start < span.end <= len(extracted_text):
            notes.append(
                {
                    "label": span.label,
                    "text": span.text,
                    "reason": "gold_span_offsets_out_of_range",
                }
            )
            continue
        actual = extracted_text[span.start : span.end]
        if actual != span.text:
            notes.append(
                {
                    "label": span.label,
                    "claimed_text": span.text,
                    "actual_text": actual,
                    "reason": "gold_span_text_normalised",
                }
            )
        repaired.append(Span(start=span.start, end=span.end, text=actual, label=span.label))

    # text_tagged cannot render overlapping spans, so resolve them here.
    final, dropped = dedupe_overlaps(repaired)
    notes.extend(
        {"reason": "gold_span_overlap_dropped", **entry} for entry in dropped
    )
    return final, notes


def _gold_record(
    row: NemotronRow,
    extracted,
    spans: list[Span],
    taxonomy: Taxonomy,
    notes: list[dict],
    sanitized: bool,
) -> Record:
    crosswalk = taxonomy.nemotron_crosswalk()
    sub = crosswalk.get(row.document_type.casefold())
    industry_id = TARGET_DOMAINS[row.domain]

    notes = list(notes)
    if sanitized:
        notes.append({"reason": "control_characters_stripped_before_write"})
    if sub is None:
        notes.append(
            {
                "reason": "document_type_not_in_industry_crosswalk",
                "document_type": row.document_type,
            }
        )

    return build_record(
        taxonomy=taxonomy,
        uid=row.key,
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
        source=SourceInfo(
            provider="nemotron",
            reference=f"{DATASET}#{row.uid}@{row.locale}",
            license=LICENSE,
            synthetic=True,
        ),
        industry=IndustryLabel(
            id=industry_id,
            category=sub.category if sub else None,
            subcategory=sub.id if sub else None,
            # Gold industry comes straight from the dataset's own `domain` field, so
            # it is certain; the subcategory is a crosswalk and is not asserted here.
            confidence=1.0,
            rationale=f"Nemotron-PII domain {row.domain!r}",
        ),
        spans=spans,
        document_type=row.document_type,
        document_description=row.document_description,
        document_format=row.document_format,
        locale=row.locale,
        judge=JudgeInfo(
            model="",
            industry_taxonomy_version=taxonomy.industry_version,
            pii_taxonomy_version=taxonomy.pii_version,
            label_source="gold",
        ),
        unresolved_evidence=notes,
    )


def fetch(
    out_dir: str | Path,
    *,
    split: str = "train",
    per_domain: int = 50,
    cache_dir: str | Path = "datax/data/cache",
    layout: str = "rich",
    taxonomy: Taxonomy | None = None,
) -> tuple[FetchReport, list[Record]]:
    """Render a balanced sample of Nemotron-PII rows into .docx files with gold labels."""
    taxonomy = taxonomy or default_taxonomy()
    report = FetchReport(provider="nemotron")
    records: list[Record] = []

    parquet_path = download_parquet(split, cache_dir)
    rows = _balanced(iter_rows(parquet_path), per_domain)
    report.requested = len(rows)
    report.notes.append(f"source={DATASET} split={split} license={LICENSE}")

    out = Path(out_dir)
    for row in rows:
        industry_id = TARGET_DOMAINS[row.domain]
        target = out / industry_id / f"{row.key}.docx"
        try:
            written = write_text_docx(target, row.text, layout=layout)
            extracted = extract_docx(target)
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
            report.fail(row.key, f"{type(exc).__name__}: {exc}")
            continue

        spans, notes = gold_spans(row, extracted.text)

        try:
            records.append(
                _gold_record(row, extracted, spans, taxonomy, notes, written.sanitized)
            )
        except Exception as exc:  # noqa: BLE001
            report.fail(row.key, f"record build failed: {type(exc).__name__}: {exc}")
            continue

        report.written += 1
        report.files.append(str(target))
        dropped = [n for n in notes if n["reason"] != "gold_span_text_normalised"]
        if dropped:
            report.notes.append(f"{row.key}: {len(dropped)} gold span(s) dropped")

    return report, records
