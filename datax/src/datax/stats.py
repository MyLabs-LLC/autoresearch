"""Dataset-level statistics, for the dataset card and for spotting skew early."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .manifest import Record


@dataclass
class DatasetStats:
    documents: int = 0
    total_words: int = 0
    total_bytes: int = 0
    by_industry: dict[str, int] = field(default_factory=dict)
    by_subcategory: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_format: dict[str, int] = field(default_factory=dict)
    by_locale: dict[str, int] = field(default_factory=dict)
    by_label_source: dict[str, int] = field(default_factory=dict)
    pii_label_document_counts: dict[str, int] = field(default_factory=dict)
    pii_label_span_counts: dict[str, int] = field(default_factory=dict)
    documents_with_pii: int = 0
    documents_with_phi: int = 0
    documents_with_special_category: int = 0
    by_max_sensitivity: dict[str, int] = field(default_factory=dict)
    labels_never_seen: list[str] = field(default_factory=list)
    subcategories_never_seen: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "total_words": self.total_words,
            "total_bytes": self.total_bytes,
            "mean_words": round(self.total_words / self.documents, 1) if self.documents else 0,
            "by_industry": self.by_industry,
            "by_subcategory": self.by_subcategory,
            "by_source": self.by_source,
            "by_format": self.by_format,
            "by_locale": self.by_locale,
            "by_label_source": self.by_label_source,
            "pii": {
                "documents_with_pii": self.documents_with_pii,
                "documents_with_phi": self.documents_with_phi,
                "documents_with_special_category": self.documents_with_special_category,
                "by_max_sensitivity": self.by_max_sensitivity,
                "label_document_counts": self.pii_label_document_counts,
                "label_span_counts": self.pii_label_span_counts,
            },
            "coverage_gaps": {
                "pii_labels_never_seen": self.labels_never_seen,
                "subcategories_never_seen": self.subcategories_never_seen,
            },
        }

    def summary(self) -> str:
        lines = [
            f"{self.documents} document(s), {self.total_words:,} words, "
            f"{self.total_bytes / 1e6:.1f} MB",
            "by industry: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.by_industry.items())),
            "by source:   " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_source.items())),
            f"documents with PII: {self.documents_with_pii} "
            f"(PHI {self.documents_with_phi}, special-category "
            f"{self.documents_with_special_category})",
            f"distinct PII labels observed: {len(self.pii_label_document_counts)}",
            f"distinct subcategories observed: {len(self.by_subcategory)}",
        ]
        if self.labels_never_seen:
            lines.append(
                f"PII labels with zero occurrences ({len(self.labels_never_seen)}): "
                + ", ".join(self.labels_never_seen[:15])
                + (" ..." if len(self.labels_never_seen) > 15 else "")
            )
        return "\n".join(lines)


def compute(records: Iterable[Record], taxonomy=None) -> DatasetStats:
    """Aggregate a manifest.

    When a taxonomy is supplied, coverage gaps are reported too: which labels and
    subcategories the dataset never exercises. That is the number that tells you where
    the next batch of documents needs to come from.
    """
    stats = DatasetStats()
    industry = Counter()
    subcategory = Counter()
    source = Counter()
    fmt = Counter()
    locale = Counter()
    label_source = Counter()
    sensitivity = Counter()
    doc_labels = Counter()
    span_labels = Counter()

    for record in records:
        stats.documents += 1
        stats.total_words += record.file.word_count
        stats.total_bytes += record.file.size_bytes
        industry[record.industry.id] += 1
        if record.industry.subcategory:
            subcategory[f"{record.industry.id}/{record.industry.subcategory}"] += 1
        source[record.source.provider] += 1
        fmt[record.document_format] += 1
        locale[record.locale] += 1
        label_source[record.judge.label_source] += 1
        sensitivity[record.pii.max_sensitivity] += 1

        if record.pii.has_pii:
            stats.documents_with_pii += 1
        if record.pii.contains_phi:
            stats.documents_with_phi += 1
        if record.pii.contains_special_category:
            stats.documents_with_special_category += 1
        for label in record.pii.labels:
            doc_labels[label] += 1
        for label, count in record.pii.count_by_label.items():
            span_labels[label] += count

    stats.by_industry = dict(sorted(industry.items()))
    stats.by_subcategory = dict(sorted(subcategory.items(), key=lambda kv: (-kv[1], kv[0])))
    stats.by_source = dict(sorted(source.items()))
    stats.by_format = dict(sorted(fmt.items()))
    stats.by_locale = dict(sorted(locale.items()))
    stats.by_label_source = dict(sorted(label_source.items()))
    stats.by_max_sensitivity = dict(sorted(sensitivity.items()))
    stats.pii_label_document_counts = dict(sorted(doc_labels.items(), key=lambda kv: (-kv[1], kv[0])))
    stats.pii_label_span_counts = dict(sorted(span_labels.items(), key=lambda kv: (-kv[1], kv[0])))

    if taxonomy is not None:
        stats.labels_never_seen = sorted(set(taxonomy.pii_ids) - set(doc_labels))
        all_subs = {f"{s.industry}/{s.id}" for s in taxonomy.subcategories()}
        stats.subcategories_never_seen = sorted(all_subs - set(subcategory))

    return stats


def write_json(path: str | Path, stats: DatasetStats) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
