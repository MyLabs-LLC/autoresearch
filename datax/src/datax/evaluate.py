"""Scoring the judge against gold labels.

The Nemotron-derived documents carry their own span annotations, so the same pipeline
that builds the dataset can measure how good its labels are. Without this the manifest
is a pile of assertions; with it, every release ships a number.

Two levels of PII scoring are reported, because they answer different questions:

*document level*
    Did the judge notice that this document contains an ``ssn`` at all? This is what
    matters for routing, DLP triage, and file classification.
*span level*
    Did it find the exact characters? Stricter, and the right measure for training an
    NER model or a redactor.

Gold and predicted records are joined on ``file.sha256`` -- the judged record gets a
fresh ``uid``, but it describes the same bytes on disk.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .manifest import Record, read_manifest
from .spans import spans_from_dicts


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvalReport:
    matched: int = 0
    gold_only: int = 0
    predicted_only: int = 0

    industry_correct: int = 0
    industry_total: int = 0
    industry_confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    subcategory_correct: int = 0
    subcategory_total: int = 0

    doc_level: PRF = field(default_factory=PRF)
    doc_level_by_label: dict[str, PRF] = field(default_factory=dict)
    span_level: PRF = field(default_factory=PRF)
    span_level_by_label: dict[str, PRF] = field(default_factory=dict)

    unresolved_evidence: int = 0
    refusals: int = 0

    @property
    def industry_accuracy(self) -> float:
        return self.industry_correct / self.industry_total if self.industry_total else 0.0

    @property
    def subcategory_accuracy(self) -> float:
        return self.subcategory_correct / self.subcategory_total if self.subcategory_total else 0.0

    def macro_doc_f1(self) -> float:
        """Unweighted mean F1 across labels that actually occur in gold.

        Reported alongside micro because rare-but-critical labels (``ssn``, ``pin``)
        are invisible in a micro average dominated by ``date`` and ``company_name``.
        """
        scored = [prf for prf in self.doc_level_by_label.values() if prf.support]
        return sum(p.f1 for p in scored) / len(scored) if scored else 0.0

    def to_dict(self) -> dict:
        return {
            "documents": {
                "matched": self.matched,
                "gold_only": self.gold_only,
                "predicted_only": self.predicted_only,
            },
            "industry": {
                "accuracy": round(self.industry_accuracy, 4),
                "correct": self.industry_correct,
                "total": self.industry_total,
                "confusion": self.industry_confusion,
            },
            "subcategory": {
                "accuracy": round(self.subcategory_accuracy, 4),
                "correct": self.subcategory_correct,
                "total": self.subcategory_total,
            },
            "pii_document_level": {
                "micro": self.doc_level.to_dict(),
                "macro_f1": round(self.macro_doc_f1(), 4),
                "by_label": {k: v.to_dict() for k, v in sorted(self.doc_level_by_label.items())},
            },
            "pii_span_level": {
                "micro": self.span_level.to_dict(),
                "by_label": {k: v.to_dict() for k, v in sorted(self.span_level_by_label.items())},
            },
            "unresolved_evidence": self.unresolved_evidence,
        }

    def summary(self) -> str:
        lines = [
            f"matched {self.matched} document(s) "
            f"(gold-only {self.gold_only}, predicted-only {self.predicted_only})",
            f"industry accuracy      {self.industry_accuracy:6.1%}  "
            f"({self.industry_correct}/{self.industry_total})",
            f"subcategory accuracy   {self.subcategory_accuracy:6.1%}  "
            f"({self.subcategory_correct}/{self.subcategory_total})",
            f"PII document-level     P {self.doc_level.precision:5.1%}  "
            f"R {self.doc_level.recall:5.1%}  F1 {self.doc_level.f1:5.1%}  "
            f"(macro F1 {self.macro_doc_f1():.1%})",
            f"PII span-level         P {self.span_level.precision:5.1%}  "
            f"R {self.span_level.recall:5.1%}  F1 {self.span_level.f1:5.1%}",
        ]
        weakest = sorted(
            (prf for prf in self.doc_level_by_label.items() if prf[1].support >= 3),
            key=lambda kv: kv[1].f1,
        )[:5]
        if weakest:
            lines.append("weakest labels (document level, support >= 3):")
            for label, prf in weakest:
                lines.append(
                    f"  {label:34s} F1 {prf.f1:5.1%}  P {prf.precision:5.1%}  "
                    f"R {prf.recall:5.1%}  n={prf.support}"
                )
        return "\n".join(lines)


def _bucket(table: dict[str, PRF], key: str) -> PRF:
    return table.setdefault(key, PRF())


def compare(gold: Iterable[Record], predicted: Iterable[Record]) -> EvalReport:
    """Score predicted records against gold records, joined on file content hash."""
    report = EvalReport()
    gold_by_hash = {r.file.sha256: r for r in gold}
    pred_by_hash = {r.file.sha256: r for r in predicted}

    report.gold_only = len(set(gold_by_hash) - set(pred_by_hash))
    report.predicted_only = len(set(pred_by_hash) - set(gold_by_hash))

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sha, gold_record in gold_by_hash.items():
        pred = pred_by_hash.get(sha)
        if pred is None:
            continue
        report.matched += 1

        # -- industry --
        report.industry_total += 1
        confusion[gold_record.industry.id][pred.industry.id] += 1
        if gold_record.industry.id == pred.industry.id:
            report.industry_correct += 1

        # Only score the subcategory where gold actually has one: Nemotron document
        # types outside the crosswalk have no gold leaf, and counting those as errors
        # would measure crosswalk coverage rather than judge quality.
        if gold_record.industry.subcategory:
            report.subcategory_total += 1
            if gold_record.industry.subcategory == pred.industry.subcategory:
                report.subcategory_correct += 1

        # -- PII, document level (label present / absent) --
        gold_labels = set(gold_record.pii.labels)
        pred_labels = set(pred.pii.labels)
        for label in gold_labels | pred_labels:
            bucket = _bucket(report.doc_level_by_label, label)
            if label in gold_labels and label in pred_labels:
                bucket.tp += 1
                report.doc_level.tp += 1
            elif label in pred_labels:
                bucket.fp += 1
                report.doc_level.fp += 1
            else:
                bucket.fn += 1
                report.doc_level.fn += 1

        # -- PII, span level (exact offsets and label) --
        gold_spans = {(s.start, s.end, s.label) for s in spans_from_dicts(gold_record.spans)}
        pred_spans = {(s.start, s.end, s.label) for s in spans_from_dicts(pred.spans)}
        for span in gold_spans | pred_spans:
            bucket = _bucket(report.span_level_by_label, span[2])
            if span in gold_spans and span in pred_spans:
                bucket.tp += 1
                report.span_level.tp += 1
            elif span in pred_spans:
                bucket.fp += 1
                report.span_level.fp += 1
            else:
                bucket.fn += 1
                report.span_level.fn += 1

        report.unresolved_evidence += len(
            [e for e in pred.pii.unresolved_evidence if e.get("reason") == "not_found_in_text"]
        )

    report.industry_confusion = {k: dict(v) for k, v in confusion.items()}
    return report


def evaluate_files(gold_path: str | Path, predicted_path: str | Path) -> EvalReport:
    return compare(read_manifest(gold_path), read_manifest(predicted_path))
