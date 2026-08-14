"""Real-world .docx documents, sourced via the superdoc-dev/docx-corpus index.

That dataset is a 736K-row index of publicly reachable .docx URLs, each pre-classified
by document type and topic. Its topic vocabulary happens to include exactly the three
industries this dataset targets -- ``healthcare`` (64.5K), ``finance`` (49.2K) and
``government`` (244.9K) -- which makes it a usable discovery layer: filter the index,
then download the real documents from the URLs it records.

Why this rather than a hardcoded list of agency URLs: guessed .gov paths rot
immediately and many agencies block non-browser clients outright. An index that is
itself versioned and downloadable is reproducible; a handwritten URL list is not.

The corpus topic is treated as a **sampling filter, not a label**. Documents still go
through the judge like everything else, and the index's own topic is retained only as
a weak reference on the record so the two can be compared.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import FetchReport
from .web import DownloadError, download_docx

DATASET = "superdoc-dev/docx-corpus"
LICENSE = "odc-by"
PARQUET_PATH = "data/train-00000-of-00001.parquet"
RESOLVE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{PARQUET_PATH}"

# Corpus topic -> datax industry. The corpus has six other topics (education, legal,
# environment, nonprofit, technology, general) which are deliberately not mapped.
TOPIC_TO_INDUSTRY = {
    "healthcare": "healthcare",
    "finance": "finance",
    "government": "government",
}

COLUMNS = ["id", "filename", "type", "topic", "language", "word_count", "confidence", "url"]


@dataclass
class CorpusEntry:
    id: str
    filename: str
    type: str
    topic: str
    language: str
    word_count: int
    confidence: float
    url: str

    @property
    def industry(self) -> str:
        return TOPIC_TO_INDUSTRY[self.topic]


def download_index(cache_dir: str | Path = "datax/data/cache") -> Path:
    """Fetch the corpus index parquet (about 64 MB), cached on disk."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "docx-corpus-index.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=DATASET,
            filename=PARQUET_PATH,
            repo_type="dataset",
            cache_dir=str(cache / "hf"),
            token=os.environ.get("HF_TOKEN"),
        )
        return Path(path)
    except ImportError:
        tmp = target.with_suffix(".partial")
        with urllib.request.urlopen(RESOLVE_URL, timeout=600) as response, tmp.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        tmp.replace(target)
        return target


def select(
    index_path: str | Path,
    *,
    per_industry: int = 25,
    language: str | None = "en",
    min_word_count: int = 150,
    min_confidence: float = 0.75,
    doc_types: set[str] | None = None,
) -> list[CorpusEntry]:
    """Choose a balanced sample of index entries.

    The defaults filter out the two things that waste judge calls: documents too short
    to classify meaningfully, and entries the corpus' own classifier was unsure about.
    Selection is deterministic (index order), so a run is reproducible without a seed.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(index_path, columns=COLUMNS)
    columns = {name: table.column(name).to_pylist() for name in COLUMNS}

    buckets: dict[str, list[CorpusEntry]] = {ind: [] for ind in set(TOPIC_TO_INDUSTRY.values())}
    for i in range(table.num_rows):
        topic = columns["topic"][i]
        industry = TOPIC_TO_INDUSTRY.get(topic)
        if industry is None or len(buckets[industry]) >= per_industry:
            continue
        if language is not None and columns["language"][i] != language:
            continue
        if (columns["word_count"][i] or 0) < min_word_count:
            continue
        if (columns["confidence"][i] or 0.0) < min_confidence:
            continue
        if doc_types is not None and columns["type"][i] not in doc_types:
            continue
        buckets[industry].append(
            CorpusEntry(
                id=columns["id"][i],
                filename=columns["filename"][i] or "",
                type=columns["type"][i],
                topic=topic,
                language=columns["language"][i],
                word_count=int(columns["word_count"][i] or 0),
                confidence=float(columns["confidence"][i] or 0.0),
                url=columns["url"][i],
            )
        )
        if all(len(b) >= per_industry for b in buckets.values()):
            break

    return [entry for bucket in buckets.values() for entry in bucket]


def fetch(
    out_dir: str | Path,
    *,
    per_industry: int = 25,
    cache_dir: str | Path = "datax/data/cache",
    language: str | None = "en",
    min_word_count: int = 150,
    min_confidence: float = 0.75,
    delay_seconds: float = 0.25,
) -> tuple[FetchReport, dict[str, CorpusEntry]]:
    """Download a balanced sample of real .docx documents.

    Returns the report plus a map from written file path to the index entry it came
    from, so the caller can attach provenance when judging.
    """
    report = FetchReport(provider="docxcorpus")
    report.notes.append(f"index={DATASET} license={LICENSE}")

    index_path = download_index(cache_dir)
    entries = select(
        index_path,
        per_industry=per_industry,
        language=language,
        min_word_count=min_word_count,
        min_confidence=min_confidence,
    )
    report.requested = len(entries)

    out = Path(out_dir)
    provenance: dict[str, CorpusEntry] = {}
    seen_hashes: set[str] = set()

    import time

    for position, entry in enumerate(entries):
        if position and delay_seconds:
            time.sleep(delay_seconds)
        target_dir = out / entry.industry
        try:
            download = download_docx(entry.url, target_dir, filename=f"{entry.id}.docx")
        except DownloadError as exc:
            report.fail(entry.url, str(exc))
            continue

        if download.sha256 in seen_hashes:
            download.path.unlink(missing_ok=True)
            report.skipped += 1
            continue
        seen_hashes.add(download.sha256)

        provenance[str(download.path)] = entry
        report.written += 1
        report.files.append(str(download.path))

    return report, provenance
