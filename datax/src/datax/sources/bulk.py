"""Bulk download of the docx-corpus into a ``<topic>/<type>/<id>.docx`` tree.

Built for a job that runs for hours over tens of thousands of files, so the things
that matter are the boring ones:

* **Resumable.** Files already on disk are skipped, so an interrupted run costs
  nothing to restart. This is the single most important property at this scale.
* **Bounded and retried.** Concurrency is capped, transient failures back off and
  retry, permanent ones (a URL that serves HTML, not a Word file) are recorded and
  never retried.
* **Every failure is written down.** Over 90,000 requests some will fail; they go to
  a JSONL file that can be fed straight back in with ``--retry-failed``.
* **Metadata travels with the files.** The folder tree encodes topic and type, but
  language, confidence, word count and the source URL would be lost -- so a sidecar
  ``index.jsonl`` records the full index row for every file written.

The directory layout is deliberately plain, so the tree is usable by
``torchvision``-style folder loaders and by ``find``::

    <out>/<topic>/<type>/<sha256-id>.docx
    <out>/index.jsonl        one JSON row per downloaded file
    <out>/failures.jsonl     one JSON row per failed URL
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .docxcorpus import COLUMNS, TOPIC_TO_INDUSTRY, CorpusEntry
from .web import USER_AGENT, DownloadError, download_docx

# The corpus' own documentation suggests parallel 4 for bulk transfer. 8 is a modest
# step up that still behaves; going much higher risks being rate limited and is not
# a neighbourly way to treat someone else's storage bill.
DEFAULT_WORKERS = 8
DEFAULT_RETRIES = 3
INDEX_FILE = "index.jsonl"
FAILURES_FILE = "failures.jsonl"

ALL_TOPICS = "all"


@dataclass
class BulkReport:
    requested: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    bytes_written: int = 0
    elapsed_seconds: float = 0.0
    failures_by_reason: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        rate = self.downloaded / self.elapsed_seconds if self.elapsed_seconds else 0.0
        lines = [
            f"downloaded {self.downloaded:,} of {self.requested:,} "
            f"({self.skipped_existing:,} already present, {self.failed:,} failed)",
            f"{self.bytes_written / 1e9:.2f} GB in {self.elapsed_seconds / 60:.1f} min "
            f"({rate:.1f} files/s)",
        ]
        if self.failures_by_reason:
            lines.append("failure reasons:")
            for reason, count in sorted(
                self.failures_by_reason.items(), key=lambda kv: -kv[1]
            )[:8]:
                lines.append(f"  {count:>6,}  {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_all(
    index_path: str | Path,
    *,
    topics: Sequence[str] | str = tuple(TOPIC_TO_INDUSTRY),
    language: str | None = "en",
    types: Sequence[str] | None = None,
    min_confidence: float = 0.0,
    min_words: int = 0,
    limit: int | None = None,
) -> list[CorpusEntry]:
    """Every index row matching the filters, in index order.

    Unlike :func:`datax.sources.docxcorpus.select`, this does not balance across
    topics -- a bulk mirror should reflect what the corpus actually contains, and
    balancing is a sampling decision to make later, from the downloaded tree.
    """
    import pyarrow.parquet as pq

    wanted_topics = None if topics == ALL_TOPICS else set(topics)
    wanted_types = set(types) if types else None

    table = pq.read_table(index_path, columns=COLUMNS)
    columns = {name: table.column(name).to_pylist() for name in COLUMNS}

    out: list[CorpusEntry] = []
    for i in range(table.num_rows):
        topic = columns["topic"][i]
        if wanted_topics is not None and topic not in wanted_topics:
            continue
        if language is not None and columns["language"][i] != language:
            continue
        if wanted_types is not None and columns["type"][i] not in wanted_types:
            continue
        if (columns["confidence"][i] or 0.0) < min_confidence:
            continue
        if (columns["word_count"][i] or 0) < min_words:
            continue
        out.append(
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
        if limit is not None and len(out) >= limit:
            break
    return out


def target_path(out_dir: Path, entry: CorpusEntry) -> Path:
    """``<out>/<topic>/<type>/<id>.docx``.

    The id is a hex digest, so it is always a safe filename and always unique -- the
    corpus' own ``filename`` column is not (``content``, ``index.php``, ``13462``).
    """
    return out_dir / entry.topic / entry.type / f"{entry.id}.docx"


def estimate_bytes(entries: Sequence[CorpusEntry], *, sample: int = 24, timeout: int = 20) -> int:
    """Estimate the download size from HEAD requests over an evenly-spaced sample.

    The index carries ``word_count`` but not file size, and .docx size correlates
    badly with word count (embedded images dominate). Sampling the real thing is
    cheap and far more honest than a guess.
    """
    if not entries:
        return 0
    step = max(1, len(entries) // sample)
    picks = entries[::step][:sample]
    sizes: list[int] = []
    for entry in picks:
        request = urllib.request.Request(
            entry.url, method="HEAD", headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length:
                    sizes.append(int(length))
        except Exception:  # noqa: BLE001 - estimation must never break the run
            continue
    if not sizes:
        return 0
    return int(sum(sizes) / len(sizes) * len(entries))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_one(
    entry: CorpusEntry,
    out_dir: Path,
    *,
    retries: int,
    timeout: int,
    verify_existing: bool,
) -> tuple[CorpusEntry, str, int, str]:
    """Returns ``(entry, status, bytes, detail)`` with status in
    ``downloaded`` / ``skipped`` / ``failed``."""
    target = target_path(out_dir, entry)

    if target.exists() and target.stat().st_size > 0:
        if not verify_existing:
            return entry, "skipped", 0, ""
        from ..extract import looks_like_docx

        if looks_like_docx(target):
            return entry, "skipped", 0, ""
        target.unlink(missing_ok=True)  # corrupt from a previous run; re-fetch

    delay = 1.0
    last = ""
    for attempt in range(retries):
        try:
            download = download_docx(
                entry.url, target.parent, filename=f"{entry.id}.docx", timeout=timeout
            )
            return entry, "downloaded", download.size_bytes, ""
        except DownloadError as exc:
            last = str(exc)
            if not exc.transient or attempt == retries - 1:
                break
            time.sleep(delay)
            delay *= 2  # exponential backoff; the host is someone else's infrastructure
    return entry, "failed", 0, last


def download_all(
    entries: Sequence[CorpusEntry],
    out_dir: str | Path,
    *,
    workers: int = DEFAULT_WORKERS,
    retries: int = DEFAULT_RETRIES,
    timeout: int = 30,
    verify_existing: bool = False,
    on_progress: Callable[[BulkReport], None] | None = None,
    progress_every: int = 250,
) -> BulkReport:
    """Download ``entries`` into a ``<topic>/<type>/`` tree, resumably."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = BulkReport(requested=len(entries))
    started = time.monotonic()

    index_fh = (out / INDEX_FILE).open("a", encoding="utf-8")
    failures_fh = (out / FAILURES_FILE).open("a", encoding="utf-8")
    write_lock = threading.Lock()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _download_one,
                    entry,
                    out,
                    retries=retries,
                    timeout=timeout,
                    verify_existing=verify_existing,
                )
                for entry in entries
            ]
            for future in as_completed(futures):
                entry, status, size, detail = future.result()
                with write_lock:
                    if status == "downloaded":
                        report.downloaded += 1
                        report.bytes_written += size
                        index_fh.write(
                            json.dumps(
                                {
                                    **asdict(entry),
                                    "path": str(target_path(out, entry).relative_to(out)),
                                    "size_bytes": size,
                                    "industry": entry.industry
                                    if entry.topic in TOPIC_TO_INDUSTRY
                                    else None,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    elif status == "skipped":
                        report.skipped_existing += 1
                    else:
                        report.failed += 1
                        reason = detail[:80] or "unknown"
                        report.failures_by_reason[reason] = (
                            report.failures_by_reason.get(reason, 0) + 1
                        )
                        failures_fh.write(
                            json.dumps({"url": entry.url, "id": entry.id, "reason": detail})
                            + "\n"
                        )

                    done = report.downloaded + report.skipped_existing + report.failed
                    if on_progress and done % progress_every == 0:
                        report.elapsed_seconds = time.monotonic() - started
                        on_progress(report)
                    # Flush periodically so an interrupted run leaves usable files.
                    if done % 100 == 0:
                        index_fh.flush()
                        failures_fh.flush()
    finally:
        index_fh.close()
        failures_fh.close()

    report.elapsed_seconds = time.monotonic() - started
    return report


def load_failures(out_dir: str | Path) -> list[str]:
    """URLs from a previous run's ``failures.jsonl``."""
    path = Path(out_dir) / FAILURES_FILE
    if not path.exists():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            urls.append(json.loads(line)["url"])
        except (json.JSONDecodeError, KeyError):
            continue
    return urls


def free_bytes(path: str | Path) -> int:
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free


def summarise_plan(entries: Iterable[CorpusEntry]) -> dict[str, dict[str, int]]:
    """``{topic: {type: count}}`` -- the shape of the tree that will be created."""
    grid: dict[str, dict[str, int]] = {}
    for entry in entries:
        grid.setdefault(entry.topic, {})
        grid[entry.topic][entry.type] = grid[entry.topic].get(entry.type, 0) + 1
    return grid
