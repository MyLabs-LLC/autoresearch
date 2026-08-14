"""Bulk downloader behaviour, with the network stubbed out.

The properties under test are the ones that decide whether a multi-hour, 90,000-file
job survives contact with reality: resumability, retry policy, and never losing a
failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datax.sources import bulk
from datax.sources.docxcorpus import CorpusEntry
from datax.sources.web import Download, DownloadError


def entry(id_="a" * 64, topic="healthcare", type_="forms", lang="en", wc=200, conf=0.9):
    return CorpusEntry(
        id=id_,
        filename="download",
        type=type_,
        topic=topic,
        language=lang,
        word_count=wc,
        confidence=conf,
        url=f"https://docxcorp.us/documents/{id_}.docx",
    )


# -- layout ----------------------------------------------------------------


def test_target_path_is_topic_type_id():
    path = bulk.target_path(Path("/out"), entry(id_="b" * 64))
    assert path == Path("/out/healthcare/forms/" + "b" * 64 + ".docx")


def test_target_path_uses_the_id_not_the_corpus_filename():
    """The corpus `filename` column is routinely 'content' or 'index.php', which would
    collide across thousands of documents."""
    path = bulk.target_path(Path("/out"), entry())
    assert path.name.startswith("a" * 10)
    assert "download" not in path.name


def test_summarise_plan_counts_the_tree():
    entries = [
        entry(id_="1" * 64, topic="finance", type_="legal"),
        entry(id_="2" * 64, topic="finance", type_="legal"),
        entry(id_="3" * 64, topic="finance", type_="forms"),
    ]
    assert bulk.summarise_plan(entries) == {"finance": {"legal": 2, "forms": 1}}


# -- download loop ---------------------------------------------------------


@pytest.fixture()
def fake_download(monkeypatch):
    """Replace the network with a recorder that writes a plausible file."""
    calls: list[str] = []

    def _download(url, dest_dir, *, filename=None, timeout=30, max_bytes=0):
        calls.append(url)
        target = Path(dest_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PK\x03\x04fake-docx-body")
        return Download(url=url, path=target, sha256="0" * 64, size_bytes=17)

    monkeypatch.setattr(bulk, "download_docx", _download)
    return calls


def test_downloads_into_the_tree_and_records_an_index(tmp_path, fake_download):
    entries = [
        entry(id_="1" * 64, topic="finance", type_="legal"),
        entry(id_="2" * 64, topic="healthcare", type_="reports"),
    ]
    report = bulk.download_all(entries, tmp_path, workers=2)

    assert report.downloaded == 2 and report.failed == 0
    assert (tmp_path / "finance" / "legal" / ("1" * 64 + ".docx")).exists()
    assert (tmp_path / "healthcare" / "reports" / ("2" * 64 + ".docx")).exists()

    rows = [json.loads(line) for line in (tmp_path / bulk.INDEX_FILE).read_text().splitlines()]
    assert len(rows) == 2
    # Metadata the folder names cannot carry must survive alongside the files.
    assert {"language", "confidence", "word_count", "url", "path"} <= set(rows[0])


def test_rerun_skips_existing_files(tmp_path, fake_download):
    entries = [entry(id_="1" * 64)]
    bulk.download_all(entries, tmp_path, workers=1)
    assert len(fake_download) == 1

    second = bulk.download_all(entries, tmp_path, workers=1)
    assert second.skipped_existing == 1
    assert second.downloaded == 0
    assert len(fake_download) == 1  # no second request


def test_zero_byte_file_is_refetched(tmp_path, fake_download):
    """A file truncated by a killed run must not be mistaken for a finished one."""
    target = bulk.target_path(tmp_path, entry(id_="1" * 64))
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    report = bulk.download_all([entry(id_="1" * 64)], tmp_path, workers=1)
    assert report.downloaded == 1


def test_verify_existing_refetches_a_corrupt_file(tmp_path, fake_download):
    target = bulk.target_path(tmp_path, entry(id_="1" * 64))
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not a docx at all")

    # Without --verify-existing the corrupt file looks finished and is kept.
    kept = bulk.download_all([entry(id_="1" * 64)], tmp_path, workers=1)
    assert kept.skipped_existing == 1

    # With it, the file is re-opened, found invalid, and fetched again.
    repaired = bulk.download_all(
        [entry(id_="1" * 64)], tmp_path, workers=1, verify_existing=True
    )
    assert repaired.downloaded == 1


# -- retry policy ----------------------------------------------------------


def test_transient_failures_are_retried(tmp_path, monkeypatch):
    attempts = {"n": 0}

    def flaky(url, dest_dir, *, filename=None, timeout=30, max_bytes=0):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise DownloadError("connection reset", transient=True)
        target = Path(dest_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PK\x03\x04ok")
        return Download(url=url, path=target, sha256="0" * 64, size_bytes=8)

    monkeypatch.setattr(bulk, "download_docx", flaky)
    monkeypatch.setattr(bulk.time, "sleep", lambda _s: None)  # no real backoff in tests

    report = bulk.download_all([entry()], tmp_path, workers=1, retries=3)
    assert report.downloaded == 1
    assert attempts["n"] == 3


def test_permanent_failures_are_not_retried(tmp_path, monkeypatch):
    """A URL serving HTML will never become a .docx; retrying wastes requests."""
    attempts = {"n": 0}

    def permanent(url, dest_dir, *, filename=None, timeout=30, max_bytes=0):
        attempts["n"] += 1
        raise DownloadError("response is not a zip container", transient=False)

    monkeypatch.setattr(bulk, "download_docx", permanent)
    report = bulk.download_all([entry()], tmp_path, workers=1, retries=5)

    assert report.failed == 1
    assert attempts["n"] == 1


def test_failures_are_written_and_reloadable(tmp_path, monkeypatch):
    def permanent(url, dest_dir, *, filename=None, timeout=30, max_bytes=0):
        raise DownloadError("HTTP 404", transient=False)

    monkeypatch.setattr(bulk, "download_docx", permanent)
    bulk.download_all([entry(id_="1" * 64), entry(id_="2" * 64)], tmp_path, workers=2)

    urls = bulk.load_failures(tmp_path)
    assert len(urls) == 2
    assert all(u.startswith("https://docxcorp.us/") for u in urls)


def test_load_failures_on_a_fresh_directory(tmp_path):
    assert bulk.load_failures(tmp_path) == []


def test_one_failure_does_not_stop_the_run(tmp_path, monkeypatch):
    def sometimes(url, dest_dir, *, filename=None, timeout=30, max_bytes=0):
        if url.endswith("1" * 64 + ".docx"):
            raise DownloadError("HTTP 404", transient=False)
        target = Path(dest_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PK\x03\x04ok")
        return Download(url=url, path=target, sha256="0" * 64, size_bytes=8)

    monkeypatch.setattr(bulk, "download_docx", sometimes)
    report = bulk.download_all(
        [entry(id_="1" * 64), entry(id_="2" * 64), entry(id_="3" * 64)], tmp_path, workers=2
    )
    assert report.downloaded == 2
    assert report.failed == 1
