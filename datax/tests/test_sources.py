"""Source-layer behaviour that does not require the network."""

from __future__ import annotations

import zipfile

import pytest

from datax.sources.docxcorpus import TOPIC_TO_INDUSTRY, CorpusEntry
from datax.sources.nemotron import TARGET_DOMAINS, NemotronRow, _balanced, _parse_spans
from datax.sources.web import DownloadError, download_docx


def make_row(uid="u1", domain="Healthcare", locale="us"):
    return NemotronRow(
        uid=uid,
        domain=domain,
        document_type="Medical Record",
        document_description="",
        document_format="unstructured",
        locale=locale,
        text="text",
        spans=[],
    )


def test_row_key_disambiguates_locale_variants():
    """Nemotron reuses a uid across its two locale variants; the key must not."""
    us, intl = make_row(locale="us"), make_row(locale="intl")
    assert us.uid == intl.uid
    assert us.key != intl.key


def test_spans_parse_from_python_repr():
    raw = "[{'start': 3, 'end': 8, 'text': 'Jason', 'label': 'first_name'}]"
    parsed = _parse_spans(raw)
    assert parsed[0]["label"] == "first_name"


def test_spans_parse_survives_garbage():
    assert _parse_spans("not a literal") == []


def test_balanced_sampling_caps_each_domain():
    rows = [make_row(uid=f"h{i}", domain="Healthcare") for i in range(10)]
    rows += [make_row(uid=f"f{i}", domain="Finance") for i in range(2)]
    selected = _balanced(iter(rows), per_domain=3)
    counts = {domain: 0 for domain in TARGET_DOMAINS}
    for row in selected:
        counts[row.domain] += 1
    assert counts["Healthcare"] == 3
    assert counts["Finance"] == 2


def test_only_unambiguous_domains_are_targeted():
    # Insurance and Pharmaceuticals straddle two industries; they must stay out of gold.
    assert set(TARGET_DOMAINS) == {"Healthcare", "Finance", "Government"}
    assert set(TOPIC_TO_INDUSTRY.values()) == {"healthcare", "finance", "government"}


def test_corpus_entry_maps_topic_to_industry():
    entry = CorpusEntry(
        id="x", filename="f.docx", type="forms", topic="healthcare",
        language="en", word_count=200, confidence=0.9, url="https://example.com/f.docx",
    )
    assert entry.industry == "healthcare"


def test_download_rejects_non_http_scheme(tmp_path):
    with pytest.raises(DownloadError, match="scheme"):
        download_docx("file:///etc/passwd", tmp_path)


def test_download_rejects_html_served_as_docx(tmp_path, monkeypatch):
    """The common real-world failure: a 200 response whose body is an error page."""
    import io
    import urllib.request

    class FakeResponse(io.BytesIO):
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"<html>404</html>")
    )
    with pytest.raises(DownloadError, match="not a zip container"):
        download_docx("https://example.com/doc.docx", tmp_path)


def test_download_rejects_zip_that_is_not_a_document(tmp_path, monkeypatch):
    import io
    import urllib.request

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("readme.txt", "not word")
    payload = buffer.getvalue()

    class FakeResponse(io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResponse(payload))
    with pytest.raises(DownloadError, match="not a usable .docx"):
        download_docx("https://example.com/doc.docx", tmp_path)
