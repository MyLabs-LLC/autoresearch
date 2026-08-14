"""Downloading .docx files from URLs.

Used directly for a caller-supplied URL list, and as the transport for
:mod:`datax.sources.docxcorpus`.

Everything here assumes the remote end is untrustworthy. A URL ending in ``.docx``
routinely serves an HTML error page with a 200 status, so a download is only accepted
after the bytes themselves are checked: zip magic first, then the presence of
``word/document.xml``. Size is capped while streaming rather than after, so a hostile
or misconfigured server cannot fill the disk.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..extract import ZIP_MAGIC, DocxReadError, extract_docx
from . import FetchReport

USER_AGENT = "datax-docx-collector/1.0 (+dataset construction; contact via repository)"
DEFAULT_TIMEOUT = 30
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024
ALLOWED_SCHEMES = {"http", "https"}


class DownloadError(RuntimeError):
    """A download failed.

    ``transient`` marks failures worth retrying -- timeouts, connection resets, 429s
    and 5xx. A file that simply is not a Word document will never become one, so
    retrying it just wastes requests against the host.
    """

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


# HTTP statuses where retrying is reasonable.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class Download:
    url: str
    path: Path
    sha256: str
    size_bytes: int
    etag_verified: bool = False
    """True when the server sent a plain-MD5 ETag and the downloaded bytes matched it.

    This detects truncation and transit corruption. It is **not** proof of
    authenticity: the same server supplies both the bytes and the ETag.
    """


def _verify_etag(etag: str, md5_hex: str) -> bool | None:
    """Compare a downloaded body against the server's ETag.

    S3-compatible stores (Cloudflare R2 included) set the ETag to the object's MD5 for
    single-part uploads, which makes it a free corruption check. Returns ``True`` on a
    match, ``False`` on a mismatch, and ``None`` when the ETag is absent or is not a
    plain MD5 -- a weak ETag (``W/"..."``) or a multipart ETag (``"<hex>-<n>"``) is not
    comparable and must not be treated as a failure.
    """
    tag = etag.strip()
    if not tag or tag.startswith("W/"):
        return None
    tag = tag.strip('"')
    if len(tag) != 32 or not all(c in "0123456789abcdefABCDEF" for c in tag):
        return None
    return tag.lower() == md5_hex.lower()


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise DownloadError(f"refusing non-http(s) URL scheme {parsed.scheme!r}")
    if not parsed.netloc:
        raise DownloadError("URL has no host")


def download_docx(
    url: str,
    dest_dir: str | Path,
    *,
    filename: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Download:
    """Download one URL and accept it only if the bytes really are a Word document."""
    _check_url(url)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    name = filename or Path(urllib.parse.urlparse(url).path).name or "document.docx"
    if not name.lower().endswith(".docx"):
        name += ".docx"
    # Never let a remote path component escape the destination directory.
    target = dest / Path(name).name
    tmp = target.with_suffix(".partial")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - integrity check against the server's ETag, not security
    total = 0
    etag = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as fh:
            etag = response.headers.get("ETag", "")
            first = response.read(4)
            if first != ZIP_MAGIC:
                raise DownloadError(
                    "response is not a zip container "
                    f"(content-type {response.headers.get('Content-Type', 'unknown')!r}); "
                    "probably an error page"
                )
            fh.write(first)
            digest.update(first)
            md5.update(first)
            total = 4
            while chunk := response.read(1 << 16):
                total += len(chunk)
                if total > max_bytes:
                    raise DownloadError(f"exceeds size cap of {max_bytes} bytes")
                digest.update(chunk)
                md5.update(chunk)
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"HTTP {exc.code}", transient=exc.code in RETRYABLE_STATUS
        ) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        # Network-level failure: worth another attempt.
        tmp.unlink(missing_ok=True)
        raise DownloadError(str(exc), transient=True) from None
    except DownloadError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(str(exc), transient=True) from None

    etag_verified = _verify_etag(etag, md5.hexdigest())
    if etag_verified is False:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"content does not match the server's ETag ({etag}); "
            "the download was truncated or corrupted",
            transient=True,
        )

    tmp.replace(target)

    # Final proof: it must parse as a Word document with extractable structure.
    try:
        extract_docx(target)
    except DocxReadError as exc:
        target.unlink(missing_ok=True)
        raise DownloadError(f"downloaded file is not a usable .docx: {exc}") from None

    return Download(
        url=url,
        path=target,
        sha256=digest.hexdigest(),
        size_bytes=total,
        etag_verified=bool(etag_verified),
    )


def fetch(
    out_dir: str | Path,
    urls: Sequence[str] | Iterable[str],
    *,
    delay_seconds: float = 0.5,
    timeout: int = DEFAULT_TIMEOUT,
) -> FetchReport:
    """Download a list of URLs, tolerating individual failures.

    ``delay_seconds`` is a courtesy pause between requests; leave it non-zero when
    pulling many documents from one host.
    """
    report = FetchReport(provider="web")
    seen_hashes: set[str] = set()
    url_list = list(urls)
    report.requested = len(url_list)

    for index, url in enumerate(url_list):
        if index and delay_seconds:
            time.sleep(delay_seconds)
        try:
            download = download_docx(url, out_dir, timeout=timeout)
        except DownloadError as exc:
            report.fail(url, str(exc))
            continue

        # Identical bytes under two URLs would double-count in every metric.
        if download.sha256 in seen_hashes:
            download.path.unlink(missing_ok=True)
            report.skipped += 1
            continue
        seen_hashes.add(download.sha256)
        report.written += 1
        report.files.append(str(download.path))

    return report
