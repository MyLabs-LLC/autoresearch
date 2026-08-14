"""Reading text and structure out of .docx files, using only the standard library.

This path runs over files downloaded from the open internet, so it avoids third-party
parsers and treats every input as hostile: zip bombs, absurd part counts, and XML
entity attacks are all bounded or rejected before parsing.

The text model is deliberately simple and must stay in lockstep with
:mod:`datax.docxio`:

* a paragraph contributes its text, paragraphs are joined with ``\\n``
* ``<w:tab/>`` becomes ``\\t`` and ``<w:br/>`` becomes ``\\n``
* a table contributes one line per row, cells joined with ``\\t``

Anything that breaks that correspondence breaks gold span offsets.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

DOCUMENT_PART = "word/document.xml"

# Guardrails for untrusted files.
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_PARTS = 5000

# A .docx is a zip; every zip starts with this signature.
ZIP_MAGIC = b"PK\x03\x04"


class DocxReadError(ValueError):
    pass


@dataclass
class ExtractedDocx:
    path: Path
    text: str
    sha256: str
    size_bytes: int
    paragraph_count: int
    table_count: int
    word_count: int
    has_headers_or_footers: bool
    core_properties: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def looks_like_docx(path: str | Path) -> bool:
    """Cheap magic-byte and part check. A file served as .docx by a web server is
    routinely an HTML error page instead, so never trust the extension."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            if fh.read(4) != ZIP_MAGIC:
                return False
        with zipfile.ZipFile(p) as zf:
            return DOCUMENT_PART in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _check_zip_safety(zf: zipfile.ZipFile) -> None:
    infos = zf.infolist()
    if len(infos) > MAX_PARTS:
        raise DocxReadError(f"refusing archive with {len(infos)} parts (limit {MAX_PARTS})")
    total = 0
    for info in infos:
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise DocxReadError("refusing archive: uncompressed size exceeds limit")
        if info.compress_size > 0 and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise DocxReadError(f"refusing archive: {info.filename} has suspicious compression ratio")


def _parse_xml(data: bytes) -> ElementTree.Element:
    # ElementTree does not expand external entities, but it will happily expand
    # internal ones; forbid a DOCTYPE outright rather than reason about billion laughs.
    head = data[:512].lstrip()
    if head.startswith(b"<?xml") :
        head = head.split(b"?>", 1)[-1].lstrip()
    if head.startswith(b"<!DOCTYPE"):
        raise DocxReadError("refusing XML part containing a DOCTYPE declaration")
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise DocxReadError(f"malformed XML part: {exc}") from None


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    # iter() walks descendants in document order, which keeps runs, tabs and breaks
    # in the sequence a reader would see them.
    for node in paragraph.iter():
        tag = node.tag
        if tag == f"{W}t":
            parts.append(node.text or "")
        elif tag == f"{W}tab":
            parts.append("\t")
        elif tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
        elif tag == f"{W}noBreakHyphen":
            parts.append("-")
    return "".join(parts)


def _block_lines(container: ElementTree.Element) -> tuple[list[str], int, int]:
    """Walk a body-like container in document order, returning its lines plus
    paragraph and table counts."""
    lines: list[str] = []
    paragraphs = 0
    tables = 0
    for child in container:
        if child.tag == f"{W}p":
            lines.append(_paragraph_text(child))
            paragraphs += 1
        elif child.tag == f"{W}tbl":
            tables += 1
            for row in child.findall(f"{W}tr"):
                cells: list[str] = []
                for cell in row.findall(f"{W}tc"):
                    cell_lines, cell_paragraphs, _ = _block_lines(cell)
                    cells.append(" ".join(part for part in cell_lines if part))
                    paragraphs += cell_paragraphs
                lines.append("\t".join(cells))
        elif child.tag == f"{W}sdt":
            # Structured document tag (content control): recurse into its content.
            content = child.find(f"{W}sdtContent")
            if content is not None:
                sub_lines, sub_paragraphs, sub_tables = _block_lines(content)
                lines.extend(sub_lines)
                paragraphs += sub_paragraphs
                tables += sub_tables
    return lines, paragraphs, tables


_CORE_PROPS_PART = "docProps/core.xml"
_CORE_FIELDS = {
    "{http://purl.org/dc/elements/1.1/}title": "title",
    "{http://purl.org/dc/elements/1.1/}creator": "creator",
    "{http://purl.org/dc/elements/1.1/}subject": "subject",
    "{http://purl.org/dc/terms/}created": "created",
    "{http://purl.org/dc/terms/}modified": "modified",
    "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy": "last_modified_by",
}


def _core_properties(zf: zipfile.ZipFile) -> dict[str, str]:
    if _CORE_PROPS_PART not in zf.namelist():
        return {}
    try:
        root = _parse_xml(zf.read(_CORE_PROPS_PART))
    except DocxReadError:
        return {}
    props: dict[str, str] = {}
    for child in root:
        key = _CORE_FIELDS.get(child.tag)
        if key and child.text:
            props[key] = child.text
    return props


def extract_docx(path: str | Path) -> ExtractedDocx:
    """Extract text and light structure from a .docx.

    Headers and footers are detected but **not** included in the text, because they
    repeat on every page and would corrupt the single linear offset space that spans
    are expressed in.
    """
    p = Path(path)
    raw = p.read_bytes()
    if not raw.startswith(ZIP_MAGIC):
        raise DocxReadError(f"{p.name} is not a zip container (probably an HTML error page)")

    with zipfile.ZipFile(p) as zf:
        _check_zip_safety(zf)
        names = zf.namelist()
        if DOCUMENT_PART not in names:
            raise DocxReadError(f"{p.name} has no {DOCUMENT_PART}; not a Word document")
        root = _parse_xml(zf.read(DOCUMENT_PART))
        core = _core_properties(zf)
        has_hf = any(n.startswith(("word/header", "word/footer")) for n in names)

    body = root.find(f"{W}body")
    if body is None:
        raise DocxReadError(f"{p.name} has no <w:body>")

    lines, paragraphs, tables = _block_lines(body)
    text = "\n".join(lines)

    return ExtractedDocx(
        path=p,
        text=text,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        paragraph_count=paragraphs,
        table_count=tables,
        word_count=len(text.split()),
        has_headers_or_footers=has_hf,
        core_properties=core,
    )


def extract_docx_text(path: str | Path) -> str:
    return extract_docx(path).text
