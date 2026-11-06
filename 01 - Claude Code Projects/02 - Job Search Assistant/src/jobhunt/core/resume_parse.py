from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"unsupported resume/cover-letter file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _iter_block_items(parent):
    """Yield each top-level Paragraph/Table in document order. python-docx's own `.paragraphs`
    and `.tables` only give flat, order-losing lists and skip nested content entirely -- several
    real resumes here use a table-based two-column layout (dates/company in one cell, bullets in
    another), which `.paragraphs` alone misses completely (confirmed live: a handful of older
    resumes extracted under 150 characters instead of several thousand)."""
    body = parent.element.body if hasattr(parent, "element") else parent._tc
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _table_text(table: Table) -> str:
    lines = []
    for row in table.rows:
        for cell in row.cells:
            text = "\n".join(_extract_blocks(cell))
            if text.strip():
                lines.append(text)
    return "\n".join(lines)


def _extract_blocks(parent) -> list[str]:
    lines = []
    for block in _iter_block_items(parent):
        if isinstance(block, Paragraph):
            if block.text.strip():
                lines.append(block.text)
        else:
            table_text = _table_text(block)
            if table_text.strip():
                lines.append(table_text)
    return lines


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(_extract_blocks(doc))
