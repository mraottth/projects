"""A plain, clean .docx from the Markdown a tailored document was written in (asked 2026-11-11).

Someone whose current resume or cover letter is a PDF (or text) gets tailored documents as
Markdown, since only a .docx can be edited in place (core/generate.py). Downloading that as a .md
file was awkward, so the download builds a Word file from it instead: headings, bullet and numbered
lists, **bold**, *italic* and links as text. No Claude call. It's a tidy default layout, not the
person's own; uploading their .docx is what gets that.
"""

import io
import re

from docx import Document
from docx.shared import Pt

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET = re.compile(r"^\s*[-*+•]\s+(.*)$")
NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
# **bold**, __bold__, *italic*, _italic_ (not inside words), ==highlight== and [text](url).
INLINE = re.compile(
    r"(\*\*(?P<b1>.+?)\*\*|__(?P<b2>.+?)__|(?<![\w*])\*(?P<i1>[^*\n]+?)\*(?!\w)"
    r"|(?<![\w_])_(?P<i2>[^_\n]+?)_(?!\w)|==(?:[if]:)?(?P<h>.+?)==|\[(?P<lt>[^\]]+)\]\((?P<lu>[^)]+)\))"
)


def _add_inline(paragraph, text: str) -> None:
    pos = 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos : m.start()])
        if m.group("b1") or m.group("b2"):
            paragraph.add_run(m.group("b1") or m.group("b2")).bold = True
        elif m.group("i1") or m.group("i2"):
            paragraph.add_run(m.group("i1") or m.group("i2")).italic = True
        elif m.group("h"):
            paragraph.add_run(m.group("h"))
        else:
            label, url = m.group("lt"), m.group("lu")
            paragraph.add_run(label if label == url else f"{label} ({url})")
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def markdown_to_docx(markdown: str) -> bytes:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    para_lines: list[str] = []

    def flush() -> None:
        if para_lines:
            _add_inline(doc.add_paragraph(), " ".join(line.strip() for line in para_lines))
            para_lines.clear()

    for raw in (markdown or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if RULE.match(line):
            flush()
            continue
        if m := HEADING.match(line):
            flush()
            _add_inline(doc.add_heading(level=min(len(m.group(1)), 4)), m.group(2).strip())
            continue
        if m := BULLET.match(line):
            flush()
            _add_inline(doc.add_paragraph(style="List Bullet"), m.group(1).strip())
            continue
        if m := NUMBERED.match(line):
            flush()
            _add_inline(doc.add_paragraph(style="List Number"), m.group(1).strip())
            continue
        para_lines.append(line)
    flush()

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
