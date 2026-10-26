"""Generates a tailored resume/cover-letter .docx by cloning the user's own template document's
paragraph styles and exact per-run formatting (bold/size) -- not by describing the formatting to
Claude in prose, which can't produce real visual fidelity, and not by building a fresh style sheet
from scratch, which would drift from what the user actually uses.

Validated live against a real resume during development: deleting all body paragraphs and
re-adding clones of a handful of "exemplar" paragraphs (one per structural role: title, contact
line, section heading, bulleted line, bold-header line, blank spacer) survives a save/reload round
trip with margins and named styles intact. The one easy-to-miss gotcha: a run's font SIZE is often
set explicitly per-run in real documents (not just inherited from the named style's default), so it
has to be captured from an exemplar run and re-applied -- otherwise text silently reverts to
whatever size the style itself defaults to.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from pydantic import BaseModel


class ResumeLine(BaseModel):
    kind: Literal["normal", "bullet", "description"]
    bold_part: str | None = None
    text: str


class ResumeSection(BaseModel):
    heading: str
    lines: list[ResumeLine]


class ResumeContent(BaseModel):
    name: str
    contact_line: str
    sections: list[ResumeSection]


class CoverLetterContent(BaseModel):
    paragraphs: list[str]


class TemplateError(Exception):
    """The template .docx doesn't have paragraph styles we can recognize/clone from."""


@dataclass
class _Exemplars:
    title: object
    contact: object
    heading: object
    bullet: object
    description: object
    normal_bold: object
    blank: object
    sizes: dict


def _first_nonempty_run(paragraph):
    for run in paragraph.runs:
        if run.text.strip():
            return run
    return None


def _capture_exemplars(doc: Document) -> _Exemplars:
    found: dict = {}
    sizes: dict = {}
    for p in doc.paragraphs:
        style = p.style.name
        runs = [r for r in p.runs if r.text.strip()]

        if style == "Title" and "title" not in found:
            found["title"] = copy.deepcopy(p._p)
            sizes["title"] = runs[0].font.size if runs else None
        elif style.startswith("Heading") and p.text.strip() and "heading" not in found:
            found["heading"] = copy.deepcopy(p._p)
            sizes["heading"] = runs[0].font.size if runs else None
        elif style == "List Paragraph" and p.text.strip() and "bullet" not in found:
            found["bullet"] = copy.deepcopy(p._p)
            sizes["bullet"] = runs[0].font.size if runs else None
        elif style not in ("Title", "Normal", "List Paragraph") and style.lower() not in (
            "heading 1",
            "heading 2",
            "heading 3",
            "heading 4",
        ) and "description" not in found and p.text.strip():
            # a distinct custom style used for a one-off italic/description-style line (e.g. "p1")
            found["description"] = copy.deepcopy(p._p)
            sizes["description"] = runs[0].font.size if runs else None
        elif style == "Normal" and not p.text.strip() and "blank" not in found:
            found["blank"] = copy.deepcopy(p._p)
        elif style == "Normal" and p.text.strip():
            bolds = [r.bold for r in runs]
            if "contact" not in found and not any(bolds):
                found["contact"] = copy.deepcopy(p._p)
                sizes["contact"] = runs[0].font.size if runs else None
            if "normal_bold" not in found and True in bolds and bolds.count(True) < len(bolds):
                found["normal_bold"] = copy.deepcopy(p._p)
                bold_run = next(r for r in runs if r.bold)
                nonbold_run = next(r for r in runs if not r.bold)
                sizes["normal_bold_bold"] = bold_run.font.size
                sizes["normal_bold_nonbold"] = nonbold_run.font.size

    required = {"title", "contact", "heading", "bullet", "normal_bold", "blank"}
    missing = required - found.keys()
    if missing:
        raise TemplateError(f"template is missing recognizable paragraph styles for: {missing}")
    if "description" not in found:
        found["description"] = found["bullet"]
        sizes["description"] = sizes.get("bullet")

    return _Exemplars(
        title=found["title"],
        contact=found["contact"],
        heading=found["heading"],
        bullet=found["bullet"],
        description=found["description"],
        normal_bold=found["normal_bold"],
        blank=found["blank"],
        sizes=sizes,
    )


def _clear_runs(p_element) -> Paragraph:
    """Strip a paragraph down to just its paragraph-properties (pPr), removing every other child
    -- not only plain <w:r> runs. A real-world document can wrap text in <w:hyperlink>,
    <w:ins>/<w:del> (tracked changes), bookmarks, etc.; python-docx's `.runs` only sees direct-
    child <w:r> elements, so clearing via `.runs` alone silently leaves that wrapped text behind
    in the paragraph (confirmed live: a hyperlinked word ("thesis") survived and leaked into
    unrelated generated content before this fix)."""
    p_pr = p_element.find(qn("w:pPr"))
    for child in list(p_element):
        if child is not p_pr:
            p_element.remove(child)
    return Paragraph(p_element, None)


def _set_plain(p_element, text: str, size=None) -> None:
    paragraph = _clear_runs(p_element)
    if not text:
        return
    run = paragraph.add_run(text)
    if size is not None:
        run.font.size = size


def _set_bold_split(p_element, bold_text: str, rest_text: str, bold_size=None, rest_size=None) -> None:
    paragraph = _clear_runs(p_element)
    run1 = paragraph.add_run(bold_text)
    run1.bold = True
    if bold_size is not None:
        run1.font.size = bold_size
    if rest_text:
        run2 = paragraph.add_run(rest_text)
        run2.bold = False
        if rest_size is not None:
            run2.font.size = rest_size


def _get_sect_pr(body):
    sect_pr = body.find(qn("w:sectPr"))
    if sect_pr is None:
        raise TemplateError("template has no section properties (sectPr) -- unexpected docx structure")
    return sect_pr


def write_resume_docx(template_path: Path, content: ResumeContent, output_path: Path) -> None:
    doc = Document(str(template_path))
    body = doc.element.body
    exemplars = _capture_exemplars(doc)

    for p in list(doc.paragraphs):
        p._p.getparent().remove(p._p)
    sect_pr = _get_sect_pr(body)

    def append(element) -> None:
        body.insert(list(body).index(sect_pr), element)

    title_el = copy.deepcopy(exemplars.title)
    _set_plain(title_el, content.name, exemplars.sizes.get("title"))
    append(title_el)

    contact_el = copy.deepcopy(exemplars.contact)
    _set_plain(contact_el, content.contact_line, exemplars.sizes.get("contact"))
    append(contact_el)

    for section in content.sections:
        heading_el = copy.deepcopy(exemplars.heading)
        _set_plain(heading_el, section.heading, exemplars.sizes.get("heading"))
        append(heading_el)
        append(copy.deepcopy(exemplars.blank))

        for line in section.lines:
            if line.kind == "bullet":
                el = copy.deepcopy(exemplars.bullet)
                _set_plain(el, line.text, exemplars.sizes.get("bullet"))
            elif line.kind == "description":
                el = copy.deepcopy(exemplars.description)
                _set_plain(el, line.text, exemplars.sizes.get("description"))
            elif line.bold_part:
                el = copy.deepcopy(exemplars.normal_bold)
                _set_bold_split(
                    el,
                    line.bold_part,
                    line.text,
                    exemplars.sizes.get("normal_bold_bold"),
                    exemplars.sizes.get("normal_bold_nonbold"),
                )
            else:
                el = copy.deepcopy(exemplars.contact)
                _set_plain(el, line.text, exemplars.sizes.get("contact"))
            append(el)
        append(copy.deepcopy(exemplars.blank))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))


def count_cover_letter_slots(template_path: Path) -> int:
    """How many non-blank paragraphs the cover-letter template has -- passed to the generation
    prompt so Claude targets roughly the same number of paragraphs, keeping the common case (same
    count) exact rather than relying on the fallback handling in write_cover_letter_docx."""
    doc = Document(str(template_path))
    return sum(1 for p in doc.paragraphs if p.text.strip())


def write_cover_letter_docx(template_path: Path, content: CoverLetterContent, output_path: Path) -> None:
    """Cover letters are simpler/more linear than resumes: replace each non-blank template
    paragraph's text positionally (same style/run-formatting, new words), preserving blank spacer
    paragraphs as-is. If there are fewer new paragraphs than template slots, leftover slots are
    cleared to blank (never left with stale text from whatever the template was originally for).
    If there are more, extras are appended using the last paragraph's formatting as a fallback."""
    doc = Document(str(template_path))
    non_blank = [p for p in doc.paragraphs if p.text.strip()]
    if not non_blank:
        raise TemplateError("cover letter template has no non-blank paragraphs to use as a style reference")

    last_run = _first_nonempty_run(non_blank[-1])
    fallback_size = last_run.font.size if last_run else None
    fallback_style_el = copy.deepcopy(non_blank[-1]._p)

    for i, p in enumerate(non_blank):
        run = _first_nonempty_run(p)
        size = run.font.size if run else None
        text = content.paragraphs[i] if i < len(content.paragraphs) else ""
        _set_plain(p._p, text, size)

    if len(content.paragraphs) > len(non_blank):
        body = doc.element.body
        sect_pr = _get_sect_pr(body)
        blank_source = next((p for p in doc.paragraphs if not p.text.strip()), None)
        for extra in content.paragraphs[len(non_blank) :]:
            if blank_source is not None:
                body.insert(list(body).index(sect_pr), copy.deepcopy(blank_source._p))
            new_el = copy.deepcopy(fallback_style_el)
            _set_plain(new_el, extra, fallback_size)
            body.insert(list(body).index(sect_pr), new_el)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
