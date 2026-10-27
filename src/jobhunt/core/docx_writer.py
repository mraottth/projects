"""Generates a tailored resume/cover-letter .docx from the user's own template document.

Resumes and cover letters use two deliberately different strategies:

- Cover letters (`write_cover_letter_docx`) are simple, linear documents -- positional in-place
  text replacement on the template's own existing paragraphs, preserving each one's own style/
  run formatting exactly. Nothing is deleted or rebuilt.
- Resumes (`write_resume_edits`) have real structural variety (distinct bullet-list numbering per
  job, tables some users rely on, per-line formatting quirks) that a from-scratch rebuild loses.
  So resumes are edited **in place** too, via a small set of context-anchored operations
  (`ResumeEditOp`) applied as native Word tracked changes (`<w:ins>`/`<w:del>`) -- every paragraph
  not explicitly targeted by an op is left byte-for-byte untouched, and the user reviews/accepts
  each change in Word (or Google Docs, which renders Word's tracked changes as its own Suggesting
  mode on import) exactly like a human editor's suggested edits, rather than trusting a wholesale
  regeneration. An earlier version of this module *did* rebuild resumes from scratch out of a
  handful of cloned "exemplar" paragraphs; that approach is what caused consistently bad output
  formatting (every bullet in the whole document ended up sharing one global list-numbering
  definition, tables were silently orphaned) and has been replaced entirely.
"""

import copy
import datetime
from pathlib import Path
from typing import Literal

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from pydantic import BaseModel, model_validator

# Forced on every run write_cover_letter_docx generates, rather than inherited from the template --
# real-world templates are often inconsistent (e.g. a resume's "Normal" style can be explicitly
# Times New Roman while its Title/Heading styles fall through to the document theme's Calibri with
# no run-level override at all), so "inherit whatever the cloned paragraph happens to use" doesn't
# reliably produce one consistent font. write_resume_edits does NOT use this -- it only ever touches
# small, locally-consistent neighborhoods of an otherwise-untouched document, so new/replaced resume
# text inherits its font from the specific neighboring run it's replacing/following instead (see
# `_rpr_of`), which is strictly more correct and needs no hardcoded font name.
GENERATED_FONT = "Calibri"

# Author attribution recorded on every tracked-change element write_resume_edits creates. This is
# metadata Word/Google Docs display next to each suggested change (not a document field, and not
# the candidate's own name/identity) -- it should read as "a tool proposed this," which is what it
# functionally is.
CHANGE_AUTHOR = "Job Search Assistant"


class CoverLetterContent(BaseModel):
    paragraphs: list[str]


class TemplateError(Exception):
    """The template .docx doesn't have paragraph styles/structure we can recognize/clone from."""


class AnchorNotFoundError(Exception):
    """A proposed edit's anchor_text doesn't match any paragraph in the template."""


class AmbiguousAnchorError(Exception):
    """A proposed edit's anchor_text matches more than one paragraph -- no safe target to pick."""


class DisallowedAnchorError(Exception):
    """A proposed edit targets a line (name/contact info) that must never vary per job."""


class TemplateLine(BaseModel):
    """One addressable line from the user's real resume template, shown to Claude so it can
    propose edits anchored to exact existing text instead of generating a document from scratch."""

    role: Literal["title", "contact", "heading", "job_header", "bullet", "description", "normal"]
    text: str
    section: str | None = None  # nearest preceding heading's text, e.g. "WORK EXPERIENCE"
    under: str | None = None  # nearest preceding job_header's text, e.g. "ACME CORP | Engineer"


class ResumeEditOp(BaseModel):
    op: Literal["replace", "insert_after", "delete"]
    anchor_text: str
    new_text: str | None = None
    bold_part: str | None = None  # only for anchors with a bold+non-bold split (job_header lines)

    @model_validator(mode="after")
    def _check_fields_for_op(self) -> "ResumeEditOp":
        if self.op in ("replace", "insert_after") and not self.new_text:
            raise ValueError(f"{self.op} requires new_text")
        if self.op == "delete" and (self.new_text or self.bold_part):
            raise ValueError("delete must not set new_text or bold_part")
        return self


class ResumeEditPlan(BaseModel):
    ops: list[ResumeEditOp]


def _first_nonempty_run(paragraph):
    for run in paragraph.runs:
        if run.text.strip():
            return run
    return None


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
    run.font.name = GENERATED_FONT
    if size is not None:
        run.font.size = size


def _set_bold_split(p_element, bold_text: str, rest_text: str, bold_size=None, rest_size=None) -> None:
    paragraph = _clear_runs(p_element)
    run1 = paragraph.add_run(bold_text)
    run1.bold = True
    run1.font.name = GENERATED_FONT
    if bold_size is not None:
        run1.font.size = bold_size
    if rest_text:
        run2 = paragraph.add_run(rest_text)
        run2.bold = False
        run2.font.name = GENERATED_FONT
        if rest_size is not None:
            run2.font.size = rest_size


def _get_sect_pr(body):
    sect_pr = body.find(qn("w:sectPr"))
    if sect_pr is None:
        raise TemplateError("template has no section properties (sectPr) -- unexpected docx structure")
    return sect_pr


# ---------------------------------------------------------------------------
# Resume editing: extracting addressable lines
# ---------------------------------------------------------------------------


def _extract_lines_with_paragraphs(doc: Document) -> list[tuple[Paragraph, TemplateLine]]:
    """Walks the template's top-level paragraphs in document order (same enumeration
    write_cover_letter_docx already uses -- not resume_parse.py's table-aware `_iter_block_items`,
    which exists for a different concern: flattening table content into plain text for candidate-
    profile synthesis. Reusing it here would let Claude anchor on text inside a table cell that
    this module can't locate/mutate. Table-based resumes stay an explicit, documented unsupported
    case.) Classifies every non-blank paragraph's structural role and pairs it with its real
    Paragraph object, so write_resume_edits can resolve an anchor straight to the paragraph it
    needs to mutate without a second pass over the document."""
    lines: list[tuple[Paragraph, TemplateLine]] = []
    section: str | None = None
    under: str | None = None
    contact_seen = False

    for p in doc.paragraphs:
        if not p.text.strip():
            continue
        style = p.style.name
        runs = [r for r in p.runs if r.text.strip()]
        bolds = [bool(r.bold) for r in runs]

        if style == "Title":
            role: str = "title"
        elif style.startswith("Heading"):
            role = "heading"
        elif style == "List Paragraph":
            role = "bullet"
        elif style not in ("Title", "Normal", "List Paragraph"):
            role = "description"
        elif bolds and any(bolds) and not all(bolds):
            role = "job_header"
        elif not contact_seen:
            role = "contact"
            contact_seen = True
        else:
            role = "normal"

        line = TemplateLine(role=role, text=p.text, section=section, under=under)
        lines.append((p, line))

        if role == "heading":
            section = p.text
            under = None
        elif role == "job_header":
            under = p.text

    return lines


def extract_addressable_lines(template_path: Path) -> list[TemplateLine]:
    """The original resume's own lines, tagged with structural role -- shown to Claude so it can
    propose edits anchored to exact existing text instead of generating a resume from scratch."""
    doc = Document(str(template_path))
    return [line for _, line in _extract_lines_with_paragraphs(doc)]


# ---------------------------------------------------------------------------
# Resume editing: applying edits as tracked changes
# ---------------------------------------------------------------------------


def _change_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _next_change_id_counter(doc: Document):
    """A w:id counter for new tracked-change elements, seeded past any w:id already present
    anywhere in the template (bookmarks, an existing tracked change, etc.) so new ids never
    collide with ones the template already uses."""
    max_id = 0
    for el in doc.element.body.iter():
        raw = el.get(qn("w:id"))
        if raw is None:
            continue
        try:
            max_id = max(max_id, int(raw))
        except ValueError:
            continue
    counter = max_id

    def next_id() -> int:
        nonlocal counter
        counter += 1
        return counter

    return next_id


def _rpr_of(run) -> object | None:
    """A deep copy of a run's <w:rPr>, or None if it has none (inherits from its paragraph style)."""
    rpr = run._r.find(qn("w:rPr"))
    return copy.deepcopy(rpr) if rpr is not None else None


def _resolve_run_rprs(runs: list) -> tuple[object | None, object | None]:
    """(bold_rpr, non_bold_rpr) copied from the first bold and first non-bold run among `runs`,
    each falling back to the other if only one kind is present, or (None, None) if `runs` is
    empty."""
    bold = next((r for r in runs if r.bold), None)
    non_bold = next((r for r in runs if not r.bold), None)
    bold_rpr = _rpr_of(bold) if bold is not None else (_rpr_of(non_bold) if non_bold is not None else None)
    non_bold_rpr = (
        _rpr_of(non_bold) if non_bold is not None else (_rpr_of(bold) if bold is not None else None)
    )
    return bold_rpr, non_bold_rpr


def _make_text_run(rpr, text: str, *, deleted: bool):
    """A <w:r> element with `text`, cloning `rpr` (if given) for its formatting -- <w:t> for live
    text, <w:delText> for text marked as a tracked deletion (the OOXML distinction Word uses to
    tell "this text still exists" from "this text is proposed for removal")."""
    run_el = OxmlElement("w:r")
    if rpr is not None:
        run_el.append(copy.deepcopy(rpr))
    text_el = OxmlElement("w:delText" if deleted else "w:t")
    text_el.set(qn("xml:space"), "preserve")
    text_el.text = text
    run_el.append(text_el)
    return run_el


def _wrap_change(tag: str, next_id, *run_elements):
    """Wraps one or more run elements in a <w:ins> or <w:del> (`tag`), with a fresh w:id and this
    module's shared w:author/w:date."""
    change_el = OxmlElement(tag)
    change_el.set(qn("w:id"), str(next_id()))
    change_el.set(qn("w:author"), CHANGE_AUTHOR)
    change_el.set(qn("w:date"), _change_timestamp())
    for run_el in run_elements:
        change_el.append(run_el)
    return change_el


def _mark_paragraph_mark(p_element, tag: str, next_id) -> None:
    """Marks the paragraph's own end-of-paragraph mark as inserted/deleted (tag="w:ins"/"w:del"),
    via <w:pPr><w:rPr><w:ins/-or-w:del/></w:rPr></w:pPr> -- without this, accepting/rejecting an
    inserted or deleted paragraph in Word would add/remove its text but leave a stray blank line
    behind, since the paragraph break itself wouldn't be marked as part of the change."""
    p_pr = p_element.find(qn("w:pPr"))
    if p_pr is None:
        p_pr = OxmlElement("w:pPr")
        p_element.insert(0, p_pr)
    r_pr = p_pr.find(qn("w:rPr"))
    if r_pr is None:
        r_pr = OxmlElement("w:rPr")
        p_pr.append(r_pr)
    mark_el = OxmlElement(tag)
    mark_el.set(qn("w:id"), str(next_id()))
    mark_el.set(qn("w:author"), CHANGE_AUTHOR)
    mark_el.set(qn("w:date"), _change_timestamp())
    r_pr.insert(0, mark_el)


def _apply_replace(paragraph: Paragraph, op: ResumeEditOp, next_id) -> None:
    p_el = paragraph._p
    old_runs = [r for r in paragraph.runs if r.text.strip()]

    if old_runs:
        del_runs = [_make_text_run(_rpr_of(r), r.text, deleted=True) for r in old_runs]
    else:
        del_runs = [_make_text_run(None, paragraph.text, deleted=True)]
    del_el = _wrap_change("w:del", next_id, *del_runs)

    if op.bold_part is not None:
        bold_rpr, rest_rpr = _resolve_run_rprs(old_runs)
        ins_runs = [
            _make_text_run(bold_rpr, op.bold_part, deleted=False),
            _make_text_run(rest_rpr, op.new_text, deleted=False),
        ]
    else:
        anchor_rpr = _rpr_of(old_runs[0]) if old_runs else None
        ins_runs = [_make_text_run(anchor_rpr, op.new_text, deleted=False)]
    ins_el = _wrap_change("w:ins", next_id, *ins_runs)

    anchor_el = old_runs[0]._r if old_runs else None
    if anchor_el is not None:
        anchor_el.addprevious(del_el)
    else:
        p_el.append(del_el)
    del_el.addnext(ins_el)
    for r in old_runs:
        p_el.remove(r._r)


def _apply_insert_after(anchor: Paragraph, op: ResumeEditOp, next_id) -> None:
    clone_el = copy.deepcopy(anchor._p)
    clone = Paragraph(clone_el, anchor._parent)
    _mark_paragraph_mark(clone_el, "w:ins", next_id)

    clone_runs = [r for r in clone.runs if r.text.strip()]
    if op.bold_part is not None:
        bold_rpr, rest_rpr = _resolve_run_rprs(clone_runs)
        new_runs = [
            _make_text_run(bold_rpr, op.bold_part, deleted=False),
            _make_text_run(rest_rpr, op.new_text, deleted=False),
        ]
    else:
        fallback_rpr = _rpr_of(clone_runs[0]) if clone_runs else None
        new_runs = [_make_text_run(fallback_rpr, op.new_text, deleted=False)]

    for r in clone_runs:
        clone_el.remove(r._r)

    ins_el = _wrap_change("w:ins", next_id, *new_runs)
    clone_el.append(ins_el)

    anchor._p.addnext(clone_el)


def _apply_delete(paragraph: Paragraph, next_id) -> None:
    p_el = paragraph._p
    old_runs = [r for r in paragraph.runs if r.text.strip()]
    for r in old_runs:
        del_el = _wrap_change("w:del", next_id, _make_text_run(_rpr_of(r), r.text, deleted=True))
        r._r.addprevious(del_el)
        p_el.remove(r._r)
    _mark_paragraph_mark(p_el, "w:del", next_id)


def write_resume_edits(template_path: Path, ops: list[ResumeEditOp], output_path: Path) -> None:
    """Applies `ops` to a fresh copy of the user's real resume template as native Word tracked
    changes. Resolves every op's anchor against the template first and raises (writing nothing to
    `output_path`) if any anchor doesn't match exactly one non-title/contact paragraph -- never
    silently guesses a target."""
    doc = Document(str(template_path))
    _get_sect_pr(doc.element.body)

    by_text: dict[str, list[tuple[Paragraph, TemplateLine]]] = {}
    for p, line in _extract_lines_with_paragraphs(doc):
        by_text.setdefault(line.text, []).append((p, line))

    resolved: list[tuple[Paragraph, ResumeEditOp]] = []
    for op in ops:
        matches = by_text.get(op.anchor_text, [])
        if not matches:
            raise AnchorNotFoundError(f"no paragraph in the template matches: {op.anchor_text!r}")
        if len(matches) > 1:
            raise AmbiguousAnchorError(
                f"{len(matches)} paragraphs match {op.anchor_text!r}; anchor text must be unique"
            )
        paragraph, line = matches[0]
        if line.role in ("title", "contact"):
            raise DisallowedAnchorError(f"cannot edit the {line.role} line: {op.anchor_text!r}")
        resolved.append((paragraph, op))

    next_id = _next_change_id_counter(doc)
    for paragraph, op in resolved:
        if op.op == "replace":
            _apply_replace(paragraph, op, next_id)
        elif op.op == "insert_after":
            _apply_insert_after(paragraph, op, next_id)
        else:
            _apply_delete(paragraph, next_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))


# ---------------------------------------------------------------------------
# Cover letters: unchanged positional in-place replacement
# ---------------------------------------------------------------------------


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
