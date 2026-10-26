import pytest
from docx import Document
from docx.oxml.ns import qn

from jobhunt.core.docx_writer import (
    CoverLetterContent,
    ResumeContent,
    ResumeLine,
    ResumeSection,
    TemplateError,
    count_cover_letter_slots,
    write_cover_letter_docx,
    write_resume_docx,
)


def _make_resume_template(path):
    doc = Document()
    doc.add_paragraph("TEMPLATE NAME", style="Title")
    doc.add_paragraph("email@example.com | 555-0000", style="Normal")
    doc.add_paragraph("WORK EXPERIENCE", style="Heading 1")
    doc.add_paragraph("", style="Normal")

    p = doc.add_paragraph(style="Normal")
    r1 = p.add_run("OLD CORP | Old Title")
    r1.bold = True
    r2 = p.add_run(" | Jan 2020 - Dec 2021")
    r2.bold = False

    desc = doc.add_paragraph("Old corp does old things.", style="Quote")

    bullet = doc.add_paragraph("Did an old thing.", style="List Paragraph")

    doc.add_paragraph("", style="Normal")
    doc.save(str(path))
    return desc, bullet  # unused, just confirms these styles exist


def _make_cover_letter_template(path, with_hyperlink=False):
    doc = Document()
    doc.add_paragraph("", style="Normal")
    doc.add_paragraph("To Whom It May Concern:", style="Normal")
    doc.add_paragraph("", style="Normal")

    p = doc.add_paragraph(style="Normal")
    p.add_run("My name is Old Person and I am applying.")
    if with_hyperlink:
        # Simulate a real-world hyperlinked word inside a paragraph (python-docx has no public
        # API for this, so inject the XML directly -- matches the actual structure found in a
        # real cover letter template during development).
        hyperlink = p._p.makeelement(qn("w:hyperlink"), {})
        run = p.add_run("hyperlinked-word")
        hyperlink.append(run._r)
        run._r.getparent().remove(run._r)
        p._p.append(hyperlink)

    doc.add_paragraph("", style="Normal")
    doc.add_paragraph("Sincerely,", style="Normal")
    doc.add_paragraph("", style="Normal")
    doc.add_paragraph("-Old Person", style="Normal")
    doc.save(str(path))


def test_write_resume_docx_preserves_margins_and_styles(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)

    content = ResumeContent(
        name="JANE TESTER",
        contact_line="jane@example.com | 555-1234",
        sections=[
            ResumeSection(
                heading="WORK EXPERIENCE",
                lines=[
                    ResumeLine(
                        kind="normal", bold_part="ACME CORP | Senior Engineer", text=" | 2024 - Present"
                    ),
                    ResumeLine(kind="description", text="Acme builds things."),
                    ResumeLine(kind="bullet", text="Shipped a thing."),
                    ResumeLine(kind="bullet", text="Shipped another thing."),
                ],
            )
        ],
    )
    output = tmp_path / "out.docx"
    write_resume_docx(template, content, output)

    doc = Document(str(output))
    orig = Document(str(template))
    assert doc.sections[0].top_margin == orig.sections[0].top_margin
    assert doc.sections[0].left_margin == orig.sections[0].left_margin

    texts_and_styles = [(p.text, p.style.name) for p in doc.paragraphs if p.text.strip()]
    assert ("JANE TESTER", "Title") in texts_and_styles
    assert ("jane@example.com | 555-1234", "Normal") in texts_and_styles
    assert ("WORK EXPERIENCE", "Heading 1") in texts_and_styles
    assert ("Acme builds things.", "Quote") in texts_and_styles
    assert ("Shipped a thing.", "List Paragraph") in texts_and_styles

    header_para = next(p for p in doc.paragraphs if "ACME CORP" in p.text)
    assert header_para.runs[0].text == "ACME CORP | Senior Engineer"
    assert header_para.runs[0].bold is True
    assert header_para.runs[1].text == " | 2024 - Present"
    assert header_para.runs[1].bold is False


def test_write_resume_docx_forces_calibri_regardless_of_template_font(tmp_path):
    # Real templates are often inconsistent about font (e.g. an explicit Times New Roman override
    # on the "Normal" style while headings fall through to the theme's Calibri with no run-level
    # override at all) -- generated text should come out uniformly Calibri regardless.
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    doc = Document(str(template))
    for p in doc.paragraphs:
        for r in p.runs:
            r.font.name = "Times New Roman"
    doc.save(str(template))

    content = ResumeContent(
        name="JANE TESTER",
        contact_line="jane@example.com",
        sections=[
            ResumeSection(
                heading="WORK EXPERIENCE",
                lines=[
                    ResumeLine(kind="normal", bold_part="ACME | Engineer", text=" | 2024"),
                    ResumeLine(kind="bullet", text="Shipped a thing."),
                ],
            )
        ],
    )
    output = tmp_path / "out.docx"
    write_resume_docx(template, content, output)

    out_doc = Document(str(output))
    for p in out_doc.paragraphs:
        for r in p.runs:
            if r.text.strip():
                assert r.font.name == "Calibri", f"expected Calibri, got {r.font.name!r} for {r.text!r}"


def test_write_resume_docx_raises_on_unrecognizable_template(tmp_path):
    doc = Document()
    doc.add_paragraph("just some plain text")
    template = tmp_path / "bad_template.docx"
    doc.save(str(template))

    content = ResumeContent(name="X", contact_line="x", sections=[])
    with pytest.raises(TemplateError):
        write_resume_docx(template, content, tmp_path / "out.docx")


def test_write_cover_letter_docx_replaces_text_positionally(tmp_path):
    template = tmp_path / "cl_template.docx"
    _make_cover_letter_template(template)

    content = CoverLetterContent(
        paragraphs=[
            "To Whom It May Concern:",
            "My name is Jane Tester and I am applying for this role.",
            "Sincerely,",
            "-Jane",
        ]
    )
    output = tmp_path / "out_cl.docx"
    write_cover_letter_docx(template, content, output)

    doc = Document(str(output))
    non_blank = [p.text for p in doc.paragraphs if p.text.strip()]
    assert non_blank == [
        "To Whom It May Concern:",
        "My name is Jane Tester and I am applying for this role.",
        "Sincerely,",
        "-Jane",
    ]


def test_write_cover_letter_docx_clears_unused_slots_instead_of_leaving_stale_text(tmp_path):
    template = tmp_path / "cl_template.docx"
    _make_cover_letter_template(template)  # 4 non-blank paragraphs

    content = CoverLetterContent(paragraphs=["To Whom It May Concern:", "Short letter."])
    output = tmp_path / "out_cl.docx"
    write_cover_letter_docx(template, content, output)

    doc = Document(str(output))
    non_blank = [p.text for p in doc.paragraphs if p.text.strip()]
    assert non_blank == ["To Whom It May Concern:", "Short letter."]
    assert "Old Person" not in " ".join(p.text for p in doc.paragraphs)


def test_write_cover_letter_docx_appends_extra_paragraphs_beyond_template(tmp_path):
    template = tmp_path / "cl_template.docx"
    _make_cover_letter_template(template)  # 4 non-blank paragraphs

    content = CoverLetterContent(
        paragraphs=["Para 1", "Para 2", "Para 3", "Para 4", "Extra para 5", "Extra para 6"]
    )
    output = tmp_path / "out_cl.docx"
    write_cover_letter_docx(template, content, output)

    doc = Document(str(output))
    non_blank = [p.text for p in doc.paragraphs if p.text.strip()]
    assert non_blank == ["Para 1", "Para 2", "Para 3", "Para 4", "Extra para 5", "Extra para 6"]


def test_write_cover_letter_docx_strips_hyperlinked_text_not_just_runs(tmp_path):
    # Regression: a word wrapped in <w:hyperlink> isn't enumerated by python-docx's `.runs`, so
    # clearing only `.runs` leaves it behind, leaking it into the new paragraph's text -- caught
    # live against a real cover letter template with a hyperlinked word ("thesis").
    template = tmp_path / "cl_template.docx"
    _make_cover_letter_template(template, with_hyperlink=True)

    content = CoverLetterContent(
        paragraphs=[
            "To Whom It May Concern:",
            "Clean replacement text.",
            "Sincerely,",
            "-Jane",
        ]
    )
    output = tmp_path / "out_cl.docx"
    write_cover_letter_docx(template, content, output)

    doc = Document(str(output))
    full_text = " ".join(p.text for p in doc.paragraphs)
    assert "hyperlinked-word" not in full_text
    assert "Clean replacement text." in full_text


def test_count_cover_letter_slots(tmp_path):
    template = tmp_path / "cl_template.docx"
    _make_cover_letter_template(template)
    assert count_cover_letter_slots(template) == 4
