import pytest
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

from jobhunt.core.docx_writer import (
    AmbiguousAnchorError,
    AnchorNotFoundError,
    CoverLetterContent,
    DisallowedAnchorError,
    ResumeEditOp,
    TemplateError,
    count_cover_letter_slots,
    extract_addressable_lines,
    write_cover_letter_docx,
    write_resume_edits,
)


def _make_resume_template(path):
    doc = Document()
    doc.add_paragraph("JANE TESTER", style="Title")
    doc.add_paragraph("jane@example.com | 555-0000", style="Normal")

    doc.add_paragraph("WORK EXPERIENCE", style="Heading 1")
    doc.add_paragraph("", style="Normal")

    p1 = doc.add_paragraph(style="Normal")
    r1 = p1.add_run("ACME CORP | Senior Engineer")
    r1.bold = True
    r2 = p1.add_run(" | Jan 2022 - Present")
    r2.bold = False

    doc.add_paragraph("Shipped a thing.", style="List Paragraph")
    doc.add_paragraph("Shipped another thing.", style="List Paragraph")

    p2 = doc.add_paragraph(style="Normal")
    r3 = p2.add_run("OLD CORP | Engineer")
    r3.bold = True
    r4 = p2.add_run(" | 2018 - 2021")
    r4.bold = False

    doc.add_paragraph("Did an old thing.", style="List Paragraph")

    doc.add_paragraph("", style="Normal")
    doc.save(str(path))


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


def _body_paragraph_xml(path, text_fragment: str) -> str:
    from lxml import etree

    doc = Document(str(path))
    for p in doc.element.body.findall(qn("w:p")):
        if text_fragment in etree.tostring(p, encoding="unicode"):
            return etree.tostring(p, encoding="unicode")
    raise AssertionError(f"no paragraph containing {text_fragment!r} found")


# ---------------------------------------------------------------------------
# extract_addressable_lines
# ---------------------------------------------------------------------------


def test_extract_addressable_lines_tags_roles_and_skips_blanks(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)

    lines = extract_addressable_lines(template)
    roles_and_text = [(line.role, line.text) for line in lines]

    assert ("title", "JANE TESTER") in roles_and_text
    assert ("contact", "jane@example.com | 555-0000") in roles_and_text
    assert ("heading", "WORK EXPERIENCE") in roles_and_text
    assert ("job_header", "ACME CORP | Senior Engineer | Jan 2022 - Present") in roles_and_text
    assert ("bullet", "Shipped a thing.") in roles_and_text
    assert ("bullet", "Did an old thing.") in roles_and_text

    # Blank spacer paragraphs are never addressable.
    assert all(line.text.strip() for line in lines)


def test_extract_addressable_lines_tracks_section_and_under(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)

    lines = extract_addressable_lines(template)
    by_text = {line.text: line for line in lines}

    assert by_text["Shipped a thing."].section == "WORK EXPERIENCE"
    assert by_text["Shipped a thing."].under == "ACME CORP | Senior Engineer | Jan 2022 - Present"
    assert by_text["Did an old thing."].under == "OLD CORP | Engineer | 2018 - 2021"


# ---------------------------------------------------------------------------
# write_resume_edits -- replace
# ---------------------------------------------------------------------------


def test_replace_touches_only_the_targeted_paragraph(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    before_other_bullet = _body_paragraph_xml(template, "Shipped another thing.")
    before_job_header = _body_paragraph_xml(template, "ACME CORP")

    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [ResumeEditOp(op="replace", anchor_text="Shipped a thing.", new_text="Shipped a tailored thing.")],
        output,
    )

    after_other_bullet = _body_paragraph_xml(output, "Shipped another thing.")
    after_job_header = _body_paragraph_xml(output, "ACME CORP")
    assert after_other_bullet == before_other_bullet
    assert after_job_header == before_job_header


def test_replace_produces_real_tracked_change_xml(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [ResumeEditOp(op="replace", anchor_text="Shipped a thing.", new_text="Shipped a tailored thing.")],
        output,
    )

    xml = _body_paragraph_xml(output, "Shipped a tailored thing.")
    assert "<w:del " in xml and 'w:id="' in xml and 'w:author="Job Search Assistant"' in xml
    assert "<w:delText" in xml and "Shipped a thing." in xml
    assert "<w:ins " in xml
    assert "Shipped a tailored thing." in xml

    # python-docx's flattened .text only sees direct <w:r> children, not ones wrapped in
    # <w:ins>/<w:del> -- confirms the tracked-change structure, not just updated flat text.
    doc = Document(str(output))
    edited = next(p for p in doc.paragraphs if "Shipped a thing" in p.text or not p.text)
    assert "Shipped a tailored thing." not in edited.text


def test_replace_on_job_header_preserves_bold_split(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [
            ResumeEditOp(
                op="replace",
                anchor_text="ACME CORP | Senior Engineer | Jan 2022 - Present",
                new_text=" | Jan 2022 - Present",
                bold_part="ACME CORP | Staff Engineer",
            )
        ],
        output,
    )
    xml = _body_paragraph_xml(output, "ACME CORP | Staff Engineer")
    assert "<w:b/>" in xml  # the new bold run
    assert "ACME CORP | Staff Engineer" in xml
    assert " | Jan 2022 - Present" in xml


# ---------------------------------------------------------------------------
# write_resume_edits -- insert_after
# ---------------------------------------------------------------------------


def test_insert_after_clones_the_correct_anchors_formatting(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    doc = Document(str(template))
    # Give the two jobs' bullets visibly different indentation, as a stand-in for "two distinct
    # list configurations" -- the regression this guards against is a new bullet picking up some
    # OTHER paragraph's formatting instead of its own anchor's.
    for p in doc.paragraphs:
        if p.text == "Shipped a thing.":
            p.paragraph_format.left_indent = None
            p_pr = p._p.get_or_add_pPr()
            ind = p_pr.makeelement(qn("w:ind"), {qn("w:left"): "720"})
            p_pr.append(ind)
        if p.text == "Did an old thing.":
            p_pr = p._p.get_or_add_pPr()
            ind = p_pr.makeelement(qn("w:ind"), {qn("w:left"): "1440"})
            p_pr.append(ind)
    doc.save(str(template))

    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [ResumeEditOp(op="insert_after", anchor_text="Shipped a thing.", new_text="Shipped a new thing.")],
        output,
    )

    xml = _body_paragraph_xml(output, "Shipped a new thing.")
    assert 'w:left="720"' in xml
    assert 'w:left="1440"' not in xml


def test_insert_after_marks_run_and_paragraph_mark_as_inserted(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [ResumeEditOp(op="insert_after", anchor_text="Shipped a thing.", new_text="Shipped a new thing.")],
        output,
    )

    xml = _body_paragraph_xml(output, "Shipped a new thing.")
    assert "<w:ins " in xml
    # The paragraph mark itself (inside pPr/rPr) is marked inserted, not just the run content.
    assert "<w:pPr>" in xml
    assert xml.index("<w:pPr>") < xml.index("Shipped a new thing.")
    assert xml.count("<w:ins ") >= 2  # one for the run, one for the paragraph mark


def test_insert_after_preserves_the_anchor_paragraph(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    before = _body_paragraph_xml(template, "Shipped a thing.")

    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [ResumeEditOp(op="insert_after", anchor_text="Shipped a thing.", new_text="Shipped a new thing.")],
        output,
    )
    after = _body_paragraph_xml(output, ">Shipped a thing.<")
    assert after == before


# ---------------------------------------------------------------------------
# write_resume_edits -- delete
# ---------------------------------------------------------------------------


def test_delete_marks_run_and_paragraph_mark_as_deleted_but_keeps_the_element(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    output = tmp_path / "out.docx"
    write_resume_edits(
        template, [ResumeEditOp(op="delete", anchor_text="Shipped another thing.")], output
    )

    xml = _body_paragraph_xml(output, "Shipped another thing.")
    assert "<w:del " in xml
    assert "<w:delText" in xml
    # The paragraph mark is also marked deleted (inside pPr/rPr), not just the text.
    assert xml.count("<w:del ") >= 2


# ---------------------------------------------------------------------------
# write_resume_edits -- error cases and font/size inheritance
# ---------------------------------------------------------------------------


def test_anchor_not_found_raises(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    with pytest.raises(AnchorNotFoundError):
        write_resume_edits(
            template,
            [ResumeEditOp(op="replace", anchor_text="NOT IN THE DOC", new_text="x")],
            tmp_path / "out.docx",
        )


def test_ambiguous_anchor_raises_without_picking_one(tmp_path):
    template = tmp_path / "template.docx"
    doc = Document()
    doc.add_paragraph("JANE TESTER", style="Title")
    doc.add_paragraph("jane@example.com", style="Normal")
    doc.add_paragraph("WORK EXPERIENCE", style="Heading 1")
    doc.add_paragraph("Same bullet text.", style="List Paragraph")
    doc.add_paragraph("Same bullet text.", style="List Paragraph")
    doc.save(str(template))

    with pytest.raises(AmbiguousAnchorError):
        write_resume_edits(
            template, [ResumeEditOp(op="delete", anchor_text="Same bullet text.")], tmp_path / "out.docx"
        )


@pytest.mark.parametrize("anchor_text", ["JANE TESTER", "jane@example.com | 555-0000"])
def test_disallowed_anchor_raises_for_title_and_contact(tmp_path, anchor_text):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    with pytest.raises(DisallowedAnchorError):
        write_resume_edits(
            template,
            [ResumeEditOp(op="replace", anchor_text=anchor_text, new_text="x")],
            tmp_path / "out.docx",
        )


def test_new_content_inherits_font_from_its_own_anchor_not_a_global_default(tmp_path):
    template = tmp_path / "template.docx"
    doc = Document()
    doc.add_paragraph("JANE TESTER", style="Title")
    doc.add_paragraph("jane@example.com", style="Normal")
    doc.add_paragraph("WORK EXPERIENCE", style="Heading 1")

    p1 = doc.add_paragraph("Bullet in Times.", style="List Paragraph")
    p1.runs[0].font.name = "Times New Roman"
    p1.runs[0].font.size = Pt(9)

    p2 = doc.add_paragraph("Bullet in Courier.", style="List Paragraph")
    p2.runs[0].font.name = "Courier New"
    p2.runs[0].font.size = Pt(14)

    doc.save(str(template))

    output = tmp_path / "out.docx"
    write_resume_edits(
        template,
        [
            ResumeEditOp(op="replace", anchor_text="Bullet in Times.", new_text="Replaced Times bullet."),
            ResumeEditOp(op="replace", anchor_text="Bullet in Courier.", new_text="Replaced Courier bullet."),
        ],
        output,
    )

    times_xml = _body_paragraph_xml(output, "Replaced Times bullet.")
    courier_xml = _body_paragraph_xml(output, "Replaced Courier bullet.")
    assert "Times New Roman" in times_xml
    assert "Courier New" not in times_xml
    assert 'w:val="18"' in times_xml  # 9pt = 18 half-points
    assert "Courier New" in courier_xml
    assert "Times New Roman" not in courier_xml
    assert 'w:val="28"' in courier_xml  # 14pt = 28 half-points


def test_invalid_op_in_a_plan_writes_nothing(tmp_path):
    template = tmp_path / "template.docx"
    _make_resume_template(template)
    output = tmp_path / "out.docx"

    with pytest.raises(AnchorNotFoundError):
        write_resume_edits(
            template,
            [
                ResumeEditOp(
                    op="replace", anchor_text="Shipped a thing.", new_text="Shipped a tailored thing."
                ),
                ResumeEditOp(op="delete", anchor_text="NOT IN THE DOC"),
            ],
            output,
        )
    assert not output.exists()


def test_write_resume_edits_raises_on_unrecognizable_template(tmp_path):
    doc = Document()
    doc.add_paragraph("just some plain text")
    # Remove sectPr to simulate a structurally-unrecognizable template.
    body = doc.element.body
    sect_pr = body.find(qn("w:sectPr"))
    if sect_pr is not None:
        body.remove(sect_pr)
    template = tmp_path / "bad_template.docx"
    doc.save(str(template))

    with pytest.raises(TemplateError):
        write_resume_edits(
            template,
            [ResumeEditOp(op="replace", anchor_text="just some plain text", new_text="x")],
            tmp_path / "out.docx",
        )


# ---------------------------------------------------------------------------
# write_cover_letter_docx -- unchanged behavior
# ---------------------------------------------------------------------------


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
