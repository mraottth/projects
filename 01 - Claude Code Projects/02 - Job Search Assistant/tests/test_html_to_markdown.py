from jobhunt.core.html_extract import html_to_markdown


def test_keeps_headings_lists_bold_links_and_paragraphs():
    html = (
        "<h2>About the role</h2><p>We need <strong>5+ years</strong> of Python, and a "
        '<a href="https://x.org/team">great team</a>.</p><h3>What you\'ll do:</h3>'
        "<ul><li>Build <b>models</b></li><li><p>Ship tools</p></li></ul>"
        "<ol><li>First</li><li>Second</li></ol><div>Line one<br>Line two</div>"
        "<script>track()</script><p>&nbsp;</p>"
    )
    assert html_to_markdown(html) == (
        "## About the role\n\n"
        "We need **5+ years** of Python, and a [great team](https://x.org/team).\n\n"
        "### What you'll do:\n\n"
        "- Build **models**\n- Ship tools\n\n"
        "1. First\n2. Second\n\n"
        "Line one\nLine two"
    )


def test_main_content_skips_page_chrome():
    page = (
        "<html><body><nav>Home Jobs</nav><aside>Related jobs</aside>"
        "<main><h1>Data Lead</h1><p>The job.</p></main><footer>(c)</footer></body></html>"
    )
    assert html_to_markdown(page, main_content=True) == "## Data Lead\n\nThe job."


def test_empty_input():
    assert html_to_markdown(None) == ""
    assert html_to_markdown("<p> </p>") == ""
