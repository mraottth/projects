from jobhunt.core.ats_detect import detect_embedded_ats


def test_greenhouse_embed_script():
    html = '<script src="https://boards.greenhouse.io/embed/job_board/js?for=bluerose"></script>'
    assert detect_embedded_ats(html, single_organization=False) == (
        "ats_greenhouse",
        {"board_token": "bluerose"},
    )


def test_greenhouse_embed_iframe_with_escaped_query():
    html = '<iframe src="https://boards.greenhouse.io/embed/job_board?b=x&amp;for=acme"></iframe>'
    assert detect_embedded_ats(html, single_organization=False) == ("ats_greenhouse", {"board_token": "acme"})


def test_ashby_and_lever_embeds():
    assert detect_embedded_ats(
        '<script src="https://jobs.ashbyhq.com/acme/embed?version=2"></script>', single_organization=False
    ) == ("ats_ashby", {"board_name": "acme"})
    assert detect_embedded_ats(
        'fetch("https://api.lever.co/v0/postings/acme?mode=json")', single_organization=False
    ) == ("ats_lever", {"company_slug": "acme"})


def test_plain_link_counts_only_for_a_single_organization():
    html = '<a href="https://jobs.lever.co/acme/123">Data Scientist</a>'
    assert detect_embedded_ats(html, single_organization=True) == ("ats_lever", {"company_slug": "acme"})
    assert detect_embedded_ats(html, single_organization=False) is None


def test_several_boards_means_a_listing_of_boards():
    html = (
        '<a href="https://boards.greenhouse.io/acme/jobs/1">A</a>'
        '<a href="https://jobs.lever.co/globex/2">B</a>'
    )
    assert detect_embedded_ats(html, single_organization=True) is None


def test_no_ats():
    assert detect_embedded_ats("<p>Email jobs@example.org</p>", single_organization=True) is None


def test_hosted_board_link_is_found():
    from jobhunt.core.ats_detect import detect_hosted_board

    html = '<a href="https://apply.workable.com/blue-state/">See all openings</a>'
    assert detect_hosted_board(html) == "https://apply.workable.com/blue-state/"
    assert detect_hosted_board('<a href="https://acme.bamboohr.com/careers/12">Job</a>') == (
        "https://acme.bamboohr.com/careers"
    )
    workday = "https://acme.wd5.myworkdayjobs.com/en-US/External/job/Analyst_123"
    assert (
        detect_hosted_board(f'<a href="{workday}">Job</a>')
        == "https://acme.wd5.myworkdayjobs.com/en-US/External"
    )


def test_no_or_several_hosted_boards_mean_none():
    from jobhunt.core.ats_detect import detect_hosted_board

    assert detect_hosted_board("<p>Email jobs@example.org</p>") is None
    two = '<a href="https://apply.workable.com/a/">A</a><a href="https://b.breezy.hr/">B</a>'
    assert detect_hosted_board(two) is None
