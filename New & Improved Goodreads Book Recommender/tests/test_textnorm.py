from goodrec.core.textnorm import author_key, clean_isbn, is_boxset, isbn10_to_13, parse_series, titlekey


def test_parse_series():
    assert parse_series("The Hunger Games (The Hunger Games, #1)") == (
        "The Hunger Games", "The Hunger Games", 1.0, False)
    assert parse_series("Catching Fire (The Hunger Games, #2)")[2] == 2.0
    assert parse_series("Harry Potter Boxset (Harry Potter, #1-7)")[3] is True
    assert parse_series("The Assassin's Blade (Throne of Glass, #0.1-0.5)")[3] is True
    assert parse_series("Saga, Volume 3")[2] == 3.0
    assert parse_series("Gone Girl") == ("Gone Girl", None, None, False)


def test_is_boxset():
    assert is_boxset("Harry Potter Boxset (Harry Potter, #1-7)")
    assert is_boxset("The Chronicles of Narnia Box Set")
    assert not is_boxset("The Hunger Games (The Hunger Games, #1)")


def test_titlekey():
    assert titlekey("The Hunger Games (The Hunger Games, #1)") == "hunger games"
    assert titlekey("Sapiens: A Brief History of Humankind") == "sapiens"
    assert titlekey("Pride & Prejudice") == "pride and prejudice"
    assert titlekey("Les Misérables") == "les miserables"


def test_author_key():
    assert author_key("J.K. Rowling") == "rowling"
    assert author_key("Tolkien, J.R.R.") == "tolkien"
    assert author_key("Martin Luther King Jr.") == "king"


def test_isbn():
    assert clean_isbn('="0374104093"') == "0374104093"
    assert clean_isbn('=""') == ""
    assert clean_isbn("978-0-316-76948-8") == "9780316769488"
    assert isbn10_to_13("0316769487") == "9780316769488"
    assert clean_isbn("03167X9487") == ""
    assert clean_isbn("043942089X") == "043942089X"
    assert isbn10_to_13("03167X9487") == ""
