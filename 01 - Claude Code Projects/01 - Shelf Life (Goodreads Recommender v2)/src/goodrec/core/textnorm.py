"""Title/author/ISBN normalization shared by the pipeline and CSV import.

Goodreads titles embed series info as a trailing "(Series Name, #N)". Parsing
it generalizes the old app's regex filters (`#2+`, `Vol. N`, `Volume N`, `#1-`).
"""

import re
import unicodedata

# "(Series, #N)", "(Series, #1-3)", and variants with trailing text such as "#3: Part 2 of 2" or
# "#0.4, 0.5, 2.5" (collections). A trailing number marks a range/collection or a split edition.
_SERIES_RE = re.compile(
    r"\s*\((?P<name>[^()#]*?),?\s*#(?P<pos>\d+(?:\.\d+)?)(?:\s*[-\u2013]\s*(?P<end>\d+(?:\.\d+)?))?"
    r"(?P<rest>[^()]*)\)\s*$"
)
# Trailing text that makes it a collection ("#0.4, 0.5") or a split edition ("#3: Part 2 of 2"),
# as opposed to a second series ("#40, Moist von Lipwig #3").
_RANGE_REST_RE = re.compile(r"^\s*(?:[,;&]\s*\d|[:,]?\s*part\s+\d)", re.I)
_VOLUME_RE = re.compile(r"\b(?:vol\.?|volume)\s*(\d+)\b", re.I)
_BOXSET_RE = re.compile(
    r"\b(box(?:ed)?\s*set|boxset|omnibus|collection\b.*#|complete\s+series|books?\s+\d+\s*-\s*\d+)", re.I
)
_ARTICLES_RE = re.compile(r"^(the|a|an)\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def parse_series(title: str) -> tuple[str, str | None, float | None, bool]:
    """Split a Goodreads title into (base_title, series_name, series_pos, is_range).

    "The Hunger Games (The Hunger Games, #1)" -> ("The Hunger Games", "The Hunger Games", 1.0, False)
    "Harry Potter Boxset (Harry Potter, #1-7)" -> (..., "Harry Potter", 1.0, True)
    """
    m = _SERIES_RE.search(title or "")
    if m:
        name = m.group("name").strip() or None
        is_range = m.group("end") is not None or bool(_RANGE_REST_RE.match(m.group("rest") or ""))
        return title[: m.start()].strip(), name, float(m.group("pos")), is_range
    v = _VOLUME_RE.search(title or "")
    if v:
        return (title or "").strip(), None, float(v.group(1)), False
    return (title or "").strip(), None, None, False


def is_boxset(title: str) -> bool:
    _, _, _, is_range = parse_series(title)
    return is_range or bool(_BOXSET_RE.search(title or ""))


def titlekey(title: str) -> str:
    """Aggressive normalization for fuzzy title matching across editions/exports."""
    base = parse_series(title)[0]
    base = ascii_fold(base).lower()
    base = base.split(":")[0]
    base = base.replace("&", " and ")
    base = _NON_ALNUM_RE.sub(" ", base)
    base = _WS_RE.sub(" ", base).strip()
    return _ARTICLES_RE.sub("", base)


def author_key(name: str) -> str:
    """Last-name key: 'J.K. Rowling' -> 'rowling', 'Tolkien, J.R.R.' -> 'tolkien'."""
    if not name:
        return ""
    name = ascii_fold(name).lower()
    if "," in name:
        last = name.split(",")[0]
    else:
        parts = _NON_ALNUM_RE.sub(" ", name).split()
        suffixes = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}
        while len(parts) > 1 and parts[-1] in suffixes:
            parts.pop()
        last = parts[-1] if parts else ""
    return _NON_ALNUM_RE.sub("", last)


def clean_isbn(s) -> str:
    """Strip Excel quoting (="0374104093") and punctuation; '' if not a plausible ISBN."""
    if s is None:
        return ""
    s = re.sub(r"[^0-9Xx]", "", str(s)).upper()
    if len(s) == 13 and s.isdigit():
        return s
    if len(s) == 10 and s[:9].isdigit():
        return s
    return ""


def isbn10_to_13(isbn10: str) -> str:
    if len(isbn10) != 10 or not isbn10[:9].isdigit():
        return ""
    core = "978" + isbn10[:9]
    total = sum((1 if i % 2 == 0 else 3) * int(d) for i, d in enumerate(core))
    return core + str((10 - total % 10) % 10)
