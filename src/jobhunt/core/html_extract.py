import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

STRIP_TAGS = ("script", "style", "nav", "footer", "header", "svg", "noscript")


def extract_readable_text(html: str, max_chars: int = 20_000, *, base_url: str | None = None) -> str:
    """Strip boilerplate/script/style and return visible text, capped to bound LLM input cost.

    With `base_url`, each link's href is inlined as `text [href]` (resolved to an absolute URL)
    before stripping tags, so a listing page's per-posting URLs survive into the plain text for
    the LLM to associate with each posting -- without this, get_text() would just discard them.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    if base_url:
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if href.startswith(("http://", "https://")):
                a.replace_with(f"{a.get_text(strip=True)} [{href}]")

    text = soup.get_text("\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    collapsed = "\n".join(lines)
    return collapsed[:max_chars]


# --- HTML -> light Markdown, for job descriptions -------------------------------------------
# extract_readable_text() flattens a description to one line per text node, so headings, bullets
# and paragraphs vanished and a bolded phrase landed on a line of its own (live descriptions read
# as run-together blocks). This keeps that structure, in the subset of Markdown the frontend's
# components/Markdown.tsx renders: "## " headings, "- "/"1. " lists, **bold**, *italic*, links.
# Pure BeautifulSoup, no Claude call.

_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "aside",
    "blockquote",
    "pre",
    "table",
    "tbody",
    "thead",
    "tr",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
}
_HEADINGS = {"h1": "##", "h2": "##", "h3": "###", "h4": "####", "h5": "####", "h6": "####"}
_SPACE = re.compile(r"\s+")


def _inline(node) -> str:
    if isinstance(node, NavigableString):
        return "" if isinstance(node, Comment) else _SPACE.sub(" ", str(node))
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return "\n"
    inner = "".join(_inline(child) for child in node.children)
    stripped = inner.strip()
    if not stripped or "\n" in stripped:
        return inner
    # Markers hug the text; the whitespace that was around it inside the tag stays outside them.
    lead = " " if inner[:1].isspace() else ""
    trail = " " if inner[-1:].isspace() else ""
    if node.name in ("strong", "b") and "**" not in stripped:
        return f"{lead}**{stripped}**{trail}"
    if node.name in ("em", "i") and "*" not in stripped:
        return f"{lead}*{stripped}*{trail}"
    if node.name == "a":
        href = node.get("href") or ""
        if href.startswith(("http://", "https://")) and stripped != href:
            return f"{lead}[{stripped}]({href}){trail}"
    return inner


def _clean_line(text: str) -> str:
    lines = [_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _blocks(node, out: list[str]) -> None:
    """Appends Markdown blocks for node's children to out."""
    buffer: list[str] = []

    def flush() -> None:
        text = _clean_line("".join(buffer))
        if text:
            out.append(text)
        buffer.clear()

    for child in node.children:
        if isinstance(child, Tag) and child.name in STRIP_TAGS:
            continue
        if not isinstance(child, Tag) or child.name not in _BLOCK_TAGS:
            buffer.append(_inline(child))
            continue
        flush()
        name = child.name
        if name in _HEADINGS:
            text = _clean_line(_inline(child)).replace("\n", " ").replace("**", "")
            if text:
                out.append(f"{_HEADINGS[name]} {text}")
        elif name in ("ul", "ol"):
            items = []
            for i, li in enumerate(child.find_all("li", recursive=False), start=1):
                parts: list[str] = []
                _blocks(li, parts)
                text = " ".join(p.replace("\n", " ") for p in parts if p).strip()
                text = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", text)
                if text:
                    items.append(f"{i}. {text}" if name == "ol" else f"- {text}")
            if items:
                out.append("\n".join(items))
        elif name == "tr":
            cells = [_clean_line(_inline(c)).replace("\n", " ") for c in child.find_all(["td", "th"])]
            if any(cells):
                out.append(" · ".join(c for c in cells if c))
        elif name == "hr":
            continue
        else:
            _blocks(child, out)
    flush()


def html_to_markdown(html: str | None, max_chars: int = 20_000, *, main_content: bool = False) -> str:
    """A job description's HTML as light Markdown (see the comment above). `main_content` is for a
    whole web page (a job's own page on a careers site): it reads only the page's <main> or
    <article> when there is one, leaving sidebars and related-jobs lists out."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    root = soup.body or soup
    if main_content:
        root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article") or root
    blocks: list[str] = []
    _blocks(root, blocks)
    return "\n\n".join(blocks)[:max_chars].strip()
