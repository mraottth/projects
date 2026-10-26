from urllib.parse import urljoin

from bs4 import BeautifulSoup

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
