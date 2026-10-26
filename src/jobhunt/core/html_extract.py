from bs4 import BeautifulSoup

STRIP_TAGS = ("script", "style", "nav", "footer", "header", "svg", "noscript")


def extract_readable_text(html: str, max_chars: int = 20_000) -> str:
    """Strip boilerplate/script/style and return visible text, capped to bound LLM input cost."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    collapsed = "\n".join(lines)
    return collapsed[:max_chars]
