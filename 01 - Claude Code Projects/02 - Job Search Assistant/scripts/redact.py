"""Scrub details that shouldn't show up verbatim in the prompt log or the Timeline page.

The app itself is private (IAP-locked), but its code and prompt log are public in the
mraottth/projects repo (since 2026-11-06), so this strips local file paths with the username,
Claude Code session links and real email addresses (deploy commands carry the admin's sign-in
address). Used by export_prompts.py (prompts/*.md/.json) and build_changelog.py (commit text).

Local literals (API keys, real company names you'd rather not have sitting in a plaintext log,
etc.) go in `.redact.local.json` next to this folder's Makefile, git-ignored so the values
themselves are never committed: a JSON list of [text, replacement] pairs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / ".redact.local.json"

BUILTIN = [
    (re.compile(r"https://claude\.ai/code/session_\w+"), "<Claude Code session link>"),
    (re.compile(r"\bsession_[A-Za-z0-9]{16,}\b"), "<session id>"),
    (re.compile(r"-Users-[A-Za-z0-9_.]+-"), "-Users-<user>-"),
    (re.compile(r"/(?:Users|home)/[A-Za-z0-9_.]+/"), "~/"),
    # Any email address but the placeholders the docs and tests use (example domains, you@gmail.com)
    # and the commit-trailer, no-reply and SSH-remote ones (anthropic.com, github.com).
    (re.compile(r"\b(?!you@gmail\.com\b)[A-Za-z0-9._%+-]+@(?!(?:[a-z0-9-]+\.)*(?:example\.(?:com|org|net)|"
                r"anthropic\.com|github\.com)\b)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
     "you@example.com"),
]

_local: list[tuple[str, str]] | None = None


def local_rules() -> list[tuple[str, str]]:
    global _local
    if _local is None:
        _local = [tuple(p) for p in json.loads(LOCAL.read_text(encoding="utf-8"))] if LOCAL.exists() else []
    return _local


def redact(text: str) -> str:
    for old, new in local_rules():
        text = text.replace(old, new)
    for pat, new in BUILTIN:
        text = pat.sub(new, text)
    return text
