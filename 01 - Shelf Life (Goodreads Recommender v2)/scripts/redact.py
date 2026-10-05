"""Scrub details that shouldn't be on the public site from the prompt log and the Changelog page.

Used by export_prompts.py (prompts/*.md/.json) and build_changelog.py (commit text). Two kinds of rules:
- built in (below): Claude Code session links and ids, and home-directory paths that contain a username;
- local literals in `.redact.local.json` next to this folder's Makefile, git-ignored so the values themselves
  are never committed: a JSON list of [text, replacement] pairs, e.g. [["my-gcp-project", "<gcp-project>"]].
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
