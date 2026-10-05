"""Export a Claude Code session transcript to prompts/: the user's prompts and Claude's final replies.

Each session becomes two files in prompts/:
  <YYYY-MM-DD>_<session8>.md    readable log: every prompt, then Claude's final reply for that turn
  <YYYY-MM-DD>_<session8>.json  the same entries, for scripts/build_changelog.py

Kept: typed prompts, plus input given through Claude's tools (answers to its questions, comments on or
rejections of a plan), the questions Claude asked and the plans the user approved. Dropped: tool calls and
output, Claude's interim status text, system reminders, compaction summaries, editor notices, and prompts
that failed with an API error (e.g. low credit balance), which are counted in the file header. Pasted files and images become short placeholders, and local paths
are shortened. Re-running regenerates identical files, and files are only written when they changed.

  python3 scripts/export_prompts.py ~/.claude/projects/<project>/<session>.jsonl [...]
  python3 scripts/export_prompts.py --from-hook      # Claude Code Stop hook: reads {"transcript_path": ...} on stdin
"""

from __future__ import annotations  # runs under the system python3 (3.9) from the Claude Code hook

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import redact  # noqa: E402  (scrubs details that shouldn't be public; scripts/redact.py)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prompts"
REPO_PREFIXES = [(str(ROOT.parent) + "/", ""), (str(Path.home()) + "/", "~/")]
SCRATCH = re.compile(r"/private/tmp/claude-\d+/[^\s`'\")]*?/scratchpad")
DROP_TEXT = ("<system-reminder", "<ide_", "<command-", "<local-command", "<task-notification", "Caveat:",
             "[Request interrupted", "[Image: original")

# Input the user gave through a tool (AskUserQuestion / ExitPlanMode), keyed by how the tool result starts.
TOOL_INPUTS = [
    (re.compile(r"(?:Your questions have been answered|User has answered your questions?|The user answered):?\s*(.*)", re.S),
     "answer"),
    (re.compile(r"User chose to stay in plan mode and continue planning\s*(?:Comments on the plan:\s*)?(.*)", re.S), "plan-comment"),
    (re.compile(r"The user doesn't want to proceed with this tool use\..*?(?:reason for the rejection:\s*)(.*)", re.S), "plan-feedback"),
]
TOOL_INPUT_TAIL = re.compile(r"\s*(?:You can now continue with these answers in mind\.|Read the answers carefully.*)$", re.S)
API_ERROR = re.compile(r"^(Credit balance is too low|API Error|Invalid API key|Prompt is too long)")
APPROVED_PLAN = re.compile(r"User has approved your plan\..*?## Approved Plan:\s*(.*)", re.S)
KIND_LABEL = {"prompt": "Prompt", "answer": "Answer to Claude's question", "plan-comment": "Comment on Claude's plan",
              "plan-feedback": "Feedback on Claude's plan"}


def clean(text: str) -> str:
    for a, b in REPO_PREFIXES:
        text = text.replace(a, b)
    return SCRATCH.sub("<scratchpad>", text).strip()


def user_text(content) -> str | None:
    """The typed part of a user message, with attachments as placeholders; None if it isn't a real prompt."""
    if isinstance(content, str):
        parts = [content]
    else:
        parts = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(block.get("text", ""))
            elif kind == "image":
                parts.append("[screenshot]")
            elif kind == "document":
                data = (block.get("source") or {}).get("data") or ""
                parts.append(f"[attached file: {block.get('title') or 'document'}, {data.count(chr(10)) + 1} lines]")
    keep = [p for p in parts if p.strip() and not p.lstrip().startswith(DROP_TEXT)]
    keep = [re.sub(r"<pasted_content[^>]*>.*?</pasted_content[^>]*>", "[pasted text]", p, flags=re.S) for p in keep]
    text = "\n\n".join(keep).strip()
    return text or None


def tool_input(content) -> tuple[str, str] | None:
    """(kind, text) if this tool result carries input the user typed into a Claude tool dialog."""
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        body = block.get("content")
        body = body if isinstance(body, str) else " ".join(b.get("text", "") for b in body or [] if isinstance(b, dict))
        for pattern, kind in TOOL_INPUTS:
            m = pattern.match(body.strip())
            if m:
                text = TOOL_INPUT_TAIL.sub("", m.group(1)).strip()
                if text:
                    return kind, text
    return None


def parse(transcript: Path) -> list[dict]:
    entries: list[dict] = []
    reply: list[str] = []          # assistant text since the last tool call (= the final reply when the turn ends)

    def close():
        # The turn's final reply belongs to the prompt that started it, not to messages queued during the turn.
        for e in reversed(entries):
            if not e.get("queued"):
                if e["reply"] is None:
                    e["reply"] = clean("\n\n".join(reply)) or None
                break

    with open(transcript, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
                continue
            kind = d.get("type")
            msg = d.get("message") or {}
            content = msg.get("content")
            if kind == "user":
                approved = approved_plan(content)
                if approved and entries:
                    entries[-1]["plan"] = clean(approved)
                ti = tool_input(content)
                has_tool_result = isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
                text = None if has_tool_result else user_text(content)
                if text and text.startswith("This session is being continued from a previous conversation"):
                    continue
                if ti or text:
                    close()
                    reply = []
                    k, t = ti if ti else ("prompt", text)
                    entries.append({"time": d.get("timestamp"), "kind": k, "text": clean(t), "reply": None})
            elif kind == "attachment" and (d.get("attachment") or {}).get("type") == "queued_command" and entries:
                # A message the user sent while Claude was working (shown mid-turn). Task notifications and other
                # system text are dropped by user_text().
                text = user_text((d.get("attachment") or {}).get("prompt"))
                if text:
                    entries.append({"time": d.get("timestamp"), "kind": "prompt", "text": clean(text), "reply": None,
                                    "queued": True})
            elif kind == "assistant" and entries:
                for block in content if isinstance(content, list) else []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        reply = []
                        if block.get("name") == "AskUserQuestion":
                            entries[-1].setdefault("asked", []).extend(
                                {"question": q.get("question", ""), "options": [o.get("label", "") for o in q.get("options", [])]}
                                for q in (block.get("input") or {}).get("questions", []))
                    elif block.get("type") == "text" and block.get("text", "").strip():
                        reply.append(block["text"])
        close()
    return entries


def approved_plan(content) -> str | None:
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            body = block.get("content")
            body = body if isinstance(body, str) else " ".join(b.get("text", "") for b in body or [] if isinstance(b, dict))
            m = APPROVED_PLAN.match(body.strip())
            if m:
                return m.group(1).strip()
    return None


def render(session: str, entries: list[dict]) -> tuple[str, str, str] | None:
    s8 = session[:8]
    day = (entries[0]["time"] or "")[:10]   # named by the session's first prompt, even if that one failed
    stem = f"{day}_{s8}"
    # Ids are assigned before failed prompts are dropped, so they never shift (prompts/categories.json keys on them).
    # Ids stay stable as the session grows: messages queued mid-turn take the id of the prompt they arrived
    # during plus a letter (fe7c091a-135a), so they never renumber later prompts.
    n, sub = 0, 0
    for e in entries:
        if e.get("queued") and n:
            sub += 1
            e["id"] = f"{s8}-{n:03d}{chr(96 + sub)}"
        else:
            n, sub = n + 1, 0
            e["id"] = f"{s8}-{n:03d}"
        e["session"] = s8
    failed = [e for e in entries if e["reply"] and API_ERROR.match(e["reply"])]
    omit = load_omit()                     # prompts/omit.json: {id: reason} for housekeeping messages
    omitted = [e for e in entries if e["id"] in omit]
    entries = [e for e in entries if e not in failed and e not in omitted]
    if not entries:
        return None
    md = [f"# Claude Code session {s8} ({day})", "",
          "<!-- Generated by scripts/export_prompts.py from the session transcript. Do not edit by hand. -->", ""]
    if failed:
        md += [f"*{len(failed)} prompt{'s' if len(failed) > 1 else ''} that failed with an API error "
               f"(e.g. \"Credit balance is too low\") and were retried are omitted.*", ""]
    if omitted:
        md += [f"*{len(omitted)} operational prompt{'s' if len(omitted) > 1 else ''} (commit and deploy requests, local "
               f"setup, troubleshooting) are omitted; ids and reasons are in prompts/omit.json.*", ""]
    for e in entries:
        stamp = (e["time"] or "")[:16].replace("T", " ")
        md += [f"## {e['id']} · {stamp} UTC", "", f"**{KIND_LABEL[e['kind']]}**", ""]
        md += ["> " + line if line else ">" for line in e["text"].splitlines()]
        md += [""]
        if e["reply"]:
            md += ["**Claude**", "", e["reply"], ""]
        if e.get("asked"):
            md += ["**Claude asked**", ""]
            md += [f"- {q['question']} ({' / '.join(q['options'])})" for q in e["asked"]]
            md += [""]
        if e.get("plan"):
            md += ["<details><summary>Claude's plan (approved)</summary>", "", e["plan"], "", "</details>", ""]
        md += ["---", ""]
    meta = {"session": session, "omitted_api_errors": len(failed), "omitted": len(omitted), "entries": entries}
    return stem, redact("\n".join(md)), redact(json.dumps(meta, ensure_ascii=False, indent=1)) + "\n"


def load_omit() -> dict:
    path = OUT / "omit.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def export(transcript: Path) -> Path | None:
    entries = parse(transcript)
    out = render(transcript.stem, entries) if entries else None
    if out is None:
        return None
    stem, md, js = out
    OUT.mkdir(exist_ok=True)
    changed = write_if_changed(OUT / f"{stem}.md", md) | write_if_changed(OUT / f"{stem}.json", js)
    print(f"{'wrote' if changed else 'unchanged'} prompts/{stem}.md ({len(entries)} entries)", file=sys.stderr)
    return OUT / f"{stem}.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="*", type=Path)
    ap.add_argument("--from-hook", action="store_true", help="read the hook payload (transcript_path) from stdin")
    args = ap.parse_args()
    paths = list(args.transcripts)
    if args.from_hook:
        try:
            paths.append(Path(json.load(sys.stdin)["transcript_path"]).expanduser())
        except Exception as e:  # a logging hook must never break the session
            print(f"export_prompts: no transcript in hook input ({e})", file=sys.stderr)
            return
    for p in paths:
        try:
            export(p)
        except Exception as e:
            print(f"export_prompts: {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
