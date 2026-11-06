"""Prompt export (scripts/export_prompts.py) and changelog build (scripts/build_changelog.py). No artifacts needed."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_prompts = _load("export_prompts")
build_changelog = _load("build_changelog")


def _line(**d):
    return json.dumps(d) + "\n"


def _transcript(tmp_path):
    home = str(Path.home())
    lines = [
        _line(type="user", timestamp="2026-10-01T10:00:00.000Z", message={"content": "First question"}),
        _line(type="assistant", message={"content": [{"type": "text", "text": "Let me look."}]}),
        _line(type="assistant", message={"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}),
        _line(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ls output"}]}),
        _line(type="assistant", message={"content": [{"type": "text", "text": f"Done: see {home}/notes.md"}]}),
        # a prompt with a pasted file, a screenshot and an editor notice
        _line(type="user", timestamp="2026-10-01T11:00:00.000Z", message={"content": [
            {"type": "document", "title": "export.csv", "source": {"data": "a,b\n1,2\n3,4"}},
            {"type": "image", "source": {}},
            {"type": "text", "text": "<ide_opened_file>x.py</ide_opened_file>"},
            {"type": "text", "text": "Use this file"}]}),
        _line(type="assistant", message={"content": [
            {"type": "tool_use", "id": "q1", "name": "AskUserQuestion",
             "input": {"questions": [{"question": "Which one?", "options": [{"label": "A"}, {"label": "B"}]}]}}]}),
        _line(type="user", timestamp="2026-10-01T11:01:00.000Z", message={"content": [
            {"type": "tool_result", "tool_use_id": "q1", "content": 'The user answered: "Which one?"="A". You can now continue with these answers in mind.'}]}),
        _line(type="assistant", message={"content": [{"type": "text", "text": "Going with A."}]}),
        # noise that must not become prompts
        _line(type="user", timestamp="2026-10-01T11:02:00.000Z", isMeta=True, message={"content": "meta"}),
        _line(type="user", timestamp="2026-10-01T11:03:00.000Z", message={"content": "<task-notification>done</task-notification>"}),
        _line(type="user", timestamp="2026-10-01T11:04:00.000Z", isCompactSummary=True, message={"content": "This session is being continued from a previous conversation"}),
        # a prompt that failed with an API error is dropped, without renumbering later prompts
        _line(type="user", timestamp="2026-10-01T12:00:00.000Z", message={"content": "retry me"}),
        _line(type="assistant", message={"content": [{"type": "text", "text": "Credit balance is too low"}]}),
        _line(type="user", timestamp="2026-10-01T12:05:00.000Z", message={"content": "Last one"}),
        _line(type="assistant", message={"content": [{"type": "text", "text": "Bye"}]}),
    ]
    path = tmp_path / "abcdef12-0000.jsonl"
    path.write_text("".join(lines))
    return path


def test_export_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(export_prompts, "OUT", tmp_path / "prompts")
    md_path = export_prompts.export(_transcript(tmp_path))
    data = json.loads(md_path.with_suffix(".json").read_text())
    es = data["entries"]
    assert [e["id"] for e in es] == ["abcdef12-001", "abcdef12-002", "abcdef12-003", "abcdef12-005"]
    assert [e["kind"] for e in es] == ["prompt", "prompt", "answer", "prompt"]
    assert es[0]["reply"] == "Done: see ~/notes.md"                     # final reply only, home path shortened
    assert es[1]["text"] == "[attached file: export.csv, 3 lines]\n\n[screenshot]\n\nUse this file"
    assert es[1]["asked"] == [{"question": "Which one?", "options": ["A", "B"]}]
    assert es[2]["text"] == '"Which one?"="A".' and es[2]["reply"] == "Going with A."
    assert data["omitted_api_errors"] == 1
    md = md_path.read_text()
    assert "Answer to Claude's question" in md and "ls output" not in md and "Let me look" not in md
    assert export_prompts.export(md_path.parent.parent / "abcdef12-0000.jsonl") == md_path   # idempotent re-run


def test_link_commits_to_prompts():
    prompts = [{"id": "s-001", "time": "2026-10-01T10:00:00.000Z"}, {"id": "s-002", "time": "2026-10-01T12:00:00.000Z"}]
    commits = [
        {"short": "aaa", "time": "2026-10-01T07:30:00-04:00", "web_session": False},   # 11:30Z -> after s-001
        {"short": "bbb", "time": "2026-10-01T12:30:00+00:00", "web_session": False},   # after s-002
        {"short": "ccc", "time": "2026-10-01T09:00:00+00:00", "web_session": False},   # before any prompt
        {"short": "ddd", "time": "2026-10-01T12:40:00+00:00", "web_session": True},    # web sessions link too
        {"short": "eee", "time": "2026-10-02T09:00:00+00:00", "web_session": True},    # >12 h after s-002: unlinked
    ]
    build_changelog.link(commits, prompts)
    assert [c["prompt"] for c in commits] == ["s-001", "s-002", None, "s-002", None]
    assert prompts[0]["commits"] == ["aaa"] and prompts[1]["commits"] == ["bbb", "ddd"]


def test_commit_categories():
    labels = {"abc1234": ["ui", "model"]}
    assert build_changelog.categorize_commit({"subject": "eval: add NDCG@50", "hash": "f00"}, labels) == ["eval"]
    assert build_changelog.categorize_commit({"subject": "Old style", "hash": "abc1234ffff"}, labels) == ["ui", "model"]
    assert build_changelog.categorize_commit({"subject": "Old style", "hash": "zzz"}, labels) == ["uncategorized"]


def test_parse_decisions():
    text = (ROOT / "DECISIONS.md").read_text()
    ds = build_changelog.parse_decisions(text)
    assert ds[0]["id"] == "D-001" and ds[0]["categories"] == ["data"] and "fe7c091a-004" in ds[0]["prompts"]
    assert [s["label"] for s in ds[0]["sections"]][:2] == ["Decision", "Context"]
    assert all(d["date"] and d["sections"] for d in ds)
    assert len({d["id"] for d in ds}) == len(ds)


def test_milestones_resolve():
    """Every reference in prompts/milestones.json points at a real prompt, decision or commit."""
    data = build_changelog.build()
    ms = data["milestones"]
    assert len(ms) >= 10 and len({m["id"] for m in ms}) == len(ms)
    raw = {m["id"]: m for m in json.loads((ROOT / "prompts" / "milestones.json").read_text())}
    for m in ms:
        assert m["prompts"] == raw[m["id"]]["prompts"], m["id"]           # nothing dropped
        assert m["decisions"] == raw[m["id"]]["decisions"], m["id"]
        assert len(m["commits"]) == len(raw[m["id"]]["commits"]), m["id"]
        assert m["category"] in data["categories"] and m["title"] and m["summary"]
    assert [m["date"] for m in ms] == sorted(m["date"] for m in ms)


def test_messages_sent_mid_turn_are_logged_without_renumbering(tmp_path):
    lines = [
        _line(type="user", timestamp="2026-10-01T10:00:00.000Z", message={"content": "Start the work"}),
        _line(type="assistant", message={"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}),
        _line(type="attachment", timestamp="2026-10-01T10:05:00.000Z",
              attachment={"type": "queued_command", "prompt": "Also sort the cards by version"}),
        _line(type="attachment", attachment={"type": "queued_command",
                                             "prompt": "<task-notification>background job done</task-notification>"}),
        _line(type="assistant", message={"content": [{"type": "text", "text": "Both done."}]}),
        _line(type="user", timestamp="2026-10-01T11:00:00.000Z", message={"content": "Next request"}),
    ]
    t = tmp_path / "session-abcdef12.jsonl"
    t.write_text("".join(lines))
    entries = export_prompts.parse(t)
    assert [e["text"] for e in entries] == ["Start the work", "Also sort the cards by version", "Next request"]
    assert entries[0]["reply"] == "Both done." and entries[1]["reply"] is None
    export_prompts.render("abcdef12-0000", entries)
    assert [e["id"] for e in entries] == ["abcdef12-001", "abcdef12-001a", "abcdef12-002"]


def test_redaction_scrubs_session_links_usernames_and_local_values(tmp_path, monkeypatch):
    """scripts/redact.py: built-in rules, plus literal values from the git-ignored .redact.local.json."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("redact", ROOT / "scripts" / "redact.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    local = tmp_path / "local.json"
    local.write_text('[["my-secret-project", "<gcp-project>"]]')
    monkeypatch.setattr(rd, "LOCAL", local)
    monkeypatch.setattr(rd, "_local", None)
    text = ("see https://claude.ai/code/session_0123abcdEFGH4567ijkl and claude --teleport session_0123abcdEFGH4567ijkl; "
            "transcripts in ~/.claude/projects/-Users-someone-Desktop-x/ and /Users/someone/code; project my-secret-project")
    out = rd.redact(text)
    assert "session_0123" not in out and "someone" not in out and "my-secret-project" not in out
    assert "<Claude Code session link>" in out and "<session id>" in out and "<gcp-project>" in out
    assert rd.redact("nothing to hide") == "nothing to hide"
    assert not (ROOT / ".redact.local.json").exists() or ".redact.local.json" in (ROOT / ".gitignore").read_text()
