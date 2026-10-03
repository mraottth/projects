"""Build frontend/src/changelog.json for the Changelog page: commits, prompts and decisions.

  python3 scripts/build_changelog.py        # run by `make frontend` and `make deploy`

Sources:
  git log           commits touching this project (both folder names: it was renamed on 2026-09-27)
  prompts/*.json    exported Claude Code sessions (scripts/export_prompts.py)
  prompts/categories.json, prompts/commit_categories.json   hand labels for past prompts / commits
  DECISIONS.md      the decision log

Commits are categorized by their message prefix (`ui: ...`), falling back to commit_categories.json.
Each commit is linked to the latest prompt before it, if that prompt is within LINK_WINDOW_H hours
(commits from a session whose prompts weren't exported then stay unlinked). Commits with a
`Claude-Session:` trailer were made in Claude Code on the web and are marked as such. The output is a build
artifact (git-ignored): it can't be committed without always lagging the commit that contains it.
make frontend / make deploy generate it locally (Cloud Build has no .git), and the upload includes it.
"""

from __future__ import annotations  # runs under the system python3 (3.9)

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "frontend" / "src" / "changelog.json"
PATHS = [ROOT.name, "New & Improved Goodreads Book Recommender"]
GITHUB = "https://github.com/mraottth/projects"

CATEGORIES = {
    "ui": "UI/UX", "model": "Recommender model", "eval": "Evaluation", "data": "Data & pipeline",
    "assistant": "AI Assistant", "infra": "Infrastructure & deploy", "docs": "Docs & repo",
    "discussion": "Questions & explanations",
}
PREFIX = re.compile(r"^(%s)(?:\([^)]*\))?:\s*" % "|".join(CATEGORIES))
TRAILER = re.compile(r"^(Co-Authored-By|Claude-Session|Signed-off-by):.*$", re.M | re.I)
SEP_REC, SEP_FIELD = "\x1e", "\x1f"
LINK_WINDOW_H = 12


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git_commits(repo: Path = REPO) -> list[dict]:
    fmt = SEP_REC + SEP_FIELD.join(["%H", "%h", "%cI", "%an", "%s", "%b"]) + SEP_FIELD
    out = subprocess.run(["git", "-C", str(repo), "log", "--no-merges", f"--format={fmt}", "--name-only", "--", *PATHS],
                         capture_output=True, text=True, check=True).stdout
    commits = []
    for rec in out.split(SEP_REC)[1:]:
        full, short, when, author, subject, body, files = rec.split(SEP_FIELD)
        session = re.search(r"^Claude-Session:\s*(\S+)", body, re.M)
        commits.append({
            "hash": full, "short": short, "time": when, "author": author, "subject": subject.strip(),
            "body": TRAILER.sub("", body).strip(), "files": len([f for f in files.split("\n") if f.strip()]),
            "web_session": bool(session), "url": f"{GITHUB}/commit/{full}",
        })
    return commits


def categorize_commit(c: dict, labels: dict) -> list[str]:
    m = PREFIX.match(c["subject"])
    if m:
        return [m.group(1)]
    for short, cats in labels.items():
        if c["hash"].startswith(short):
            return cats
    return ["uncategorized"]


def load_prompts(folder: Path) -> list[dict]:
    labels = load_json(folder / "categories.json", {})
    prompts = []
    for f in sorted(folder.glob("*.json")):
        if f.name in ("categories.json", "commit_categories.json", "omit.json"):
            continue
        for e in load_json(f, {}).get("entries", []):
            e = dict(e)
            e["categories"] = labels.get(e["id"], ["uncategorized"])
            prompts.append(e)
    return sorted(prompts, key=lambda e: e["time"] or "")


def link(commits: list[dict], prompts: list[dict]) -> None:
    """Attach each commit to the latest prompt before it, within LINK_WINDOW_H hours
    (prompt["commits"] / commit["prompt"])."""
    from datetime import datetime, timedelta
    for p in prompts:
        p["commits"] = []
    times = [p["time"] for p in prompts]
    for c in sorted(commits, key=lambda c: c["time"]):
        c["prompt"] = None
        t = utc(c["time"])
        idx = max((i for i, pt in enumerate(times) if pt and pt <= t), default=None)
        if idx is None:
            continue
        gap = datetime.fromisoformat(t.replace("Z", "+00:00")) - datetime.fromisoformat(times[idx].replace("Z", "+00:00"))
        if gap <= timedelta(hours=LINK_WINDOW_H):
            c["prompt"] = prompts[idx]["id"]
            prompts[idx]["commits"].append(c["short"])


def utc(iso: str) -> str:
    """Normalize a git ISO timestamp (with offset) to the transcripts' UTC 'Z' form for string comparison."""
    from datetime import datetime, timezone
    return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_decisions(text: str) -> list[dict]:
    out = []
    for block in re.split(r"^## (?=D-\d+)", text, flags=re.M)[1:]:
        head, _, rest = block.partition("\n")
        num, _, title = head.partition(" · ")
        fields = dict(re.findall(r"^- \*\*(\w+):\*\*\s*(.+)$", rest, re.M))
        sections = [(label.strip(), body.strip()) for label, body in
                    re.findall(r"^\*\*([^*]+?)\.\*\*\s*(.+?)(?=^\*\*[^*]+?\.\*\*|\Z)", rest, re.M | re.S)]
        listed = lambda k: [x.strip() for x in fields.get(k, "").split(",") if x.strip()]  # noqa: E731
        out.append({"id": num.strip(), "title": title.strip(), "date": fields.get("Date", ""),
                    "categories": listed("Category") or ["uncategorized"], "prompts": listed("Prompts"),
                    "commits": listed("Commits"), "sections": [{"label": l, "text": t} for l, t in sections]})
    return out


def build() -> dict:
    commits = git_commits()
    labels = load_json(ROOT / "prompts" / "commit_categories.json", {})
    for c in commits:
        c["categories"] = categorize_commit(c, labels)
    prompts = load_prompts(ROOT / "prompts")
    link(commits, prompts)
    decisions = parse_decisions((ROOT / "DECISIONS.md").read_text(encoding="utf-8"))
    for kind, items in (("prompts", prompts), ("commits", commits)):
        missing = [x.get("id") or x["short"] for x in items if x["categories"] == ["uncategorized"]]
        if missing:
            print(f"build_changelog: {len(missing)} uncategorized {kind}: {', '.join(missing[:8])}"
                  f"{' ...' if len(missing) > 8 else ''}", file=sys.stderr)
    omitted = len(load_json(ROOT / "prompts" / "omit.json", {}))
    return {"repo": GITHUB, "categories": CATEGORIES, "commits": sorted(commits, key=lambda c: c["time"]),
            "prompts": prompts, "decisions": decisions, "omitted_prompts": omitted}


def main() -> None:
    text = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") == text:
        print("changelog.json unchanged", file=sys.stderr)
        return
    OUT.write_text(text, encoding="utf-8")
    d = json.loads(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(d['commits'])} commits, {len(d['prompts'])} prompts, "
          f"{len(d['decisions'])} decisions", file=sys.stderr)


if __name__ == "__main__":
    main()
