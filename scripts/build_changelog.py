"""Build frontend/src/changelog.json for the Timeline page: commits, prompts and decisions.

  python3 scripts/build_changelog.py        # run by `make frontend` and `make deploy`

Sources:
  git log           every commit in this repo
  prompts/*.json    exported Claude Code sessions (scripts/export_prompts.py)
  prompts/categories.json, prompts/commit_categories.json   hand labels for past prompts / commits
  DECISIONS.md      the decision log
  prompts/milestones.json   hand-curated major changes for the visual timeline

This repo is a single, standalone project (unlike Shelf Life, which lives inside the larger
`projects` portfolio repo and has to filter git log by folder), so commits are just `git log`
over the whole repo. They're categorized first by `commit_categories.json` (hand overrides, keyed
by short hash), then by which files they touch (FILE_CATEGORIES below) -- this repo's commits use
conventional-commit types (`feat:`/`fix:`/`docs:`), not a bare `category:` prefix, so there's no
regex step the way Shelf Life has one. Commits with a `Claude-Session:` trailer were made in
Claude Code on the web and are marked as such -- see CLAUDE.md for how to import that session's
own prompt-by-prompt transcript (a manual `claude --teleport` step; not run automatically here).
Each commit is linked to the latest prompt before it, if that prompt is within LINK_WINDOW_H hours.
The output is a build artifact (git-ignored): it can't be committed without always lagging the
commit that contains it. `make frontend` / `make deploy` generate it locally.
"""

from __future__ import annotations  # runs under the system python3 (3.9) from the Claude Code hook

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact import redact  # noqa: E402  (scrubs local paths/session ids; scripts/redact.py)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "src" / "changelog.json"
GITHUB = "https://github.com/mraottth/job-search-assistant"

CATEGORIES = {
    "ui": "UI/UX", "ai": "AI (scoring, generation, Jobby)", "pipeline": "Ingest & pipeline",
    "infra": "Infrastructure & deploy", "docs": "Docs & repo", "discussion": "Questions & explanations",
}
# Longest/most-specific prefixes first within each list -- a file can match more than one category.
FILE_CATEGORIES: list[tuple[str, list[str]]] = [
    ("frontend/", ["ui"]),
    ("src/jobhunt/core/generate.py", ["ai"]), ("src/jobhunt/core/scoring.py", ["ai"]),
    ("src/jobhunt/core/discovery.py", ["ai"]), ("src/jobhunt/core/jobby.py", ["ai"]),
    ("src/jobhunt/core/profile.py", ["ai"]), ("src/jobhunt/core/docx_writer.py", ["ai"]),
    ("src/jobhunt/core/llm.py", ["ai"]), ("src/jobhunt/core/company_resolver.py", ["ai"]),
    ("src/jobhunt/api/generate.py", ["ai"]), ("src/jobhunt/api/discovery.py", ["ai"]),
    ("src/jobhunt/api/jobby.py", ["ai"]), ("src/jobhunt/api/onboarding.py", ["ai", "ui"]),
    ("src/jobhunt/api/resumes.py", ["ai"]),
    ("src/jobhunt/ingest/", ["pipeline"]), ("src/jobhunt/pipeline/", ["pipeline"]),
    ("src/jobhunt/core/digest.py", ["pipeline"]), ("src/jobhunt/core/queue.py", ["pipeline"]),
    ("src/jobhunt/core/dedupe.py", ["pipeline"]), ("src/jobhunt/core/prefilter.py", ["pipeline"]),
    ("src/jobhunt/core/html_extract.py", ["pipeline"]), ("src/jobhunt/core/page_fetch.py", ["pipeline"]),
    ("src/jobhunt/core/browser_fetch.py", ["pipeline"]), ("src/jobhunt/core/ats_detect.py", ["pipeline"]),
    ("src/jobhunt/core/generic_postings.py", ["pipeline"]), ("src/jobhunt/core/salary_extract.py", ["pipeline"]),
    ("src/jobhunt/core/retag.py", ["pipeline"]), ("src/jobhunt/core/reset.py", ["pipeline"]),
    ("src/jobhunt/core/email_digest.py", ["pipeline"]), ("src/jobhunt/api/sources.py", ["pipeline"]),
    ("src/jobhunt/api/jobs.py", ["pipeline"]), ("src/jobhunt/api/pipeline.py", ["pipeline"]),
    ("Makefile", ["infra"]), ("Dockerfile", ["infra"]), (".deploy.local.mk", ["infra"]),
    (".gcloudignore", ["infra"]), (".dockerignore", ["infra"]), ("src/jobhunt/db/", ["infra"]),
    ("src/jobhunt/config.py", ["infra"]), ("src/jobhunt/api/internal.py", ["infra"]),
    ("src/jobhunt/api/settings.py", ["infra"]), ("src/jobhunt/api/main.py", ["infra"]),
    ("CLAUDE.md", ["docs"]), ("README.md", ["docs"]), ("DECISIONS.md", ["docs"]),
]
TRAILER = re.compile(r"^(Co-Authored-By|Claude-Session|Signed-off-by):.*$", re.M | re.I)
SEP_REC, SEP_FIELD = "\x1e", "\x1f"
LINK_WINDOW_H = 12


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def git_commits(repo: Path = ROOT) -> list[dict]:
    fmt = SEP_REC + SEP_FIELD.join(["%H", "%h", "%cI", "%an", "%s", "%b"]) + SEP_FIELD
    out = subprocess.run(["git", "-C", str(repo), "log", "--no-merges", f"--format={fmt}", "--name-only"],
                         capture_output=True, text=True, check=True).stdout
    commits = []
    for rec in out.split(SEP_REC)[1:]:
        full, short, when, author, subject, body, files = rec.split(SEP_FIELD)
        session = re.search(r"^Claude-Session:\s*(\S+)", body, re.M)
        paths = [f for f in files.split("\n") if f.strip()]
        commits.append({
            "hash": full, "short": short, "time": when, "author": author, "subject": redact(subject.strip()),
            "body": redact(TRAILER.sub("", body).strip()), "files": len(paths), "_paths": paths,
            # Only touches prompts/ (prompt-log updates, labels): the Timeline folds these into "Housekeeping".
            "housekeeping": bool(paths) and all(f.split("/", 1)[0] == "prompts" for f in paths),
            "web_session": bool(session), "url": f"{GITHUB}/commit/{full}",
        })
    return commits


def categorize_commit(c: dict, overrides: dict) -> list[str]:
    for short, cats in overrides.items():
        if c["hash"].startswith(short):
            return cats
    cats: list[str] = []
    for prefix, file_cats in FILE_CATEGORIES:
        if any(p.startswith(prefix) for p in c["_paths"]):
            cats += [fc for fc in file_cats if fc not in cats]
    return cats or ["uncategorized"]


def load_prompts(folder: Path) -> list[dict]:
    labels = load_json(folder / "categories.json", {})
    omit, cuts = load_json(folder / "omit.json", {}), load_json(folder / "trim.json", {})
    prompts = []
    for f in sorted(folder.glob("*.json")):
        if f.name in ("categories.json", "commit_categories.json", "omit.json", "trim.json", "milestones.json"):
            continue
        for e in load_json(f, {}).get("entries", []):
            if e["id"] in omit:
                continue
            e = dict(e, text=cuts.get(e["id"], e["text"]))
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
        listed = lambda k: [x.strip() for x in fields.get(k, "").split(",") if x.strip()]  # noqa: E731,B023
        out.append({"id": num.strip(), "title": title.strip(), "date": fields.get("Date", ""),
                    "categories": listed("Category") or ["uncategorized"], "prompts": listed("Prompts"),
                    "commits": listed("Commits"), "sections": [{"label": lbl, "text": t} for lbl, t in sections]})
    return out


def resolve_milestones(milestones: list[dict], commits: list[dict], prompts: list[dict], decisions: list[dict]) -> list[dict]:
    """Check each milestone's references; commits resolve by short-hash prefix to {short, subject, url}.
    Unknown references are dropped with a warning."""
    pids = {p["id"] for p in prompts}
    dids = {d["id"] for d in decisions}
    out = []
    for m in milestones:
        m = dict(m)
        missing = [p for p in m.get("prompts", []) if p not in pids] + [d for d in m.get("decisions", []) if d not in dids]
        resolved = []
        for short in m.get("commits", []):
            c = next((c for c in commits if c["hash"].startswith(short)), None)
            if c:
                resolved.append({"short": c["short"], "subject": c["subject"], "url": c["url"]})
            else:
                missing.append(short)
        if missing:
            print(f"build_changelog: milestone {m['id']} drops unknown references: {', '.join(missing)}", file=sys.stderr)
        m["prompts"] = [p for p in m.get("prompts", []) if p in pids]
        m["decisions"] = [d for d in m.get("decisions", []) if d in dids]
        m["commits"] = resolved
        out.append(m)
    return out


def build() -> dict:
    commits = git_commits()
    overrides = load_json(ROOT / "prompts" / "commit_categories.json", {})
    for c in commits:
        c["categories"] = categorize_commit(c, overrides)
        del c["_paths"]
    prompts = load_prompts(ROOT / "prompts")
    link(commits, prompts)
    decisions = parse_decisions((ROOT / "DECISIONS.md").read_text(encoding="utf-8")) if (ROOT / "DECISIONS.md").exists() else []
    for kind, items in (("prompts", prompts), ("commits", commits)):
        missing = [x.get("id") or x["short"] for x in items if x["categories"] == ["uncategorized"]]
        if missing:
            print(f"build_changelog: {len(missing)} uncategorized {kind}: {', '.join(missing[:8])}"
                  f"{' ...' if len(missing) > 8 else ''}", file=sys.stderr)
    omitted = len(load_json(ROOT / "prompts" / "omit.json", {}))
    milestones = resolve_milestones(load_json(ROOT / "prompts" / "milestones.json", []), commits, prompts, decisions)
    return {"repo": GITHUB, "categories": CATEGORIES, "commits": sorted(commits, key=lambda c: c["time"]),
            "prompts": prompts, "decisions": decisions, "milestones": milestones, "omitted_prompts": omitted}


def main() -> None:
    text = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") == text:
        print("changelog.json unchanged", file=sys.stderr)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    d = json.loads(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(d['commits'])} commits, {len(d['prompts'])} prompts, "
          f"{len(d['decisions'])} decisions, {len(d['milestones'])} milestones", file=sys.stderr)


if __name__ == "__main__":
    main()
