"""The Product Timeline's milestones (prompts/milestones.json) stay current and correct: every
decision belongs to one, and every prompt, decision and commit a milestone names exists. They had
fallen behind once (milestones stopped at M10 / D-014 while decisions reached D-027)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_changelog", ROOT / "scripts" / "build_changelog.py")
build_changelog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_changelog)

MILESTONES = json.loads((ROOT / "prompts" / "milestones.json").read_text(encoding="utf-8"))
DECISIONS = build_changelog.parse_decisions((ROOT / "DECISIONS.md").read_text(encoding="utf-8"))


def test_every_decision_belongs_to_a_milestone():
    missing = build_changelog.decisions_without_milestone(MILESTONES, DECISIONS)
    assert not missing, (
        f"Decisions with no milestone: {', '.join(missing)}. Add or extend a milestone in "
        "prompts/milestones.json (or, for a process-only decision, add it to NO_MILESTONE in "
        "scripts/build_changelog.py)."
    )


def test_milestones_are_numbered_and_in_date_order():
    assert [m["id"] for m in MILESTONES] == [f"M{i:02d}" for i in range(1, len(MILESTONES) + 1)]
    dates = [m["date"] for m in MILESTONES]
    assert dates == sorted(dates)
    for m in MILESTONES:
        assert m["title"] and m["summary"] and m["icon"], m["id"]
        assert m["category"] in build_changelog.CATEGORIES, m["id"]


def test_every_reference_exists():
    prompt_ids = {p["id"] for p in build_changelog.load_prompts(ROOT / "prompts")}
    decision_ids = {d["id"] for d in DECISIONS}
    for m in MILESTONES:
        assert set(m["prompts"]) <= prompt_ids, (m["id"], set(m["prompts"]) - prompt_ids)
        assert set(m["decisions"]) <= decision_ids, (m["id"], set(m["decisions"]) - decision_ids)


def test_every_commit_exists():
    if subprocess.run(["git", "rev-parse"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("not a git checkout")
    for m in MILESTONES:
        for short in m["commits"]:
            found = subprocess.run(
                ["git", "cat-file", "-e", f"{short}^{{commit}}"], cwd=ROOT, capture_output=True
            )
            assert found.returncode == 0, f"{m['id']} names commit {short}, which isn't in this repo"
