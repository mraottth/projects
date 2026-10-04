"""Evaluation page data (scripts/build_evaluations.py): versions, references and the report list."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_evaluations", ROOT / "scripts" / "build_evaluations.py")
be = importlib.util.module_from_spec(spec)
spec.loader.exec_module(be)


@pytest.fixture(scope="module")
def data():
    return be.build()


def test_every_version_resolves_its_report_and_commits(data):
    curated = json.loads((ROOT / "eval" / "versions.json").read_text())
    assert [v["id"] for v in data["versions"]] == [v["id"] for v in curated]
    for v in data["versions"]:
        assert set(v["metrics"]["-1"]) == set(be.METRICS)
        assert v["commits"] and all(c["url"].startswith("https://github.com/") for c in v["commits"])
    assert len({v["split_hash"] for v in data["versions"]}) == 1      # all scored on the same split


def test_champion_is_flagged_and_first(data):
    champ = json.loads((ROOT / "eval" / "champion.json").read_text())
    assert data["reports"][0]["champion"] and data["reports"][0]["id"] == Path(champ["report"]).stem
    assert sum(r["champion"] for r in data["reports"]) == 1
    assert sum(v["champion"] for v in data["versions"]) == 1
    rest = data["reports"][1:]
    assert [(r["date"], r["time"]) for r in rest] == sorted(((r["date"], r["time"]) for r in rest), reverse=True)


def test_only_markdown_reports_are_listed(data):
    ids = {r["id"] for r in data["reports"]}
    assert ids == {p.stem for p in (ROOT / "eval" / "reports").glob("*.md")}


def test_paired_ci_only_against_the_previous_version(data):
    vs = data["versions"]
    for prev, cur in zip(vs, vs[1:]):
        if "ci_vs_previous" in cur:
            ch = cur["ci_vs_previous"]["-1"]
            assert ch["ci95"][0] <= ch["mean_diff"] <= ch["ci95"][1]
            gap = cur["metrics"]["-1"]["ndcg@10"] - prev["metrics"]["-1"]["ndcg@10"]
            assert ch["mean_diff"] == pytest.approx(gap, abs=1e-6)


def test_report_tables_use_version_names(data):
    champ = data["reports"][0]
    v = next(x for x in data["versions"] if x["champion"])
    assert champ["model"] == f"{v['id']} · {v['title']}"
    assert f"**{v['id']} · {v['title']}**" in champ["markdown"]          # the model's row in the metric tables
    prev = data["versions"][[x["id"] for x in data["versions"]].index(v["id"]) - 1]
    assert f"vs. Previous best: {prev['id']} · {prev['title']}" in champ["markdown"]


def test_one_card_per_version_newest_version_first(data):
    groups = data["groups"]
    vgroups = [g for g in groups if g["id"] != "earlier"]
    assert [g["id"] for g in vgroups] == [v["id"] for v in sorted(data["versions"], key=lambda v: (v["date"], v["id"]), reverse=True)]
    assert vgroups[0]["champion"]                                        # the newest version is the champion
    placed = [g["main"] for g in vgroups] + [r for g in groups for r in g["others"]]
    assert sorted(placed) == sorted(r["id"] for r in data["reports"])     # every report in exactly one card
