"""Build frontend/src/evaluations.json for the Evaluation page: model versions and every evaluation report.

  python3 scripts/build_evaluations.py        # run by `make frontend` and `make deploy`

Sources:
  eval/versions.json    hand-curated model versions (title, summary, commits, decisions, the test report that
                        scores each one; earlier versions are re-scored on today's split with `run.py --params`)
  eval/reports/*.md     every evaluation report (the .json next to a report adds its metrics)
  eval/champion.json    the current champion (its report's card opens first)

Like changelog.json, the output is a git-ignored build artifact: make frontend / make deploy generate it.
"""

from __future__ import annotations  # runs under the system python3 (3.9)

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_changelog import ROOT, git_commits, parse_decisions  # noqa: E402

EVAL = ROOT / "eval"
OUT = ROOT / "frontend" / "src" / "evaluations.json"
METRICS = ["ndcg@10", "recall@10", "precision@10", "ndcg@20", "recall@20", "precision@20"]
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(?:_(.+))?$")


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def row(report: dict, key: str) -> dict | None:
    return next((r for r in report.get("rows", []) if r["key"] == key), None)


def metrics_of(r: dict) -> dict:
    """{n: {metric: value}} for one report row, n as in the report ("1", ..., "-1" = full history)."""
    return {n: {m: cell[m] for m in METRICS} for n, cell in r["by_n"].items()}


def report_kind(stem: str, report: dict | None) -> str:
    if stem.startswith("als_sweep"):
        return "ALS sweep (earlier evaluation)"
    m = STAMP.match(stem)
    if not m or not m.group(4):
        return "Earlier evaluation (random holdout)"
    if m.group(4) == "validation":
        return "Validation run"
    return "Test set"


def build_reports(champion: dict | None) -> list[dict]:
    champ_stem = Path(champion["report"]).stem if champion else None
    out = []
    for md in sorted((EVAL / "reports").glob("*.md")):
        stem = md.stem
        text = md.read_text(encoding="utf-8")
        report = load_json(md.with_suffix(".json"))
        m = STAMP.match(stem) or re.search(r"(\d{4}-\d{2}-\d{2})", stem)
        date = m.group(1) if m else ""
        time = f"{m.group(2)}:{m.group(3)}" if m and STAMP.match(stem) else ""
        new_format = bool(report and "rows" in report)
        sl = row(report, "shelf_life") if new_format else None
        out.append({
            "id": stem, "date": date, "time": time, "kind": report_kind(stem, report),
            "title": text.split("\n", 1)[0].lstrip("# ").strip(),
            "model": report["model"]["name"] if new_format else None,
            "set": report.get("set") if new_format else None,
            "n_users": report.get("n_users") if new_format else None,
            "split_hash": report.get("split_hash") if new_format else None,
            "ndcg10": sl["by_n"]["-1"]["ndcg@10"] if sl else None,
            "champion": stem == champ_stem,
            "markdown": text,
        })
    champ = [r for r in out if r["champion"]]                   # the champion first, then newest first
    rest = sorted((r for r in out if not r["champion"]), key=lambda r: (r["date"], r["time"]), reverse=True)
    return champ + rest


def build_versions(commits: list[dict], champion: dict | None, decisions: dict) -> list[dict]:
    """Versions with their metrics, resolved commits and decisions (title and sections from DECISIONS.md)."""
    by_short = lambda s: next((c for c in commits if c["hash"].startswith(s)), None)  # noqa: E731
    versions = []
    for v in load_json(EVAL / "versions.json", []):
        report = load_json(EVAL / "reports" / f"{v['report']}.json")
        if report is None:
            raise SystemExit(f"build_evaluations: version {v['id']} names a missing report {v['report']}")
        sl = row(report, "shelf_life")
        resolved = []
        for short in v.get("commits", []):
            c = by_short(short)
            if c is None:
                raise SystemExit(f"build_evaluations: version {v['id']} names an unknown commit {short}")
            resolved.append({"short": c["short"], "subject": c["subject"], "url": c["url"]})
        decs = []
        for did in v.get("decisions", []):
            if did not in decisions:
                raise SystemExit(f"build_evaluations: version {v['id']} names an unknown decision {did}")
            d = decisions[did]
            decs.append({"id": did, "title": d["title"], "sections": d["sections"]})
        champ = next((h for h in report.get("head_to_head", []) if h["key"] == "champion"), None)
        versions.append({**v, "commits": resolved, "decisions": decs,
                         "metrics": metrics_of(sl), "n_users": report["n_users"],
                         "split_hash": report["split_hash"], "_vs_champion": champ,
                         "champion": bool(champion) and Path(champion["report"]).stem == v["report"]})
    # The paired CI applies when the report's previous best is the previous version (same full-history NDCG@10).
    for prev, cur in zip(versions, versions[1:]):
        ch = cur["_vs_champion"]
        if ch and abs(ch["by_n"]["-1"]["mean_b"] - prev["metrics"]["-1"]["ndcg@10"]) < 1e-6:
            cur["ci_vs_previous"] = {n: {"mean_diff": c["mean_diff"], "ci95": c["ci95"], "win": c["win"], "loss": c["loss"]}
                                     for n, c in ch["by_n"].items() if c.get("n")}
    for v in versions:
        v.pop("_vs_champion")
    return versions


def build() -> dict:
    champion = load_json(EVAL / "champion.json")
    commits = git_commits()
    decisions = {d["id"]: d for d in parse_decisions((ROOT / "DECISIONS.md").read_text(encoding="utf-8"))}
    versions = build_versions(commits, champion, decisions)
    latest = load_json(EVAL / "reports" / f"{versions[-1]['report']}.json") if versions else {}
    refs = []
    # Baselines for comparison: the 2023 project's best method (its similar-readers lists) and popular books.
    for key, label in ((latest.get("best_2023"), "2023 Book Recommender performance"), ("popular", "Most popular books")):
        r = row(latest, key) if key else None
        if r:
            refs.append({"key": r["key"], "label": label, "metrics": metrics_of(r)})
    return {"versions": versions, "references": refs, "reports": build_reports(champion), "metrics": METRICS,
            "buckets": latest.get("buckets", [1, 3, 5, 10, 25, -1])}


def main() -> None:
    text = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") == text:
        print("evaluations.json unchanged", file=sys.stderr)
        return
    OUT.write_text(text, encoding="utf-8")
    d = json.loads(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(d['versions'])} versions, {len(d['reports'])} reports", file=sys.stderr)


if __name__ == "__main__":
    main()
