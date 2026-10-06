"""Build frontend/src/evaluations.json for the Evaluation page: model versions and every evaluation report,
on two tracks: ranking (recommendation quality) and rating (rating prediction).

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
METRICS = ["ndcg@10", "precision@10", "recall@10", "ndcg@20", "precision@20", "recall@20"]   # NDCG, Precision, Recall
RMETRICS = ["mae", "rmse", "nmae", "pearson", "spearman", "within1"]                          # rating track
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(?:_(.+))?$")


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def row(report: dict, key: str) -> dict | None:
    return next((r for r in report.get("rows", []) if r["key"] == key), None)


def metrics_of(r: dict, names: list[str] = METRICS) -> dict:
    """{n: {metric: value}} for one report row, n as in the report ("1", ..., "-1" = full history)."""
    return {n: {m: cell[m] for m in names} for n, cell in r["by_n"].items()}


def rating_row(report: dict | None, key: str) -> dict | None:
    return next((r for r in (report or {}).get("rating_rows", []) if r["key"] == key), None)


SECTION = re.compile(r"^## (Track 1|Track 2|Metrics|Notes|Descriptions)\b", re.M)


def split_sections(text: str) -> dict:
    """A report's Markdown as a shared head (title, model, exposure), the Track 1 (ranking) and Track 2 (rating)
    sections and a shared tail (notes, descriptions). Reports from before the rating track have no Track 2 (their
    "## Metrics" onward is Track 1); the earliest, unstructured ones are all Track 1. head + track1 + tail is
    always the whole report."""
    marks = {m.group(1): m.start() for m in SECTION.finditer(text)}
    t1 = marks.get("Track 1", marks.get("Metrics"))
    if t1 is None:
        return {"head": "", "track1": text, "track2": None, "tail": ""}
    tail = min((marks[k] for k in ("Notes", "Descriptions") if k in marks and marks[k] > t1), default=len(text))
    t2 = marks.get("Track 2")
    return {"head": text[:t1], "track1": text[t1:t2 if t2 is not None else tail],
            "track2": text[t2:tail] if t2 is not None else None, "tail": text[tail:]}


def report_kind(stem: str, report: dict | None) -> str:
    if stem.startswith("als_sweep"):
        return "ALS sweep (earlier evaluation)"
    if stem.startswith("rating_sweep"):
        return "Rating model sweep (validation)"
    m = STAMP.match(stem)
    if not m or not m.group(4):
        return "Earlier evaluation (random holdout)"
    if m.group(4) == "validation":
        return "Validation run"
    return "Test set"


# Settings added after early reports were written, with the value those reports implicitly used.
PARAM_DEFAULTS = {"a_max": 1.0, "recency_half_life": None}


def norm_params(p: dict, rating: dict | None = None) -> str:
    """Model settings that define a version: the ranking Params (author_penalty is a display step, not part of
    the model) plus the rating predictor (v6 changed only that; reports before it have none = item-kNN).
    Reconstructed item means / calibration (D-051) don't define a version."""
    num = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v  # noqa: E731  1 == 1.0
    key = {k: num(v) for k, v in {**PARAM_DEFAULTS, **p}.items() if k != "author_penalty"}
    key["\x01predictor"] = (rating or {}).get("predictor", "knn")
    return json.dumps(key, sort_keys=True)


def dataset_of(report: dict | None, versions: list[dict]) -> str:
    """Which dataset a report's split belongs to (versions record theirs); unknown splits count as the first's."""
    by_split = {v["split_hash"]: v.get("dataset", "reviews") for v in versions}
    return by_split.get((report or {}).get("split_hash"), versions[0].get("dataset", "reviews") if versions else "reviews")


def version_key(report: dict, dataset: str) -> str:
    return norm_params(report.get("params", {}), report.get("rating")) + "|" + dataset


def version_labels(report: dict, versions: list[dict]) -> dict[str, str]:
    """Model names in a report -> the page's version names ("v5 · Recent reading counts more"): the model under
    test is matched by its settings, the previous best by the champion report it cites. Other rows keep their names."""
    by_params = {v["_params"]: v for v in versions}
    by_report = {v["report"]: v for v in versions}
    out = {}
    v = by_params.get(version_key(report, dataset_of(report, versions)))
    if v:
        out["\x01version"] = v["id"]
        name = report["model"]["name"]
        extra = re.search(r"\((before|after) leak fix\)", name)
        out[name] = f"{v['id']} · {v['title']}" + (f" ({extra.group(0)[1:-1]})" if extra else "")
    champ = row(report, "champion")
    m = re.search(r"reports/([^)\s]+)\.md", champ["description"]) if champ else None
    if m and m.group(1) in by_report:
        cv = by_report[m.group(1)]
        out[champ["name"]] = f"Previous best: {cv['id']} · {cv['title']}"
    return out


def relabel(text: str, labels: dict[str, str]) -> str:
    """Replace model names, longest first, through placeholders so one replacement can't feed another."""
    keys = sorted(labels, key=len, reverse=True)
    for i, k in enumerate(keys):
        text = text.replace(k, f"\x00{i}\x00")
    for i, k in enumerate(keys):
        text = text.replace(f"\x00{i}\x00", labels[k])
    return text


def build_reports(champion: dict | None, versions: list[dict]) -> list[dict]:
    champ_stem = Path(champion["report"]).stem if champion else None
    out = []
    for md in sorted((EVAL / "reports").glob("*.md")):
        stem = md.stem
        text = md.read_text(encoding="utf-8")
        report = load_json(md.with_suffix(".json"))
        m = STAMP.match(stem) or re.search(r"(\d{4}-\d{2}-\d{2})_?(\d{2})?(\d{2})?", stem)
        date = m.group(1) if m else ""
        time = f"{m.group(2)}:{m.group(3)}" if m and m.group(2) else ""
        new_format = bool(report and "rows" in report)
        sl = row(report, "shelf_life") if new_format else None
        rsl = rating_row(report, "shelf_life") if new_format else None
        labels = version_labels(report, versions) if new_format else {}
        version = labels.pop("\x01version", None)
        if version is None and new_format and STAMP.match(stem):
            # Candidate and validation runs (other settings or models) belong to a version of the same dataset:
            # the one current when they ran (the latest whose report of record is older), else the first one
            # after them (runs that led up to a dataset's first version).
            ds = dataset_of(report, versions)
            same = [v for v in versions if v.get("dataset", "reviews") == ds] or versions
            older = [v for v in same if v["report"] <= stem]
            newer = [v for v in same if v["report"] > stem]
            version = (max(older, key=lambda v: v["report"]) if older
                       else min(newer, key=lambda v: v["report"]) if newer else None)
            version = version["id"] if version else None
        if stem.startswith("rating_sweep") and report and report.get("base_report"):   # tuned against this version
            base = Path(report["base_report"]).stem
            version = next((v["id"] for v in versions if v["report"] == base), None)
        text = relabel(text, labels)
        kind = report_kind(stem, report)
        if new_format and versions and report.get("split_hash") not in {v["split_hash"] for v in versions}:
            kind += " (earlier split)"
        out.append({
            "id": stem, "date": date, "time": time, "kind": kind, "version": version,
            "title": text.split("\n", 1)[0].lstrip("# ").strip(),
            "model": labels.get(report["model"]["name"], report["model"]["name"]) if new_format else None,
            "set": report.get("set") if new_format else None,
            "n_users": report.get("n_users") if new_format else None,
            "split_hash": report.get("split_hash") if new_format else None,
            "ndcg10": sl["by_n"]["-1"]["ndcg@10"] if sl else None,
            "mae": rsl["by_n"]["-1"]["mae"] if rsl else None,
            "champion": stem == champ_stem,
            **split_sections(text),
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
        rsl = rating_row(report, "shelf_life")
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
        versions.append({**v, "dataset": v.get("dataset", "reviews"), "commits": resolved, "decisions": decs,
                         "_params": version_key(report, v.get("dataset", "reviews")), "_report": report,
                         "metrics": metrics_of(sl), "rmetrics": metrics_of(rsl, RMETRICS) if rsl else None,
                         "n_users": report["n_users"], "split_hash": report["split_hash"],
                         "champion": bool(champion) and Path(champion["report"]).stem == v["report"]})
    return versions


def paired_ci(h2h: list[dict], value: float | None, metric: str) -> dict | None:
    """The head-to-head (from a version's report) whose baseline scored `value`: the paired per-reader CI of the
    change from the previous point on the chart (the previous version, a dataset's bridge, or the 2023 point)."""
    if value is None:
        return None
    for h in h2h:
        c = h["by_n"].get("-1", {})
        if c.get("n") and abs(c["mean_b"] - value) < 1e-6:
            return {n: {"mean_diff": x["mean_diff"], "ci95": x["ci95"], "win": x["win"], "loss": x["loss"]}
                    for n, x in h["by_n"].items() if x.get("n")}
    return None


DATASET_LABEL = {"reviews": "Reviews data (15.7M ratings with a written review)",
                 "interactions": "All ratings (104M, to-read shelves excluded)"}
DATASET_SHORT = {"reviews": "Trained on 15M ratings", "interactions": "Switched to use all 104M ratings"}  # section headers
DATASET_NOTE = {"reviews": ["only ratings that came with a written review"],     # ... and the line under each
                "interactions": ["new data and test: scores restart"]}


def build_eras(versions: list[dict]) -> list[dict]:
    """One era per dataset, in order. Scores are only comparable within an era (each has its own test split). An
    era's chart starts with the 2023 Book Recommender's best method on that era's test (and, after the first era,
    the previous era's last version run on the new test: the bridge), so every line starts from where 2023 was."""
    eras = []
    for ds in dict.fromkeys(v["dataset"] for v in versions):
        vs = [v for v in versions if v["dataset"] == ds]
        latest = vs[-1]["_report"]
        b, rb = row(latest, latest.get("best_2023")), rating_row(latest, latest.get("rating_best_2023"))
        baseline = {"id": "2023", "label": "2023 Book Recommender", "kind": "baseline",
                    "ranking_method": b["name"] if b else None, "rating_method": rb["name"] if rb else None,
                    "metrics": metrics_of(b) if b else {}, "rmetrics": metrics_of(rb, RMETRICS) if rb else {},
                    "report": vs[-1]["report"]}
        bridge = None
        if eras:                                      # the previous era's last version on this era's test
            prev = eras[-1]["versions"][-1]
            first = vs[0]["_report"]
            c, rc = row(first, "champion"), rating_row(first, "champion")
            if c:
                bridge = {"id": f"{prev}*", "of": prev, "label": f"{prev} on the new test", "kind": "bridge",
                          "metrics": metrics_of(c), "rmetrics": metrics_of(rc, RMETRICS) if rc else {},
                          "report": vs[0]["report"]}
        refs = [{"key": "popular", "label": "Popular books", "metrics": metrics_of(row(latest, "popular"))}] \
            if row(latest, "popular") else []
        rrefs = [{"key": "book_avg", "label": "Book average", "metrics": metrics_of(rating_row(latest, "book_avg"), RMETRICS)}] \
            if rating_row(latest, "book_avg") else []
        # Change from the previous point on this era's line, with the paired CI when a report has it.
        prev_pt = bridge or baseline
        for v in vs:
            rep_ = v["_report"]
            v["previous"] = prev_pt["id"]
            ci = paired_ci(rep_.get("head_to_head", []), (prev_pt["metrics"].get("-1") or {}).get("ndcg@10"), "ndcg@10")
            rci = paired_ci(rep_.get("rating_head_to_head", []), (prev_pt["rmetrics"].get("-1") or {}).get("mae"), "mae")
            if ci:
                v["ci_vs_previous"] = ci
            if rci:
                v["rci_vs_previous"] = rci
            prev_pt = v
        eras.append({"id": ds, "label": DATASET_LABEL.get(ds, ds), "short": DATASET_SHORT.get(ds, ds),
                     "note": DATASET_NOTE.get(ds, []),
                     "versions": [v["id"] for v in vs],
                     "baseline": baseline, "bridge": bridge, "references": refs, "rating_references": rrefs})
    return eras


def strip_private(versions: list[dict]) -> list[dict]:  # noqa: D401
    return [{k: x for k, x in v.items() if not k.startswith("_")} for v in versions]


def build_groups(versions: list[dict], reports: list[dict]) -> list[dict]:
    """One card per model version, newest version first (by when the version was made, not when it was tested):
    the version's report of record plus its other runs (newest first). Runs that match no version (the earliest
    random-holdout reports, the ALS sweep) form a final "Earlier evaluations" group."""
    by_id = {r["id"]: r for r in reports}
    newest = lambda rs: sorted(rs, key=lambda r: (r["date"], r["time"]), reverse=True)  # noqa: E731
    groups = []
    for v in sorted(versions, key=lambda v: (v["date"], v["id"]), reverse=True):
        main = by_id[v["report"]]
        others = newest(r for r in reports if r["version"] == v["id"] and r["id"] != main["id"])
        groups.append({"id": v["id"], "title": f"{v['id']} · {v['title']}", "date": v["date"], "champion": v["champion"],
                       "ndcg10": main["ndcg10"], "mae": main["mae"], "main": main["id"], "others": [r["id"] for r in others]})
    rest = newest(r for r in reports if r["version"] is None)
    if rest:
        groups.append({"id": "earlier", "title": "Earlier evaluations (before the time-based test)", "date": rest[-1]["date"],
                       "champion": False, "ndcg10": None, "mae": None, "main": None, "others": [r["id"] for r in rest]})
    return groups


def build() -> dict:
    champion = load_json(EVAL / "champion.json")
    commits = git_commits()
    decisions = {d["id"]: d for d in parse_decisions((ROOT / "DECISIONS.md").read_text(encoding="utf-8"))}
    versions = build_versions(commits, champion, decisions)
    eras = build_eras(versions)
    latest = versions[-1]["_report"] if versions else {}
    reports = build_reports(champion, versions)
    return {"versions": strip_private(versions), "eras": eras, "reports": reports,
            "groups": build_groups(versions, reports), "metrics": METRICS, "rating_metrics": RMETRICS,
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
