"""Probe: how book-description neighbours (s12, text_nbrs_*) relate to co-rating neighbours (s07, item_nbrs_*).

Answers, before any model change, whether description embeddings add information the collaborative signal lacks:
- agreement: share of a book's top-k text neighbours that are also among its top-k co-rating neighbours, and the
  rank correlation of text cosine with co-rating similarity over co-rating neighbour pairs;
- coverage where co-rating evidence is thin: books with few co-rating neighbours, by number of raters;
- what text neighbours are made of: same parent genre, same author, same series (vs the co-rating neighbours).

    uv run python scripts/text_probe.py            # writes eval/text_probe.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goodrec.config import ARTIFACTS_DIR, EVAL_DIR  # noqa: E402
from goodrec.core.artifacts import load_artifacts  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(k: int = 20) -> None:
    art = load_artifacts(with_readers=False)
    m = art.meta
    z = np.load(ARTIFACTS_DIR / "text_emb.npz")
    emb, has = z["emb"].astype(np.float32), z["has_text"]
    t_idx = np.load(ARTIFACTS_DIR / "text_nbrs_idx.npy")[:, :k]
    i_idx = np.load(ARTIFACTS_DIR / "item_nbrs_idx.npy")
    i_sim = np.load(ARTIFACTS_DIR / "item_nbrs_sim.npy").astype(np.float32)
    n = m.n

    # Agreement on books that have both kinds of neighbour.
    i_top = i_idx[:, :k]
    i_count = (i_idx >= 0).sum(axis=1)
    both = has & (i_count >= k)
    overlap = np.array([len(np.intersect1d(t_idx[b], i_top[b])) / k for b in np.flatnonzero(both)])

    # Text cosine vs co-rating similarity over co-rating neighbour pairs (a sample of books).
    rng = np.random.default_rng(0)
    sample = rng.choice(np.flatnonzero(both), size=min(5000, int(both.sum())), replace=False)
    a, s = [], []
    for b in sample:
        j = i_idx[b][i_idx[b] >= 0]
        j = j[has[j]]
        a.append(emb[j] @ emb[b])
        s.append(i_sim[b][: len(i_idx[b][i_idx[b] >= 0])][has[i_idx[b][i_idx[b] >= 0]]])
    a, s = np.concatenate(a), np.concatenate(s)
    rho = spearman(a, s)
    rand = emb[rng.choice(np.flatnonzero(has), 20000)] @ emb[rng.choice(np.flatnonzero(has), 20000)].T
    rand_cos = float(np.mean(np.diag(rand)))

    # Thin co-rating evidence: books with fewer than k co-rating neighbours, by raters.
    raters = m.n_raters
    bands = [(0, 100), (100, 300), (300, 1000), (1000, 10**9)]
    thin_rows = []
    for lo, hi in bands:
        sel = (raters >= lo) & (raters < hi)
        thin_rows.append((lo, hi, int(sel.sum()), float((i_count[sel] < k).mean()), float(has[sel].mean()),
                          float(np.median(i_count[sel]))))

    # Composition: same genre / author / series among the top-k neighbours.
    def share(idx, attr, valid):
        rows = np.flatnonzero(valid)
        nb = idx[rows]
        ok = nb >= 0
        same = (attr[np.where(ok, nb, 0)] == attr[rows][:, None]) & ok
        return float(same.sum() / max(ok.sum(), 1))
    series = getattr(m, "series_id", None)
    comp = {
        "same parent genre": (share(t_idx, m.parent_genre, both & (m.parent_genre >= 0)),
                              share(i_top, m.parent_genre, both & (m.parent_genre >= 0))),
        "same author": (share(t_idx, m.author_id, both), share(i_top, m.author_id, both)),
    }
    if series is not None:
        comp["same series"] = (share(t_idx, series, both & (series >= 0)), share(i_top, series, both & (series >= 0)))
    t_pop = float(np.median(m.n_raters[t_idx[both][t_idx[both] >= 0]]))
    i_pop = float(np.median(m.n_raters[i_top[both][i_top[both] >= 0]]))

    lines = [f"# Description embeddings vs co-rating neighbours (top {k})", "",
             f"Model: {z['model']} reduced to {int(z['dim'])} dims. {int(has.sum()):,} of {n:,} books have a description "
             f"({has.mean():.1%}).", "",
             "## Agreement", "",
             f"- Of each book's top {k} text neighbours, **{overlap.mean():.1%}** are also among its top {k} co-rating "
             f"neighbours (median {np.median(overlap):.0%}; {both.sum():,} books with both).",
             f"- Over co-rating neighbour pairs, text cosine vs co-rating similarity: Spearman **{rho:.3f}** "
             f"({len(a):,} pairs). Mean text cosine of co-rating neighbours {a.mean():.3f} vs {rand_cos:.3f} for random pairs.",
             "", "## Where co-rating evidence is thin", "",
             f"| Raters | Books | Fewer than {k} co-rating neighbours | Has a description | Median co-rating neighbours |",
             "|---|---|---|---|---|"]
    for lo, hi, nb, thin, h, med in thin_rows:
        lab = f"{lo:,}+" if hi > 10**8 else f"{lo:,}–{hi - 1:,}"
        lines.append(f"| {lab} | {nb:,} | {thin:.1%} | {h:.1%} | {med:.0f} |")
    lines += ["", "## What the neighbours are", "", "| Share of top-k neighbours | Text | Co-rating |", "|---|---|---|"]
    for key, (tv, iv) in comp.items():
        lines.append(f"| {key} | {tv:.1%} | {iv:.1%} |")
    lines.append(f"| median raters of a neighbour | {t_pop:,.0f} | {i_pop:,.0f} |")
    out = EVAL_DIR / "text_probe.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
