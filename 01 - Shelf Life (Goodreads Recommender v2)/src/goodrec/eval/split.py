"""Per-user temporal hide-and-predict split for offline evaluation.

For each held-out user (s05's test users, never seen by any model), ratings are ordered by the date the
user read the book (`read_at`), or the date it was shelved (`date_added`) when no plausible read date is
given, and the most recent `holdout_frac` are hidden. Books with the same date stay on the same side: the
split moves to the date boundary closest to the target, because their order within a day is unknown. Everything dated on or after the split (ratings and read-unrated books)
is excluded from the user's input.

A user is evaluated if they have >= min_ratings ratings, >= min_visible visible ratings and
>= min_hidden_relevant hidden books rated >= relevant_min_rating. Eligible users are split once, with a
fixed seed, into a validation set (val_frac, for tuning) and a test set (reported results).

  uv run python -m goodrec.eval.split      # build data/interim/eval_split.npz and print a summary
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass

import numpy as np

from goodrec.config import INTERIM_DIR, load_config
from goodrec.core.scoring import UserInput

SPLIT_PATH = INTERIM_DIR / "eval_split.npz"
ORDER_BY = "read_date_then_added"   # part of the split's identity: changing the order rule rebuilds it
SPLIT_KEYS = ("holdout_frac", "min_ratings", "min_visible", "min_hidden_relevant", "relevant_min_rating",
              "val_frac", "seed")


@dataclass
class Case:
    user: int                 # original user_idx (ratings.parquet)
    is_test: bool             # False = validation set
    split_date: int           # yyyymmdd; hidden = ordered (read, else shelved) on or after this date
    visible: np.ndarray       # work_idx, oldest first (same-day order is a seeded shuffle)
    visible_r: np.ndarray     # ratings 1-5, aligned with `visible`
    read: np.ndarray          # read-unrated work_idx shelved before the split
    hidden: np.ndarray        # work_idx shelved on or after the split
    hidden_r: np.ndarray
    visible_d: np.ndarray | None = None   # yyyymmdd order date of each visible rating (not part of the split hash)

    def relevant(self, min_rating: int) -> set[int]:
        return set(self.hidden[self.hidden_r >= min_rating].tolist())

    def user_input(self, n: int) -> UserInput:
        """The model's input: the n most recent visible ratings (n < 0: all of them, plus read-unrated books and
        each rating's date, like an import). Truncated histories simulate people who rate a handful of books on
        the site, so they carry ratings only."""
        if n < 0:
            dates = dict(zip(self.visible.tolist(), self.visible_d.tolist())) if self.visible_d is not None else None
            return UserInput(ratings=dict(zip(self.visible.tolist(), self.visible_r.tolist())),
                             read=set(self.read.tolist()), dates=dates)
        return UserInput(ratings=dict(zip(self.visible[-n:].tolist(), self.visible_r[-n:].tolist())))


def choose_split(dates: np.ndarray, frac: float) -> int | None:
    """Index into date-sorted `dates` where the hidden part starts: the date boundary whose hidden share is
    closest to `frac` (ties: hide less). None if all ratings share one date."""
    n = len(dates)
    bounds = np.flatnonzero(dates[1:] != dates[:-1]) + 1      # positions where a new date starts
    if not len(bounds):
        return None
    target = n - int(round(frac * n))                          # number of visible ratings wanted
    return int(bounds[np.argmin(np.abs(bounds - target) * 2 + (bounds < target))])


def order_dates(read_date: np.ndarray, date_added: np.ndarray) -> np.ndarray:
    """The date each rating is ordered by: when the book was read if that's given and plausible
    (1900-2017; the data was collected in late 2017), otherwise when it was shelved."""
    ok = (read_date >= 19000101) & (read_date <= 20171231)
    return np.where(ok, read_date, date_added)


def split_user(dates: np.ndarray, items: np.ndarray, ratings: np.ndarray, cfg: dict, rng: np.random.Generator):
    """Split one user's history. `ratings` 0 = read but unrated. Returns (visible, visible_r, read, hidden,
    hidden_r, split_date, visible dates) or None if the user isn't eligible."""
    rated = ratings > 0
    d, it, r = dates[rated], items[rated], ratings[rated]
    if len(r) < cfg["min_ratings"]:
        return None
    # Date order; a seeded shuffle within each day (used only to pick "most recent n" for truncation).
    order = np.lexsort((rng.random(len(d)), d))
    d, it, r = d[order], it[order], r[order]
    cut = choose_split(d, cfg["holdout_frac"])
    if cut is None:
        return None
    hid_r = r[cut:]
    if cut < cfg["min_visible"] or (hid_r >= cfg["relevant_min_rating"]).sum() < cfg["min_hidden_relevant"]:
        return None
    split_date = int(d[cut])
    unrated = ~rated
    read = items[unrated & (dates < split_date)]
    return it[:cut], r[:cut], read, it[cut:], hid_r, split_date, d[:cut]


@dataclass
class Split:
    cases: list[Case]
    hash: str
    params: dict

    def select(self, which: str = "test", users: int | None = None, seed: int = 0) -> list[Case]:
        """Cases in the "test", "validation" or "all" set, optionally a fixed random subsample of `users`."""
        cs = [c for c in self.cases if which == "all" or c.is_test == (which == "test")]
        if users and users < len(cs):
            keep = np.sort(np.random.default_rng(seed).choice(len(cs), users, replace=False))
            cs = [cs[i] for i in keep]
        return cs


def build_split(cfg: dict | None = None) -> Split:
    import polars as pl  # pipeline dependency group

    cfg = cfg or load_config()["eval"]
    test_users = np.load(INTERIM_DIR / "test_users.npy")
    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_id", "work_idx"])
    r = (pl.read_parquet(INTERIM_DIR / "ratings.parquet")
           .filter(pl.col("user_idx").is_in(test_users.tolist()))
           .join(cat, on="work_id")
           .sort("user_idx"))
    rng = np.random.default_rng(cfg["seed"])
    cases = []
    for (u,), g in r.group_by(["user_idx"], maintain_order=True):
        out = split_user(order_dates(g["read_date"].to_numpy(), g["date"].to_numpy()),
                         g["work_idx"].to_numpy().astype(np.int32),
                         g["rating"].to_numpy().astype(np.int8), cfg, rng)
        if out:
            vis, vis_r, read, hid, hid_r, sd, vis_d = out
            cases.append(Case(int(u), True, sd, vis, vis_r, read, hid, hid_r, vis_d.astype(np.int32)))
    # Validation / test assignment: one fixed permutation of the eligible users.
    perm = np.random.default_rng(cfg["seed"] + 1).permutation(len(cases))
    for i in perm[: int(round(cfg["val_frac"] * len(cases)))]:
        cases[i].is_test = False
    params = {k: cfg[k] for k in SPLIT_KEYS} | {"order_by": ORDER_BY}
    return Split(cases, split_hash(cases, params), params)


def split_hash(cases: list[Case], params: dict) -> str:
    h = hashlib.sha256(repr(sorted(params.items())).encode())
    for c in cases:
        h.update(np.array([c.user, c.is_test, c.split_date], np.int64).tobytes())
        for a in (c.visible, c.visible_r, c.read, c.hidden, c.hidden_r):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def _flat(arrays: list[np.ndarray], dtype) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(arrays) + 1, np.int64)
    ptr[1:] = np.cumsum([len(a) for a in arrays])
    return (np.concatenate(arrays).astype(dtype) if arrays else np.zeros(0, dtype)), ptr


def save_split(s: Split, path=SPLIT_PATH) -> None:
    cs = s.cases
    arrays = {"user": np.array([c.user for c in cs], np.int64), "is_test": np.array([c.is_test for c in cs]),
              "split_date": np.array([c.split_date for c in cs], np.int32)}
    for name, dtype in (("visible", np.int32), ("visible_r", np.int8), ("read", np.int32), ("hidden", np.int32),
                        ("hidden_r", np.int8), ("visible_d", np.int32)):
        arrays[name], arrays[name + "_ptr"] = _flat([getattr(c, name) for c in cs], dtype)
    np.savez_compressed(path, hash=np.array(s.hash), params=np.array(repr(sorted(s.params.items()))), **arrays)


def load_split(path=SPLIT_PATH, cfg: dict | None = None) -> Split:
    """Load the saved split, rebuilding it if missing or built with different split settings."""
    cfg = cfg or load_config()["eval"]
    params = {k: cfg[k] for k in SPLIT_KEYS} | {"order_by": ORDER_BY}
    if path.exists():
        with np.load(path) as f:
            z = {k: f[k] for k in f.files}         # each NpzFile access re-reads the array
        if str(z["params"]) == repr(sorted(params.items())) and "visible_d" in z:
            def part(name, i):
                p = z[name + "_ptr"]
                return z[name][p[i]:p[i + 1]]
            cases = [Case(int(u), bool(t), int(sd), part("visible", i), part("visible_r", i), part("read", i),
                          part("hidden", i), part("hidden_r", i), part("visible_d", i))
                     for i, (u, t, sd) in enumerate(zip(z["user"], z["is_test"], z["split_date"]))]
            return Split(cases, str(z["hash"]), params)
    s = build_split(cfg)
    save_split(s, path)
    return s


def main() -> None:
    s = build_split()
    save_split(s)
    cs = s.cases
    n_test = sum(c.is_test for c in cs)
    vis = np.array([len(c.visible) for c in cs])
    rel = np.array([len(c.relevant(s.params["relevant_min_rating"])) for c in cs])
    frac = np.array([len(c.hidden) / (len(c.hidden) + len(c.visible)) for c in cs])
    print(f"  split {s.hash}: {len(cs):,} users ({n_test:,} test, {len(cs) - n_test:,} validation); "
          f"median visible {np.median(vis):.0f}, hidden relevant {np.median(rel):.0f}, "
          f"hidden share {np.median(frac):.2f} -> {SPLIT_PATH}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
