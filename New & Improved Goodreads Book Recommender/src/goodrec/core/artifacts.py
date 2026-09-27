"""Load serving artifacts (written by s07/s08/s09/s10) into one object.

The same loader is used by the API and by `goodrec.eval`, so eval always
exercises the serving code path.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import orjson
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR
from goodrec.core.textnorm import ascii_fold


@dataclass
class ItemMeta:
    year: np.ndarray            # int32, 0 = unknown
    avg_rating: np.ndarray      # float32, Goodreads-wide
    ratings_count: np.ndarray   # int64, Goodreads-wide
    n_raters: np.ndarray        # int32, raters in our data
    bayes: np.ndarray           # float32, Bayesian-shrunk mean of our data
    log_pop: np.ndarray         # float32, log1p(readers in training data)
    reader_rate: np.ndarray     # float32, share of training users who read the book
    author_id: np.ndarray       # int64
    is_boxset: np.ndarray       # bool
    is_children: np.ndarray     # bool
    is_comic: np.ndarray        # bool
    is_ya: np.ndarray           # bool, young adult (s10: parent genre YA or YA vote share >= catalog.ya_min_share)
    series_id: np.ndarray       # int32, -1 = not in a series
    series_pos: np.ndarray      # float32, nan = unknown
    parent_genre: np.ndarray    # int16, -1 = none
    genre_names: list[str]
    tag_indptr: np.ndarray      # CSR over tag vocab: tags of item i = tag_ids[tag_indptr[i]:tag_indptr[i+1]]
    tag_ids: np.ndarray
    tag_names: list[str]
    search_text: np.ndarray     # object array: accent-folded lowercase "title author genre tags" for result filtering
    tag_owner: np.ndarray = None  # item index for each entry of tag_ids (inverse of tag_indptr)

    @property
    def n(self) -> int:
        return len(self.year)


@dataclass
class Artifacts:
    root: Path
    Y: np.ndarray                   # item factors (N x F)
    YtY: np.ndarray
    nbr_idx: np.ndarray             # (N x K) int32, -1 = none
    nbr_sim: np.ndarray             # (N x K) float32
    meta: ItemMeta
    user_factors: np.ndarray | None = None   # (U x F) float32, L2-normalized
    readers: sparse.csr_matrix | None = None  # (U x N) int8: 1-5 rating, 6 = read unrated
    params: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.root / "catalog.db"


def load_meta(root: Path) -> ItemMeta:
    z = np.load(root / "item_meta.npz", allow_pickle=False)
    names = orjson.loads((root / "item_meta_names.json").read_bytes())
    con = sqlite3.connect(root / "catalog.db")
    text = np.empty(len(z["year"]), dtype=object)
    for idx, title, author, genre, tags in con.execute("SELECT work_idx, title, author, parent_genre, tags FROM works"):
        text[idx] = ascii_fold(f"{title} {author} {genre or ''} {' '.join(orjson.loads(tags or '[]'))}").lower()
    con.close()
    return ItemMeta(
        **{k: z[k] for k in ("year", "avg_rating", "ratings_count", "n_raters", "bayes", "log_pop", "reader_rate",
                             "author_id", "is_boxset", "is_children", "is_comic", "is_ya", "series_id", "series_pos",
                             "parent_genre", "tag_indptr", "tag_ids")},
        genre_names=names["genres"], tag_names=names["tags"], search_text=text,
        tag_owner=np.repeat(np.arange(len(z["year"]), dtype=np.int32), np.diff(z["tag_indptr"])),
    )


def load_artifacts(root: Path = ARTIFACTS_DIR, with_readers: bool = True) -> Artifacts:
    root = Path(root)
    readers = None
    uf = None
    if with_readers and (root / "readers_csr.npz").exists():
        readers = sparse.load_npz(root / "readers_csr.npz").tocsr()
        uf = np.load(root / "user_factors.npy").astype(np.float32)
    manifest = root / "manifest.json"
    return Artifacts(
        root=root,
        Y=np.load(root / "item_factors.npy"),
        YtY=np.load(root / "yty.npy"),
        nbr_idx=np.load(root / "item_nbrs_idx.npy"),
        nbr_sim=np.load(root / "item_nbrs_sim.npy").astype(np.float32),
        meta=load_meta(root),
        user_factors=uf,
        readers=readers,
        params=orjson.loads(manifest.read_bytes()).get("params", {}) if manifest.exists() else {},
    )
