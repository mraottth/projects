"""s10: package serving artifacts + sanity checks.

Outputs (artifacts):
  readers_csr.npz        rows aligned with user_factors.npy; int8 values 1-5 = rating, 6 = read unrated
  item_meta.npz          per-item numpy arrays for filtering/scoring (see core.artifacts.ItemMeta)
  item_meta_names.json   parent genre + tag vocabularies
  starter_shelf.json     popular, genre-diverse books for the empty "Rate books" page
  manifest.json          counts, params, sha256 of every artifact
"""

import argparse
import datetime as dt
import hashlib

import numpy as np
import orjson
import polars as pl
from scipy import sparse

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, load_config
from goodrec.pipeline.s06_genres import PARENTS

READ_UNRATED = 6


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_meta(cat: pl.DataFrame, R: sparse.csr_matrix, Read: sparse.csr_matrix, bayes_m: float) -> tuple[dict, dict]:
    n_users = R.shape[0]
    readers = np.asarray((R > 0).sum(axis=0)).ravel() + np.asarray(Read.sum(axis=0)).ravel()
    mu = float(R.data.mean())
    n_raters = cat["n_raters"].to_numpy().astype(np.float32)
    data_avg = cat["data_avg"].to_numpy().astype(np.float32)

    series_keys = [f"{s.lower()}|{a}" if s else None
                   for s, a in zip(cat["series_name"].to_list(), cat["author_id"].to_list())]
    sid_map: dict[str, int] = {}
    series_id = np.array([sid_map.setdefault(k, len(sid_map)) if k else -1 for k in series_keys], dtype=np.int32)

    # Young adult: parent genre YA, or a large share of readers' UCSD genre votes are "young-adult".
    gcols = [c for c in cat.columns if c.startswith("g_")]
    votes = cat.select(gcols).fill_null(0).to_numpy().astype(np.float64)
    ya_share = np.divide(cat["g_ya"].fill_null(0).to_numpy(), votes.sum(axis=1),
                         out=np.zeros(cat.height), where=votes.sum(axis=1) > 0)
    is_ya = ((cat["parent_genre"] == "Young Adult").fill_null(False).to_numpy()
             | (ya_share >= load_config()["catalog"]["ya_min_share"]))

    parent_idx = {p: i for i, p in enumerate(PARENTS)}
    parent = np.array([parent_idx.get(p, -1) if p else -1 for p in cat["parent_genre"].to_list()], dtype=np.int16)

    tag_vocab: dict[str, int] = {}
    indptr, ids = [0], []
    for tags in cat["tags"].to_list():
        ids.extend(tag_vocab.setdefault(t, len(tag_vocab)) for t in (tags or []))
        indptr.append(len(ids))

    meta = dict(
        year=cat["year"].fill_null(0).to_numpy().astype(np.int32),
        avg_rating=cat["avg_rating"].fill_null(0).to_numpy().astype(np.float32),
        ratings_count=cat["ratings_count"].fill_null(0).to_numpy().astype(np.int64),
        n_raters=n_raters.astype(np.int32),
        bayes=((data_avg * n_raters + bayes_m * mu) / (n_raters + bayes_m)).astype(np.float32),
        log_pop=np.log1p(readers).astype(np.float32),
        reader_rate=(readers / n_users).astype(np.float32),
        author_id=cat["author_id"].fill_null(-1).to_numpy().astype(np.int64),
        is_boxset=cat["is_boxset"].to_numpy().astype(bool),
        is_children=cat["is_children"].to_numpy().astype(bool),
        is_comic=cat["is_comic"].to_numpy().astype(bool),
        is_ya=is_ya,
        series_id=series_id,
        series_pos=cat["series_pos"].fill_null(np.nan).to_numpy().astype(np.float32),
        parent_genre=parent,
        tag_indptr=np.asarray(indptr, dtype=np.int32),
        tag_ids=np.asarray(ids, dtype=np.int32),
    )
    return meta, {"genres": PARENTS, "tags": list(tag_vocab)}


def population_stats(cat: pl.DataFrame, meta: dict, names: dict) -> dict:
    """Reader-population distributions for the "Your books" insights.

    books_read: per reader, catalog books rated or marked read (the same basis as a user's matched shelf).
    harshness:  per reader with >= 10 ratings, mean(rating - item mean) using the Bayesian item mean
                (the baseline the predicted rating uses); negative = tougher than typical.
    genres:     mean rating and count of all ratings per parent genre.
    """
    widx = cat.select("work_id", "work_idx")
    r = pl.scan_parquet(INTERIM_DIR / "ratings.parquet").join(widx.lazy(), on="work_id").select(
        "user_idx", "work_idx", "rating").collect()
    counts = r.group_by("user_idx").len()["len"].to_numpy()
    vals, freq = np.unique(counts, return_counts=True)
    cum = np.cumsum(freq) / freq.sum()
    rated = r.filter(pl.col("rating") > 0)
    widx_arr = rated["work_idx"].to_numpy()
    rated = rated.with_columns(item_mean=pl.Series(meta["bayes"][widx_arr].astype(np.float64)),
                               genre=pl.Series(meta["parent_genre"][widx_arr].astype(np.int64)))
    bias = (rated.group_by("user_idx").agg(pl.len().alias("n"), (pl.col("rating") - pl.col("item_mean")).mean().alias("b"))
                 .filter(pl.col("n") >= 10)["b"].to_numpy())
    by_genre = rated.filter(pl.col("genre") >= 0).group_by("genre").agg(pl.len().alias("n"), pl.col("rating").mean().alias("avg"))
    return {
        "books_read": {"values": vals.tolist(), "cum_frac": np.round(cum, 6).tolist(), "n_readers": int(len(counts)),
                       "median": float(np.median(counts)), "p90": float(np.percentile(counts, 90))},
        "harshness": {"quantiles": np.round(np.quantile(bias, np.linspace(0, 1, 1001)), 4).tolist(),
                      "n_readers": int(len(bias)), "median": float(np.median(bias))},
        "genres": {names["genres"][int(k)]: {"avg": round(float(a), 3), "n_ratings": int(n)}
                   for k, n, a in by_genre.iter_rows()},
        # Share of all 1-5 star ratings at each level: the prior for calibrating predicted ratings.
        "rating_dist": np.round(np.bincount(rated["rating"].to_numpy(), minlength=6)[1:] / rated.height, 5).tolist(),
    }


def starter_shelf(cat: pl.DataFrame, meta: dict, per_genre: int = 5, max_genres: int = 16) -> list[int]:
    """Most-read standalone/first-in-series adult books, round-robin across the biggest genres."""
    ok = (~meta["is_boxset"] & ~meta["is_children"] & ~meta["is_comic"]
          & ~((meta["series_id"] >= 0) & (meta["series_pos"] > 1)) & (meta["parent_genre"] >= 0))
    idx = np.flatnonzero(ok)  # catalog is sorted by popularity already
    by_genre: dict[int, list[int]] = {}
    for i in idx:
        g = int(meta["parent_genre"][i])
        if len(by_genre.setdefault(g, [])) < per_genre:
            by_genre[g].append(int(i))
    genres = sorted(by_genre, key=lambda g: min(by_genre[g]))[:max_genres]
    return [by_genre[g][k] for k in range(per_genre) for g in genres if k < len(by_genre[g])]


def main(force: bool = False) -> None:
    cfg = load_config()
    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet")
    genres_path = INTERIM_DIR / "work_genres.parquet"
    if genres_path.exists():
        cat = cat.join(pl.read_parquet(genres_path), on="work_id", how="left")
    else:
        print("  note: no work_genres.parquet; packaging without genres")
        cat = cat.with_columns(parent_genre=pl.lit(None, pl.String), tags=pl.lit(None, pl.List(pl.String)))
    cat = cat.sort("work_idx")

    R = sparse.load_npz(INTERIM_DIR / "R_train.npz").tocsr()
    Read = sparse.load_npz(INTERIM_DIR / "Read_train.npz").tocsr()
    meta, names = build_meta(cat, R, Read, cfg["blend"]["bayes_m"])
    np.savez(ARTIFACTS_DIR / "item_meta.npz", **meta)
    (ARTIFACTS_DIR / "item_meta_names.json").write_bytes(orjson.dumps(names))

    rows = np.load(ARTIFACTS_DIR / "user_rows.npy")
    readers = (R + Read.multiply(READ_UNRATED)).tocsr()[rows].astype(np.int8)
    readers.sort_indices()
    sparse.save_npz(ARTIFACTS_DIR / "readers_csr.npz", readers)

    (ARTIFACTS_DIR / "population_stats.json").write_bytes(orjson.dumps(population_stats(cat, meta, names)))

    starter = starter_shelf(cat, meta)
    (ARTIFACTS_DIR / "starter_shelf.json").write_bytes(orjson.dumps(starter))

    # ---- sanity checks
    N = cat.height
    Y = np.load(ARTIFACTS_DIR / "item_factors.npy")
    nbr_idx = np.load(ARTIFACTS_DIR / "item_nbrs_idx.npy")
    nbr_sim = np.load(ARTIFACTS_DIR / "item_nbrs_sim.npy")
    uf = np.load(ARTIFACTS_DIR / "user_factors.npy")
    assert (cat["work_idx"].to_numpy() == np.arange(N)).all(), "work_idx must be dense 0..N-1"
    assert Y.shape[0] == N and nbr_idx.shape[0] == N, "model artifacts don't match catalog size"
    assert np.isfinite(Y).all() and np.isfinite(uf.astype(np.float32)).all(), "NaN/inf in factors"
    assert nbr_idx.max() < N and ((nbr_sim >= -1) & (nbr_sim <= 1.001)).all(), "bad neighbor lists"
    assert readers.shape == (len(uf), N), "readers_csr must align with user_factors"
    genre_cov = float((meta["parent_genre"] >= 0).mean())
    tag_cov = float((np.diff(meta["tag_indptr"]) > 0).mean())
    if genres_path.exists():
        assert genre_cov >= 0.95, f"only {genre_cov:.1%} of works have a parent genre"

    files = sorted(p for p in ARTIFACTS_DIR.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "data": {"ratings_source": cfg["data"]["files"]["ratings"]},
        "counts": {"works": N, "train_users": int(R.shape[0]), "train_ratings": int(R.nnz),
                   "neighbor_users": int(len(uf)), "tags": len(names["tags"])},
        "coverage": {"parent_genre": round(genre_cov, 4), "tags": round(tag_cov, 4),
                     "cover_url": round(float(cat["cover_url"].is_not_null().mean()), 4)},
        "params": {k: cfg[k] for k in ("catalog", "item_knn", "als", "blend", "similar_readers")},
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files},
    }
    (ARTIFACTS_DIR / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"  packaged {len(files)} artifacts ({total / 1e6:.0f} MB); coverage {manifest['coverage']}; "
          f"starter shelf {len(starter)} books")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    main(**vars(p.parse_args()))
