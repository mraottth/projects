"""s12: sentence embeddings of each catalog book's description, and its nearest books by text.

Text per work: "title by author. description". The description is the catalog edition's (s09 shows the same one),
else the longest one among the work's English or unlabeled editions; HTML is stripped. Books with no description
get a zero vector and has_text = False, so they carry no text signal.

The model (config `text.model`, a small local sentence-transformers model) runs on Apple's GPU (MPS) when present.
Embeddings are L2-normalized, reduced to `text.dim` dimensions by PCA and normalized again, so a dot product is a
cosine similarity. Needs the `text` dependency group (torch), which serving never installs:

    uv run --group text python -m goodrec.pipeline.s12_text_embed

Outputs (artifacts):
  text_emb.npz         emb (N x dim float16), has_text (N bool), model, dim
  text_nbrs_idx.npy    N x K int32 nearest books by text (-1 = none), excluding the book itself
  text_nbrs_sim.npy    N x K float16 cosine similarity
"""

import argparse
import html
import re
import time

import numpy as np
import polars as pl

from goodrec.config import ARTIFACTS_DIR, INTERIM_DIR, load_config
from goodrec.pipeline.io import skip_if_done

TAG = re.compile(r"<[^>]+>")


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", text or ""))).strip()


def work_texts(languages: list[str]) -> tuple[list[str], np.ndarray]:
    """One text per catalog work (work_idx order) and whether it has a description."""
    cat = pl.read_parquet(INTERIM_DIR / "catalog.parquet", columns=["work_idx", "work_id", "book_id", "title", "author"])
    ed = (pl.scan_parquet(INTERIM_DIR / "editions.parquet").select("book_id", "work_id", "language_code")
            .join(cat.lazy().select("work_id"), on="work_id", how="semi"))
    desc = (pl.scan_parquet(INTERIM_DIR / "descriptions.parquet")
              .join(ed, on="book_id")
              .filter(pl.col("language_code").is_null() | pl.col("language_code").is_in(languages))
              .with_columns(n=pl.col("description").str.len_chars())
              .join(cat.lazy().select("book_id", main=pl.lit(True)), on="book_id", how="left")
              .sort(["work_id", "main", "n"], descending=[False, True, True], nulls_last=True)
              .group_by("work_id", maintain_order=True).first()
              .select("work_id", "description")
              .collect())
    cat = cat.join(desc, on="work_id", how="left").sort("work_idx")
    descs = [clean(d) for d in cat["description"].to_list()]
    has = np.array([len(d) >= 20 for d in descs])
    texts = [f"{t} by {a}. {d}" if h else "" for t, a, d, h in
             zip(cat["title"].to_list(), cat["author"].to_list(), descs, has)]
    return texts, has


def embed(texts: list[str], model_name: str, batch: int) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    # Longest texts first keeps padding per batch small.
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = model.encode([texts[i] for i in order], batch_size=batch, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    emb = np.empty_like(out)
    emb[order] = out
    return emb.astype(np.float32)


def pca(emb: np.ndarray, has: np.ndarray, dim: int) -> np.ndarray:
    """Project onto the top `dim` principal axes of the books with text; renormalize (rows without text stay 0)."""
    X = emb[has]
    mu = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    Z = np.zeros((len(emb), dim), np.float32)
    Z[has] = (X - mu) @ vt[:dim].T
    Z[has] /= np.linalg.norm(Z[has], axis=1, keepdims=True)
    return Z


def neighbors(Z: np.ndarray, has: np.ndarray, k: int, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    n = len(Z)
    idx = np.full((n, k), -1, np.int32)
    sim = np.zeros((n, k), np.float16)
    rows = np.flatnonzero(has)
    for s in range(0, len(rows), block):
        r = rows[s:s + block]
        S = Z[r] @ Z.T                                   # (block x n) cosines; rows without text score 0
        S[np.arange(len(r)), r] = -np.inf                # never your own neighbour
        S[:, ~has] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        ts = np.take_along_axis(S, top, axis=1)
        o = np.argsort(-ts, axis=1)
        idx[r] = np.take_along_axis(top, o, axis=1)
        sim[r] = np.take_along_axis(ts, o, axis=1)
    return idx, sim


def main(force: bool = False) -> None:
    cfg = load_config()
    tc = cfg["text"]
    out = ARTIFACTS_DIR / "text_emb.npz"
    out_idx, out_sim = ARTIFACTS_DIR / "text_nbrs_idx.npy", ARTIFACTS_DIR / "text_nbrs_sim.npy"
    if skip_if_done(out, out_idx, out_sim, force=force):
        return
    t0 = time.time()
    texts, has = work_texts(cfg["catalog"]["languages"])
    print(f"  {len(texts):,} works, {has.sum():,} with a description ({has.mean():.1%})")
    sel = np.flatnonzero(has)
    e = embed([texts[i] for i in sel], tc["model"], tc["batch"])
    full = np.zeros((len(texts), e.shape[1]), np.float32)
    full[sel] = e
    print(f"  embedded with {tc['model']} ({e.shape[1]} dims) in {time.time() - t0:.0f}s")
    np.save(INTERIM_DIR / "text_emb_full.npy", full.astype(np.float16))     # unreduced, for analysis
    Z = pca(full, has, tc["dim"])
    idx, sim = neighbors(Z, has, tc["k"])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=Z.astype(np.float16), has_text=has, model=tc["model"], dim=tc["dim"])
    np.save(out_idx, idx)
    np.save(out_sim, sim)
    print(f"  wrote {out.name}, text_nbrs_* (k={tc['k']}) in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
