"""Streaming helpers: resumable downloads and gzip JSON-lines readers.

The raw UCSD files are several GB compressed, so nothing here ever decompresses
to disk or loads a whole file into memory.
"""

import gzip
import sys
from pathlib import Path
from typing import Iterator

import orjson
import polars as pl
import requests
from tqdm import tqdm


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Download url to dest, resuming a partial file via HTTP Range."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(requests.head(url, allow_redirects=True, timeout=30).headers["content-length"])
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if dest.exists() and dest.stat().st_size == total:
        print(f"  {dest.name}: already complete")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored Range; start over
            have = 0
        with open(part, "ab" if have else "wb") as f, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, desc=dest.name, file=sys.stdout
        ) as bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))
    if part.stat().st_size != total:
        raise IOError(f"{dest.name}: got {part.stat().st_size} bytes, expected {total}")
    part.rename(dest)


def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rb") as f:
        for line in f:
            yield orjson.loads(line)


class ColumnWriter:
    """Append rows column-wise and flush to one parquet file in bounded-memory batches.

    Usage: with ColumnWriter(path, {"a": pl.Int64, ...}) as w: w.add(a=1, ...)
    """

    def __init__(self, dest: Path, schema: dict, batch_size: int = 500_000):
        self.dest, self.schema, self.batch_size = dest, schema, batch_size
        self.tmp = dest.with_suffix(".tmp.parquet")
        self.cols = {k: [] for k in schema}
        self.writer = None
        self.n = 0

    def add(self, **row) -> None:
        for k, col in self.cols.items():
            col.append(row[k])
        if len(next(iter(self.cols.values()))) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        import pyarrow.parquet as pq

        tbl = pl.DataFrame(self.cols, schema=self.schema).to_arrow()
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp, tbl.schema, compression="zstd")
        self.writer.write_table(tbl)
        self.n += tbl.num_rows
        self.cols = {k: [] for k in self.schema}

    def __enter__(self):
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self._flush()
            self.writer.close()
            self.tmp.rename(self.dest)
        elif self.writer is not None:
            self.writer.close()
            self.tmp.unlink(missing_ok=True)


def skip_if_done(*outputs: Path, force: bool = False) -> bool:
    if not force and all(p.exists() for p in outputs):
        print(f"  skip: {', '.join(p.name for p in outputs)} exist (use --force)")
        return True
    return False
