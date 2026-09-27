"""s00: download raw UCSD Goodreads files (resumable, never decompressed)."""

from goodrec.config import RAW_DIR, load_config
from goodrec.pipeline.io import download


def main() -> None:
    cfg = load_config()["data"]
    for name in cfg["files"].values():
        download(cfg["base_url"] + name, RAW_DIR / name)


if __name__ == "__main__":
    main()
