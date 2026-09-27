"""Project paths and pipeline config (config/pipeline.yaml)."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("GOODREC_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
ARTIFACTS_DIR = Path(os.environ.get("GOODREC_ARTIFACTS", ROOT / "artifacts"))


@lru_cache
def load_config() -> dict:
    with open(CONFIG_DIR / "pipeline.yaml") as f:
        return yaml.safe_load(f)
