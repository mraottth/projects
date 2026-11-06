"""Project paths and pipeline config (config/pipeline.yaml).

Production uses the defaults. An experiment (e.g. a different ratings source) runs in its own folders so it can
never change what `make deploy` ships (see the Makefile's exp-* targets):
  GOODREC_ARTIFACTS  artifacts folder           (default artifacts/)
  GOODREC_DATA       intermediate data folder   (default data/interim/)
  GOODREC_EVAL       evaluation output folder   (default eval/: reports, runs, champion.json)
  GOODREC_CONFIG     a YAML overlay merged over config/pipeline.yaml (nested keys override, the rest is kept)
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("GOODREC_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = Path(os.environ.get("GOODREC_DATA", ROOT / "data" / "interim"))
ARTIFACTS_DIR = Path(os.environ.get("GOODREC_ARTIFACTS", ROOT / "artifacts"))
EVAL_DIR = Path(os.environ.get("GOODREC_EVAL", ROOT / "eval"))


def merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` into a copy of `base` (dicts merge key by key; anything else replaces)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache
def load_config() -> dict:
    with open(CONFIG_DIR / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)
    overlay = os.environ.get("GOODREC_CONFIG")
    if overlay:
        with open(overlay) as f:
            cfg = merge(cfg, yaml.safe_load(f) or {})
    return cfg
