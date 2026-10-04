"""Write frontend/src/codeLinks.json: key -> "path#Lnn" for the code the About page links to.

Line numbers are looked up from the current source so the links never drift. Runs as part of
`make frontend`; fails loudly if a symbol is renamed or moved so a broken link can't ship.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# key: (path relative to the project root, symbol to anchor on or None for the whole file)
LINKS = {
    # data pipeline
    "pipeline_config": ("config/pipeline.yaml", None),
    "books_stream": ("src/goodrec/pipeline/s01_books.py", "main"),
    "ratings_to_works": ("src/goodrec/pipeline/s03_ratings.py", "main"),
    "catalog": ("src/goodrec/pipeline/s04_catalog.py", "main"),
    "parse_series": ("src/goodrec/core/textnorm.py", "parse_series"),
    "matrix_split": ("src/goodrec/pipeline/s05_matrix.py", "main"),
    "genres_apply": ("src/goodrec/pipeline/s06_genres.py", "apply"),
    "genre_map": ("config/shelf_genres.yaml", None),
    # similar books
    "item_knn": ("src/goodrec/pipeline/s07_item_knn.py", "item_knn"),
    "ii_weights": ("src/goodrec/core/scoring.py", "item_item_weights"),
    "ii_scores": ("src/goodrec/core/scoring.py", "item_item_scores"),
    "explain": ("src/goodrec/core/scoring.py", "explain"),
    # taste model
    "als_confidence": ("src/goodrec/pipeline/s08_als.py", "confidence_matrix"),
    "als_train": ("src/goodrec/pipeline/s08_als.py", "train_als"),
    "fold_in": ("src/goodrec/core/scoring.py", "fold_in"),
    "als_sweep": ("src/goodrec/eval/tune_als.py", "main"),
    # blend
    "blend": ("src/goodrec/core/scoring.py", "blend"),
    "prediction_floor": ("src/goodrec/core/scoring.py", "prediction_floor"),
    "fame_weight": ("src/goodrec/core/scoring.py", "fame_weight"),
    "filter_mask": ("src/goodrec/core/scoring.py", "filter_mask"),
    "next_in_series": ("src/goodrec/core/scoring.py", "next_in_series"),
    "recommend": ("src/goodrec/core/scoring.py", "recommend"),
    # readers like you
    "neighbors": ("src/goodrec/core/similar_readers.py", "neighbors"),
    "similar_readers": ("src/goodrec/core/similar_readers.py", "similar_readers"),
    # predicted rating
    "predict_ratings": ("src/goodrec/core/scoring.py", "predict_ratings"),
    "calibration": ("src/goodrec/core/scoring.py", "calibration"),
    "rating_track": ("src/goodrec/eval/run.py", "summarize_rating"),
    "rating_metrics_user": ("src/goodrec/eval/metrics.py", "rating_user"),
    "rating_settings": ("src/goodrec/eval/rating.py", "RatingSettings"),
    # evaluation
    "eval_split": ("src/goodrec/eval/split.py", "split_user"),
    "evaluate": ("src/goodrec/eval/run.py", "evaluate"),
    "metrics": ("src/goodrec/eval/metrics.py", "at_k"),
    "paired": ("src/goodrec/eval/metrics.py", "paired"),
    "baselines": ("src/goodrec/eval/models.py", "simple_baselines"),
    "legacy2023": ("src/goodrec/eval/legacy2023.py", None),
    "eval_reports": ("eval/reports", None),
    # serving
    "package": ("src/goodrec/pipeline/s10_package.py", "main"),
    "load_artifacts": ("src/goodrec/core/artifacts.py", "load_artifacts"),
    "api_recommend": ("src/goodrec/api/main.py", "recommend_route"),
    "api_search": ("src/goodrec/api/catalog.py", "search"),
    "csv_matching": ("src/goodrec/api/matching.py", "match_row"),
    "shelf_store": ("frontend/src/store.tsx", "ShelfProvider"),
    "chat_tools": ("src/goodrec/api/chat.py", "Toolbox"),
    "chat_loop": ("src/goodrec/api/chat.py", "run_chat"),
    "chat_digest": ("src/goodrec/api/chat.py", "library_digest"),
}


def find_line(path: Path, symbol: str) -> int:
    pat = re.compile(rf"^\s*(?:async\s+)?(?:def|class|export\s+function|function)\s+{re.escape(symbol)}\b")
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if pat.match(line):
            return n
    raise SystemExit(f"code_links: symbol {symbol!r} not found in {path.relative_to(ROOT)}")


def main() -> None:
    out = {}
    for key, (rel, symbol) in LINKS.items():
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"code_links: {rel} does not exist")
        out[key] = f"{rel}#L{find_line(path, symbol)}" if symbol else rel
    dest = ROOT / "frontend" / "src" / "codeLinks.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"code_links: wrote {len(out)} links to {dest.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
