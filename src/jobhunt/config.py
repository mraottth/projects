import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("JOBHUNT_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
DB_PATH = Path(os.environ.get("JOBHUNT_DB_PATH", DATA_DIR / "jobhunt.db"))
RESUME_STORAGE_DIR = Path(os.environ.get("JOBHUNT_RESUME_DIR", DATA_DIR / "resumes"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

DEFAULT_SETTINGS = {
    "daily_llm_cap": 50,
    "prefilter_keywords_include": [],
    "prefilter_keywords_exclude": [],
    "prefilter_titles_include": [],
    "prefilter_sectors": [],
    "prefilter_locations": [],
    "prefilter_remote_only": False,
    "salary_floor": None,
}
