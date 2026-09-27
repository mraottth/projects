from pydantic import BaseModel, Field


class Rating(BaseModel):
    id: int                       # Goodreads work_id
    rating: int = Field(ge=1, le=5)


class FilterSpec(BaseModel):
    genres: list[str] = []
    authors_include: list[int] = []
    authors_exclude: list[int] = []
    year_min: int | None = None
    year_max: int | None = None
    min_avg_rating: float | None = None
    min_ratings_count: int | None = None
    max_ratings_count: int | None = None
    text: str = ""
    include_children: bool = False
    include_comics: bool = False
    include_series_continuations: bool = False


class RecommendRequest(BaseModel):
    ratings: list[Rating] = []
    read: list[int] = []          # work_ids read but unrated
    to_read: list[int] = []       # work_ids on the user's to-read shelf
    dismissed: list[int] = []
    filters: FilterSpec = FilterSpec()
    limit: int = Field(40, ge=1, le=200)
    offset: int = Field(0, ge=0)
