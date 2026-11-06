from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

# A star rating. Goodreads now allows quarter stars (4.25, 4.5, 4.75) and they're used as they are; whole values stay ints (5, not 5.0) so responses,
# cache keys and the Assistant's text read as before. The training data is whole stars.
Stars = Annotated[float, AfterValidator(lambda v: int(v) if float(v).is_integer() else v)]


class Rating(BaseModel):
    id: int                       # Goodreads work_id
    rating: Stars = Field(ge=1, le=5)
    date: str | None = None       # YYYY-MM-DD read (else shelved / rated here); orders ratings for recency


class FilterSpec(BaseModel):
    genres: list[str] = []
    tags: list[str] = []            # subgenre tags, e.g. "Epic Fantasy"
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
    include_ya: bool = False
    include_series_continuations: bool = False


class RecommendRequest(BaseModel):
    ratings: list[Rating] = []
    read: list[int] = []          # work_ids read but unrated
    to_read: list[int] = []       # work_ids on the user's to-read shelf
    dismissed: list[int] = []
    filters: FilterSpec = FilterSpec()
    limit: int = Field(40, ge=1, le=200)
    offset: int = Field(0, ge=0)
    sort: Literal["match", "predicted"] = "match"                        # For you and to-read
    readers_sort: Literal["popularity", "rating", "predicted"] = "popularity"   # From similar readers
    readers_relative: bool = False   # vs all readers (popularity) or the Goodreads average (rating, predicted)


class BrowseRequest(BaseModel):
    """Explore the whole catalog: same filters as recommendations, no personalization in the ranking."""
    filters: FilterSpec = FilterSpec()
    sort: Literal["popular", "rating", "newest", "oldest", "title"] = "popular"
    ratings: list[Rating] = []    # optional: only used to show predicted ratings on the page
    limit: int = Field(40, ge=1, le=200)
    offset: int = Field(0, ge=0)


class BooksBatchRequest(BaseModel):
    ids: list[int] = Field(default_factory=list, max_length=5000)   # Goodreads work_ids


class InsightsRequest(BaseModel):
    ratings: list[Rating] = []
    read: list[int] = []
