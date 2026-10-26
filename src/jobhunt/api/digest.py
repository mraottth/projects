from fastapi import APIRouter

from jobhunt.core.digest import get_digest, get_digest_dates, today_str

router = APIRouter(prefix="/api/digest", tags=["digest"])


@router.get("/today")
def digest_today():
    date = today_str()
    return {"date": date, "entries": get_digest(date)}


@router.get("/dates")
def digest_dates():
    return get_digest_dates()


@router.get("/{digest_date}")
def digest_by_date(digest_date: str):
    return {"date": digest_date, "entries": get_digest(digest_date)}
