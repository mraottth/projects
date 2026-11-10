"""The location dealbreaker (asked 2026-11-04): with "Only show jobs in these locations" on, a job
whose location is clearly somewhere else is never scored, never on New & Noteworthy or in the
email, and hidden on All Jobs unless asked for. My Jobs is never filtered.

It's built to err towards keeping a job. A job is hidden only when its location is determined and
clearly elsewhere; everything else stays:
- remote jobs (the user's call: a remote job can be done from anywhere);
- a missing or unreadable location ("TBD", "Office", a city we don't know);
- several locations ("New York, NY; San Francisco, CA", "3 locations", "+2 more");
- a country with no state ("United States") when you asked for a city or state.

"Elsewhere" needs evidence: a US state (a full name, or ", VA"-style), a metro core name that
implies one ("San Francisco"), or a non-US country or big foreign city. A city on its own
("Arlington", "Portland") never decides it, since many names repeat across states.

Your entries are read generously. "DC" is the Washington, DC metro area (DC, Arlington, Bethesda,
Tysons...), as asked. "Washington" or "NY" match every place they could mean. An entry nothing here
recognizes is matched as written, and Settings shows which entries were recognized."""

import re
from dataclasses import dataclass, field

STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "DC": "district of columbia",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho", "IL": "illinois",
    "IN": "indiana", "IA": "iowa", "KS": "kansas", "KY": "kentucky", "LA": "louisiana",
    "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan", "MN": "minnesota",
    "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon",
    "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia",
    "WA": "washington", "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming", "PR": "puerto rico",
}  # fmt: skip
NAME_TO_STATE = {name: code for code, name in STATES.items()}


@dataclass(frozen=True)
class Metro:
    key: str
    label: str
    states: frozenset[str]
    # The state its core names imply ("San Francisco" alone is evidence of CA).
    home: str
    # Names that mean this metro wherever they appear (subject to a conflicting state).
    core: tuple[str, ...]
    # Other places in its commuting area. They match too, but never prove "elsewhere".
    cities: tuple[str, ...] = ()
    # What a user might type for it, beyond the core names and cities.
    aliases: tuple[str, ...] = ()


def _m(key, label, states, core, cities=(), aliases=()):
    codes = states.split()
    return Metro(key, label, frozenset(codes), codes[0], tuple(core), tuple(cities), tuple(aliases))


METROS = [
    _m("dc", "Washington, DC metro", "DC VA MD",
       ["washington dc", "dc", "district of columbia", "washington district of columbia", "dc metro",
        "dmv", "national capital region"],
       ["arlington", "alexandria", "bethesda", "north bethesda", "silver spring", "rockville",
        "gaithersburg", "germantown", "chevy chase", "takoma park", "college park", "hyattsville",
        "greenbelt", "laurel", "bowie", "largo", "landover", "lanham", "suitland", "oxon hill",
        "national harbor", "fort washington", "tysons", "tysons corner", "mclean", "vienna",
        "falls church", "fairfax", "reston", "herndon", "chantilly", "ashburn", "sterling",
        "leesburg", "manassas", "springfield", "annandale", "merrifield", "crystal city", "rosslyn",
        "ballston", "clarendon", "pentagon city", "woodbridge", "lorton", "dulles", "columbia",
        "fort meade", "potomac", "kensington", "wheaton", "olney", "burke", "centreville",
        "great falls", "dunn loring"],
       ["washington", "dc area", "washington dc area", "dc metro area", "washington metro",
        "washington metro area", "ncr", "northern virginia", "nova"]),
    _m("nyc", "New York City metro", "NY NJ CT",
       ["new york city", "nyc", "new york", "manhattan", "brooklyn", "queens", "bronx", "the bronx",
        "staten island"],
       ["jersey city", "hoboken", "newark", "long island city", "white plains", "yonkers",
        "stamford", "greenwich", "new rochelle", "tarrytown", "edison", "secaucus", "weehawken",
        "fort lee", "paramus", "hackensack", "morristown", "long island", "garden city",
        "hempstead", "mineola", "melville", "purchase"],
       ["ny", "new york metro", "tri state", "tristate", "nyc metro"]),
    _m("sf", "San Francisco Bay Area", "CA",
       ["san francisco", "bay area", "sf bay area", "san francisco bay area", "silicon valley", "sf"],
       ["oakland", "berkeley", "san jose", "palo alto", "mountain view", "menlo park",
        "redwood city", "sunnyvale", "santa clara", "cupertino", "san mateo", "emeryville",
        "fremont", "south san francisco", "burlingame", "foster city", "walnut creek",
        "pleasanton", "alameda", "milpitas", "los gatos", "san carlos", "belmont", "daly city",
        "san rafael", "hayward", "richmond"],
       ["sf"]),
    _m("la", "Los Angeles metro", "CA",
       ["los angeles", "greater los angeles"],
       ["santa monica", "culver city", "pasadena", "burbank", "glendale", "long beach", "irvine",
        "el segundo", "playa vista", "venice", "beverly hills", "west hollywood", "hollywood",
        "torrance", "manhattan beach", "marina del rey", "anaheim", "costa mesa", "newport beach",
        "santa ana", "thousand oaks", "woodland hills", "calabasas", "sherman oaks", "encino",
        "van nuys", "inglewood"],
       ["la", "socal", "southern california"]),
    _m("chicago", "Chicago metro", "IL IN WI",
       ["chicago"],
       ["evanston", "oak brook", "naperville", "schaumburg", "skokie", "oak park", "deerfield",
        "northbrook", "rosemont", "des plaines", "arlington heights", "downers grove", "lombard"]),
    _m("boston", "Boston metro", "MA NH RI",
       ["boston"],
       ["cambridge", "somerville", "waltham", "newton", "quincy", "brookline", "burlington",
        "lexington", "woburn", "needham", "watertown", "medford", "framingham", "wellesley",
        "bedford", "natick", "andover"]),
    _m("seattle", "Seattle metro", "WA",
       ["seattle"],
       ["bellevue", "redmond", "kirkland", "tacoma", "everett", "bothell", "renton", "issaquah",
        "tukwila", "kent"]),
    _m("austin", "Austin metro", "TX",
       ["austin"], ["round rock", "cedar park", "georgetown", "pflugerville", "san marcos"]),
    _m("denver", "Denver metro", "CO",
       ["denver"],
       ["boulder", "aurora", "lakewood", "englewood", "littleton", "broomfield", "golden",
        "centennial", "louisville", "westminster", "arvada"]),
    _m("atlanta", "Atlanta metro", "GA",
       ["atlanta"],
       ["alpharetta", "marietta", "decatur", "sandy springs", "roswell", "duluth", "norcross",
        "kennesaw", "dunwoody", "smyrna"]),
    _m("philadelphia", "Philadelphia metro", "PA NJ DE",
       ["philadelphia", "philly"],
       ["king of prussia", "conshohocken", "camden", "cherry hill", "wilmington", "malvern",
        "wayne", "plymouth meeting", "bala cynwyd"]),
    _m("baltimore", "Baltimore metro", "MD",
       ["baltimore"],
       ["towson", "columbia", "ellicott city", "owings mills", "annapolis", "linthicum",
        "hunt valley"]),
    _m("dallas", "Dallas-Fort Worth metro", "TX",
       ["dallas", "dfw", "dallas fort worth", "fort worth"],
       ["plano", "irving", "frisco", "arlington", "richardson", "addison", "grapevine", "mckinney",
        "denton", "carrollton"]),
    _m("houston", "Houston metro", "TX",
       ["houston"], ["the woodlands", "sugar land", "katy", "pasadena", "spring"]),
    _m("miami", "Miami metro", "FL",
       ["miami", "south florida"],
       ["fort lauderdale", "boca raton", "west palm beach", "coral gables", "miami beach", "doral",
        "hollywood"]),
    _m("minneapolis", "Minneapolis-St. Paul metro", "MN",
       ["minneapolis", "twin cities", "st paul", "saint paul"],
       ["bloomington", "eden prairie", "edina", "plymouth", "minnetonka", "eagan"]),
    _m("portland", "Portland, OR metro", "OR WA",
       [], ["portland", "beaverton", "hillsboro", "lake oswego", "vancouver"]),
    _m("sandiego", "San Diego metro", "CA",
       ["san diego"], ["la jolla", "carlsbad", "chula vista", "oceanside", "escondido", "del mar"]),
    _m("phoenix", "Phoenix metro", "AZ",
       ["phoenix"], ["scottsdale", "tempe", "mesa", "chandler", "gilbert", "glendale"]),
    _m("raleigh", "Raleigh-Durham (Research Triangle)", "NC",
       ["raleigh", "research triangle", "research triangle park", "rtp"],
       ["durham", "chapel hill", "cary", "morrisville"], ["triangle"]),
    _m("nashville", "Nashville metro", "TN",
       ["nashville"], ["franklin", "brentwood", "murfreesboro"]),
    _m("pittsburgh", "Pittsburgh metro", "PA", ["pittsburgh"]),
    _m("detroit", "Detroit metro", "MI",
       ["detroit"], ["ann arbor", "dearborn", "troy", "southfield", "novi"]),
    _m("saltlake", "Salt Lake City metro", "UT",
       ["salt lake city", "slc"], ["lehi", "provo", "sandy", "draper", "ogden"]),
]  # fmt: skip

US_NAMES = ("united states", "united states of america", "usa", "us", "u s", "u s a", "america")

# Non-US countries, and foreign cities whose name rarely means a US place.
COUNTRIES = {
    "canada": ["canada", "toronto", "montreal", "ottawa", "calgary", "ontario", "quebec",
               "british columbia", "alberta"],
    "uk": ["united kingdom", "uk", "england", "scotland", "wales", "great britain", "edinburgh",
           "manchester uk"],
    "ireland": ["ireland"],
    "india": ["india", "bangalore", "bengaluru", "mumbai", "new delhi", "delhi", "hyderabad",
              "pune", "chennai", "gurgaon", "gurugram", "noida"],
    "germany": ["germany", "berlin", "munich", "hamburg", "frankfurt"],
    "france": ["france"],
    "spain": ["spain", "madrid", "barcelona"],
    "portugal": ["portugal", "lisbon"],
    "netherlands": ["netherlands", "amsterdam"],
    "mexico": ["mexico", "mexico city", "guadalajara"],
    "brazil": ["brazil", "sao paulo"],
    "australia": ["australia", "sydney"],
    "new zealand": ["new zealand"],
    "singapore": ["singapore"],
    "japan": ["japan", "tokyo"],
    "israel": ["israel", "tel aviv"],
    "poland": ["poland", "warsaw", "krakow"],
    "philippines": ["philippines", "manila"],
    "sweden": ["sweden", "stockholm"],
    "switzerland": ["switzerland", "zurich", "geneva"],
    "italy": ["italy", "milan", "rome italy"],
    "argentina": ["argentina", "buenos aires"],
    "colombia": ["colombia", "bogota", "medellin"],
    "nigeria": ["nigeria", "lagos"],
    "kenya": ["kenya", "nairobi"],
    "south africa": ["south africa", "cape town", "johannesburg"],
    "china": ["china", "shanghai", "beijing"],
    "hong kong": ["hong kong"],
    "south korea": ["south korea", "korea", "seoul"],
    "taiwan": ["taiwan", "taipei"],
    "uae": ["united arab emirates", "uae", "dubai", "abu dhabi"],
    "denmark": ["denmark", "copenhagen"],
    "norway": ["norway", "oslo"],
    "finland": ["finland", "helsinki"],
    "belgium": ["belgium", "brussels"],
    "austria": ["austria"],
    "czechia": ["czech republic", "czechia", "prague"],
    "romania": ["romania", "bucharest"],
    "ukraine": ["ukraine", "kyiv"],
    "chile": ["chile", "santiago chile"],
    "costa rica": ["costa rica"],
    "pakistan": ["pakistan", "karachi", "lahore"],
    "egypt": ["egypt", "cairo"],
}  # fmt: skip

REMOTE_WORDS = ("remote", "anywhere", "wfh", "work from home", "work from anywhere", "distributed",
                "telecommute", "telework", "home based", "virtual", "fully remote")  # fmt: skip
# Wording that says the list isn't complete, or the job could be almost anywhere. A job with any
# of it is never hidden for missing your locations.
VAGUE = re.compile(
    r"\+\s*\d+|\b\d+\s+(?:other\s+)?(?:locations|offices|cities|sites)\b|\bother\s+locations\b"
    r"|\ball\s+locations\b|\bmultiple\b|\bvarious\b|\bseveral\b|\b(?:and|&)\s+(?:more|others)\b"
    r"|\betc\b|\bnationwide\b|\b(?:across|throughout)\s+the\s+(?:us|u\.s\.|united states|country)\b",
    re.IGNORECASE,
)
# Between places in a list. " or "/" and " are lowercase only: "OR" is Oregon ("Portland, OR").
LIST_SEPARATORS = re.compile(r"\s*(?:;|\||\n|/|\s&\s|\s(?:or|and)\s)\s*")
# How the work is done, not where: "New York City, NY (Hybrid)" read "NY (Hybrid)" as a second,
# unreadable place, which kept the job (caught live, 2026-11-10). Dropped before splitting.
ARRANGEMENT = re.compile(
    r"\(\s*(?:hybrid|on[- ]?site|in[- ]office|in[- ]person|office[- ]based)[^)]*\)"
    r"|\b(?:hybrid|on[- ]?site|in[- ]office|in[- ]person)\b\s*[-:]?",
    re.IGNORECASE,
)
# "Locations: Austin, TX" -- a label, not part of the place.
LABEL = re.compile(r"^\s*(?:job\s+)?locations?\s*:\s*", re.IGNORECASE)
# A comma part that only names the state/country of the place before it ("VA", "VA 22201",
# "Virginia", "USA", "France").
ZIP = re.compile(r"\s*\d{5}(?:-\d{4})?\s*$")
# A state code counts only where one is written: after a comma ("Austin, TX"), in a Workday-style
# "US-VA-Arlington", or "DC" on its own. A bare uppercase pair elsewhere ("IN OFFICE") doesn't.
STATE_ABBR = re.compile(r",\s*([A-Z]{2})\b|\bUSA?-([A-Z]{2})-|\b(DC)\b")
COMMA_WASHINGTON = re.compile(r",\s*washington\b(?!\s*,?\s*d\.?\s*c\b)", re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase words separated by single spaces, padded with a space at each end, so a phrase is
    matched as ' phrase ' on word boundaries. "D.C." and "D. C." become "dc", "St." becomes "st"."""
    text = re.sub(r"\bd\s*\.\s*c\b\.?", "dc", text, flags=re.IGNORECASE)
    text = text.lower().replace(".", "").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {text.strip()} "


def _has(norm: str, phrase: str) -> bool:
    return f" {phrase} " in norm


# ---------------------------------------------------------------------------------------------
# What the user asked for


@dataclass
class Place:
    kind: str  # "metro" | "state" | "us" | "country" | "remote" | "city" | "text"
    value: str
    label: str
    state: str | None = None  # for "city": the state it was given with, if any


@dataclass
class Wanted:
    entry: str
    places: list[Place] = field(default_factory=list)

    @property
    def recognized(self) -> bool:
        return any(p.kind != "text" for p in self.places)


def _title(name: str) -> str:
    title = " ".join(w if w == "of" else w.capitalize() for w in name.split())
    # Both also name a city, so say which one this is.
    return f"{title} State" if name in ("washington", "new york") else title


def _metro_names(metro: Metro) -> tuple[str, ...]:
    return (*metro.core, *metro.aliases, metro.key)


def resolve_entry(entry: str) -> Wanted:
    """Everything an entry could mean (every reading counts, so an ambiguous entry keeps more)."""
    norm = normalize(entry)
    phrase = norm.strip()
    wanted = Wanted(entry)
    if not phrase:
        return wanted
    if any(_has(norm, w) for w in REMOTE_WORDS):
        wanted.places.append(Place("remote", "remote", "Remote"))
        return wanted
    for metro in METROS:
        if phrase in _metro_names(metro) or phrase in metro.cities:
            wanted.places.append(Place("metro", metro.key, metro.label))
    upper = entry.strip().upper().replace(".", "")
    if upper in STATES:
        wanted.places.append(Place("state", upper, _title(STATES[upper])))
    elif phrase in NAME_TO_STATE:
        code = NAME_TO_STATE[phrase]
        wanted.places.append(Place("state", code, _title(STATES[code])))
    if phrase in US_NAMES:
        wanted.places.append(Place("us", "us", "Anywhere in the US"))
    for country, names in COUNTRIES.items():
        if phrase in names:
            wanted.places.append(Place("country", country, country.title()))
    if not wanted.places and "," in entry:
        # "Arlington, VA" / "Ellicott City, Maryland": the city's metro if it's in one in that
        # state, otherwise that city in that state.
        city_raw, _, state_raw = entry.rpartition(",")
        city, state_part = normalize(city_raw).strip(), state_raw.strip()
        state = state_part.upper() if state_part.upper() in STATES else NAME_TO_STATE.get(state_part.lower())
        if city and state:
            for metro in METROS:
                if state in metro.states and (city in metro.cities or city in _metro_names(metro)):
                    wanted.places.append(Place("metro", metro.key, metro.label))
            if not wanted.places:
                wanted.places.append(Place("city", city, f"{city.title()}, {state}", state))
    if not wanted.places:
        wanted.places.append(Place("text", phrase, f'"{entry.strip()}"'))
    return wanted


def describe(entries: list[str]) -> list[dict]:
    """How each entry is read, for Settings: what it matches and whether it was recognized."""
    out = []
    for entry in entries:
        wanted = resolve_entry(entry)
        labels = []
        for place in wanted.places:
            if place.kind == "metro":
                metro = next(m for m in METROS if m.key == place.value)
                examples = ", ".join(n.title() for n in metro.cities[:4])
                labels.append(f"{metro.label}" + (f" (incl. {examples}...)" if examples else ""))
            elif place.kind == "text":
                labels.append(f"locations containing {place.label}")
            elif (
                place.kind == "state" and place.value == "DC" and any(p.value == "dc" for p in wanted.places)
            ):
                continue  # already covered by the DC metro line
            else:
                labels.append(place.label)
        out.append({"entry": entry, "recognized": wanted.recognized, "matches": labels})
    return out


# ---------------------------------------------------------------------------------------------
# Reading a job's location


@dataclass
class JobPlace:
    norm: str
    states: set[str]  # stated outright, or implied by a metro's core name
    countries: set[str]  # non-US
    us: bool  # "United States" or similar


def read_job_location(location: str) -> JobPlace:
    """One place's text (see split_places). A state code written out ("Kansas City, MO") is
    trusted over state names inside a city's name ("Kansas", "Virginia Beach")."""
    norm = normalize(location)
    countries = {c for c, names in COUNTRIES.items() if any(_has(norm, n) for n in names)}
    us = any(_has(norm, n) for n in US_NAMES)
    explicit: set[str] = set()
    for match in STATE_ABBR.finditer(location):
        code = next(g for g in match.groups() if g)
        if code in STATES:
            explicit.add(code)
    if explicit:
        return JobPlace(norm, explicit, countries, us)
    states: set[str] = set()
    rest = norm
    for name in sorted(NAME_TO_STATE, key=len, reverse=True):
        if name == "washington":
            continue  # alone it's as likely DC as the state; see COMMA_WASHINGTON below
        if _has(rest, name):
            states.add(NAME_TO_STATE[name])
            rest = rest.replace(f" {name} ", " ")
    if COMMA_WASHINGTON.search(location):
        states.add("WA")
    for metro in METROS:
        if any(_has(norm, name) for name in metro.core):
            states.add(metro.home)
    return JobPlace(norm, states, countries, us)


def _only_region(part: str) -> bool:
    """Whether a comma part names only a state or a country (it belongs to the place before it)."""
    bare = ZIP.sub("", part).strip()
    if not bare:
        return True
    if bare.upper().replace(".", "") in STATES:
        return True
    phrase = normalize(bare).strip()
    return (
        phrase in NAME_TO_STATE or phrase in US_NAMES or any(phrase in names for names in COUNTRIES.values())
    )


def split_places(location: str) -> list[str]:
    """The separate places a location lists: "Austin, TX, Seattle, WA" -> ["Austin, TX",
    "Seattle, WA"]; "Washington, DC, United States" stays one place."""
    places = []
    for chunk in LIST_SEPARATORS.split(LABEL.sub("", ARRANGEMENT.sub(" ", location))):
        current: list[str] = []
        for part in (p.strip(" -–—:") for p in chunk.split(",")):
            if not part:
                continue
            if current and _only_region(part):
                current.append(part)
            else:
                if current:
                    places.append(", ".join(current))
                current = [part]
        if current:
            places.append(", ".join(current))
    return places


def _metro_matches(metro: Metro, job: JobPlace) -> bool:
    if job.states and not (job.states & metro.states):
        return False
    return any(_has(job.norm, n) for n in (*metro.core, *metro.cities))


def _place_matches(place: Place, job: JobPlace) -> bool:
    if place.kind == "metro":
        return _metro_matches(next(m for m in METROS if m.key == place.value), job)
    if place.kind == "state":
        return place.value in job.states or _has(job.norm, STATES[place.value])
    if place.kind == "us":
        return bool(job.states) or job.us or any(_metro_matches(m, job) for m in METROS)
    if place.kind == "country":
        return place.value in job.countries
    if place.kind == "city":
        return _has(job.norm, place.value) and (not job.states or place.state in job.states)
    if place.kind == "text":
        return _has(job.norm, place.value)
    return False


def verdict(location: str | None, remote_type: str | None, entries: list[str]) -> str:
    """'match', 'remote', 'unknown', 'multiple', or 'elsewhere'. Only 'elsewhere' is hidden."""
    if remote_type == "remote":
        return "remote"
    if not location or not location.strip():
        return "unknown"
    norm = normalize(location)
    if any(_has(norm, w) for w in REMOTE_WORDS):
        return "remote"
    wanted = [resolve_entry(e) for e in entries if e.strip()]
    places = [p for w in wanted for p in w.places]
    if not places:
        return "unknown"
    outcomes = []
    for piece in split_places(location) or [location]:
        job = read_job_location(piece)
        if any(_place_matches(p, job) for p in places):
            return "match"
        outcomes.append("elsewhere" if job.states or job.countries else "unknown")
    if len(outcomes) == 1 and not VAGUE.search(location):
        return outcomes[0]
    # Several places (or wording like "+2 more"): hidden only when every one is clearly elsewhere
    # and nothing says the list is incomplete.
    if all(o == "elsewhere" for o in outcomes) and not VAGUE.search(location):
        return "elsewhere"
    return "multiple"


def dealbreaker_on(settings: dict) -> bool:
    return bool(settings.get("location_dealbreaker")) and any(
        (e or "").strip() for e in settings.get("prefilter_locations") or []
    )


def blocks(job: dict, settings: dict) -> bool:
    """Whether the location dealbreaker hides this job."""
    if not dealbreaker_on(settings):
        return False
    return (
        verdict(job.get("location"), job.get("remote_type"), settings["prefilter_locations"]) == "elsewhere"
    )
