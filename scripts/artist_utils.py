"""Shared text and credit-parsing helpers.

Every script imports its normalization and credit splitting from here so the
rules cannot drift apart between scripts. Do not copy these functions into a
script -- import them.

Billboard writes an artist credit as one string, e.g.
"Cardi B Featuring 21 Savage". split_credit() turns that into a list of
individual artists, primary first, while leaving three kinds of name intact:

  1. PROTECTED_ACTS -- names that contain "," or "&" but are one act
     ("Earth, Wind & Fire", "Tyler, The Creator", "Maddie & Tae").
  2. Name suffixes -- ", Jr." / ", Sr." / ", II" and friends are part of a
     person's name, not a separator ("Grover Washington, Jr.").
  3. SUPERGROUP_BILLINGS -- a project name followed by its members
     ("¥$: Ye & Ty Dolla $ign"). These yield the group name AND the members,
     so each member's own chart history still counts.
"""

import re

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Separators
# ---------------------------------------------------------------------------
# Word separators take \b so they cannot fire inside a name (Charli XCX keeps
# its x). Symbol separators must NOT take \b: a word boundary needs a word
# character on one side, and " & " has spaces on both, so \b&\b never matches.
# "+" is deliberately absent, which keeps duo names such as "Dan + Shay" whole.
CREDIT_SPLIT = re.compile(
    r"\s*(?:"
    r"\b(?:a\s+)?duet\s+with\b"          # "Patti Austin A Duet With James Ingram"
    r"|\bfeaturing\b|\bfeauring\b|\bfeatauring\b"   # last two are Billboard typos
    r"|\bfeat\b\.?|\bft\b\.?"
    r"|\bwith\b|\bw/"
    r"|\bvs\b\.?|\bversus\b|\bintroducing\b|\bpresents\b"
    r"|\bx\b|&|,"
    r")\s*",
    re.IGNORECASE,
)

# Separator spellings that appear in the archive only as Billboard mistakes or
# abbreviations. Found by fuzzy-matching every word in every credit against the
# real separators; everything else that scored highly ("wit" in "A Boogie Wit da
# Hoodie", "fat" in "Fat Boys", the artist "Yeat") is a real name, so these are
# listed literally rather than matched by similarity.
SEPARATOR_VARIANTS = re.compile(
    r"\bfeauring\b|\bfeatauring\b|\bft\b\.?|\bw/|\ba\s+duet\s+with\b", re.IGNORECASE
)

FEATURE_PATTERN = re.compile(
    r"\bfeaturing\b|\bfeauring\b|\bfeatauring\b|\bfeat\b\.?|\bft\b\.?", re.IGNORECASE
)

# Bump whenever credit splitting or candidate selection changes. API scripts
# stamp this on each row they write, so --retry can find rows fetched under
# older rules.
#   2 = fixed &/, separators, protected acts, supergroup billings
#   3 = prefer the candidate matching the FULL Billboard credit when scores tie
#   4 = fall back to the release date when a recording has no first-release-date
MATCH_RULES_VERSION = 4

# A match at or above this score is trusted. Shared by the lyrics and
# MusicBrainz fetchers and by the quality report, so "matched" means the same
# thing everywhere.
MATCH_SCORE_THRESHOLD = 72

# ", Jr." and friends are part of the name. Matched with or without the space,
# because Billboard prints both ("Ray Parker,Jr." and "Grover Washington, Jr.").
NAME_SUFFIX = re.compile(r",\s*(jr|sr|ii|iii|iv)\b\.?", re.IGNORECASE)

# Sentinels used while splitting, then restored. Chosen because they cannot
# appear in a Billboard credit.
_SUFFIX_MARK = "\x01"
_PROTECT_FMT = "\x00{}\x00"
_PAREN_FMT = "\x02{}\x02"

# A trailing parenthetical is usually a disambiguator that belongs to the name
# ("Sylvia (r&b)"), so splitting inside it would invent an artist called "b".
# One that carries a feature credit ("... (Feat. Mase)") is a real separator.
PAREN_GROUP = re.compile(r"\([^)]*\)")


# ---------------------------------------------------------------------------
# Names that must never be split
# ---------------------------------------------------------------------------
# Derived by scanning the full 1958-2026 archive for credits containing "," or
# "&" that are billed as a recurring unit (the whole string appears as its own
# credit on 3+ songs, or before "Featuring" 3+ times) and whose members do not
# each have a separate solo chart career. Joint albums by two established
# artists ("Future & Metro Boomin", "Lil Baby & Lil Durk") are deliberately NOT
# here: those are two artists and should split.
PROTECTED_ACTS = [
    '10,000 Maniacs',
    '? (Question Mark) & The Mysterians',
    'Aly & AJ',
    'Archie Bell & The Drells',
    'Ashford & Simpson',
    'B. Bumble & The Stingers',
    'BeBe & CeCe Winans',
    'Big & Rich',
    'Bill Deal & The Rhondels',
    'Blood, Sweat & Tears',
    'Bob Seger & The Silver Bullet Band',
    'Bobby Taylor & The Vancouvers',
    'Bobby Womack & Peace',
    "Booker T. & The MG's",
    'Brenda & The Tabulations',
    'Brooks & Dunn',
    'Bruce Hornsby & The Range',
    'Captain & Tennille',
    'Carl Dobkins, Jr.',
    'Chad & Jeremy',
    'Cheech & Chong',
    'Cliff Nobles & Co.',
    'Cornelius Brothers & Sister Rose',
    'Crosby, Stills & Nash',
    'Crosby, Stills, Nash & Young',
    'D.J. Jazzy Jeff & The Fresh Prince',
    'Dale & Grace',
    'Danny & The Juniors',
    'Delaney & Bonnie',
    'Delaney & Bonnie & Friends',
    'Dennis Coffey & The Detroit Guitar Band',
    'Dino, Desi & Billy',
    'Disco Tex & The Sex-O-Lettes',
    'Donny & Marie Osmond',
    'Earth, Wind & Fire',
    'Ecstasy, Passion & Pain',
    'Emerson, Lake & Palmer',
    'England Dan & John Ford Coley',
    'Eric Burdon & The Animals',
    'Faith, Hope And Charity',
    'Ferrante & Teicher',
    'Five Stairsteps & Cubie',
    'Franke & The Knockouts',
    'Garnet Mimms & The Enchanters',
    'Hamilton, Joe Frank & Reynolds',
    'Heavy D & The Boyz',
    'Herb Alpert & The Tijuana Brass',
    'Hootie & The Blowfish',
    'Huey Lewis & The News',
    'Ike & Tina Turner',
    'James & Bobby Purify',
    'Jan & Dean',
    'Jay & The Americans',
    'Joan Jett & The Blackhearts',
    'Joey Dee & the Starliters',
    'John Cafferty & The Beaver Brown Band',
    'Jr. Walker & The All Stars',
    'K-Ci & JoJo',
    'Kenny Rogers & The First Edition',
    'King Curtis & The Kingpins',
    'Kool & The Gang',
    'LeBlanc & Carr',
    'Lil Jon & The East Side Boyz',
    'Lisa Loeb & Nine Stories',
    'Loggins & Messina',
    'Macklemore & Ryan Lewis',
    'Maddie & Tae',
    'Marilyn McCoo & Billy Davis Jr.',
    'Marky Mark & The Funky Bunch',
    'Martha & The Vandellas',
    'Martha Reeves & The Vandellas',
    'Maurice Williams & The Zodiacs',
    'Michael Nesmith & The First National Band',
    'Mickey & Sylvia',
    'Mumford & Sons',
    'Nino Tempo & April Stevens',
    'Oscar Toney, Jr.',
    'Otis & Carla',
    'Paul Revere & The Raiders',
    'Peaches & Herb',
    'Peggy Scott & Jo Jo Benson',
    'Peter, Paul & Mary',
    'Puff Daddy & The Family',
    'R. Kelly & Public Announcement',
    'Ray Parker Jr. & Raydio',
    'Ray, Goodman & Brown',
    'Rene & Angela',
    'Sam & Dave',
    'Santo & Johnny',
    'Seals & Crofts',
    'Selena Gomez & The Scene',
    "Sergio Mendes & Brasil '66",
    'Simon & Garfunkel',
    'Sly & The Family Stone',
    'Sonny & Cher',
    'The Mamas & The Papas',
    'Timbaland & Magoo',
    'Tommy Boyce & Bobby Hart',
    'Tyler, The Creator',
    'Waylon & Willie',
    'Wisin & Yandel',
    'Yarbrough & Peoples',
    'Young T & Bugsey',
    'Zac Efron & Vanessa Anne Hudgens',
]

# Longest first so "Delaney & Bonnie & Friends" is matched before the shorter
# "Delaney & Bonnie" that it contains.
def _compile_protected(names):
    return [
        (name, re.compile(re.escape(name), re.IGNORECASE))
        for name in sorted(names, key=len, reverse=True)
    ]

# Project/supergroup names that Billboard bills as "GROUP: members",
# "GROUP, members" or "GROUP (members)" -- it uses all three for the same act.
# The group is returned as an artist in its own right AND the members are split
# out, so a member's prior chart history still counts toward the song. Matched
# only against these exact names, so lookalikes such as "M:G" or "Sylvia (r&b)"
# are left alone.
SUPERGROUP_BILLINGS = [
    '¥$',
    'HUNTR/X',
    'Saja Boys',
    'Silk Sonic',
    'Stars On 54',
    'THE ANXIETY',
    'The Swell Season',
]

_SUPERGROUP_PATTERNS = [
    (name, re.compile(rf"^{re.escape(name)}\s*(?::|,|\()\s*", re.IGNORECASE))
    for name in sorted(SUPERGROUP_BILLINGS, key=len, reverse=True)
]

# Group names are also protected, so one billed without its members
# ("HUNTR/X" alone) is not cut apart by the \bx\b separator.
_PROTECTED_PATTERNS = _compile_protected(PROTECTED_ACTS + SUPERGROUP_BILLINGS)


# ---------------------------------------------------------------------------
# Normalization (per CLAUDE.md rules)
# ---------------------------------------------------------------------------
def cell_text(value) -> str:
    """A CSV cell as clean text, treating pandas' NaN as empty.

    Needed because float('nan') is truthy, so `value or ""` keeps the NaN and
    str() turns it into the non-empty string "nan" -- which makes an empty cell
    look like real content. Use this for any value read back out of a CSV
    before testing it for emptiness or matching against it.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_text(s: str) -> str:
    """Lowercase, drop parentheticals, drop feat./featuring, strip punctuation.

    Used as the matching key for lyrics/metadata lookups and as the artist key
    for history counting.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\bfeat\.?\b|\bfeaturing\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Credit splitting
# ---------------------------------------------------------------------------
def _tidy(part: str) -> str:
    """Clean one split part: drop stray colons and half-open parentheses.

    Splitting inside "(Featuring T.I. & T-Pain)" leaves "T-Pain)" with a dangling
    bracket, but "Sylvia (r&b)" is balanced and must keep both of its own.
    """
    part = part.strip().strip(":").strip()
    while part and part[-1] in "()" and part.count("(") != part.count(")"):
        part = part[:-1].strip()
    while part and part[0] in "()" and part.count("(") != part.count(")"):
        part = part[1:].strip()
    return part


def split_credit(credit: str) -> list[str]:
    """Split a Billboard credit into individual artists, primary first.

    >>> split_credit("Cardi B Featuring 21 Savage")
    ['Cardi B', '21 Savage']
    >>> split_credit("Earth, Wind & Fire")
    ['Earth, Wind & Fire']
    >>> split_credit("¥$: Ye & Ty Dolla $ign")
    ['¥$', 'Ye', 'Ty Dolla $ign']
    """
    if not isinstance(credit, str) or not credit.strip():
        return []

    working = credit.strip()
    leading = []

    # 1. Supergroup billing: keep the project name, then split the members.
    for name, pattern in _SUPERGROUP_PATTERNS:
        match = pattern.match(working)
        if match:
            leading.append(name)
            working = working[match.end():].replace(")", " ")
            break

    # 2. Hide protected names behind sentinels so the splitter cannot cut them.
    restore = {}
    for i, (name, pattern) in enumerate(_PROTECTED_PATTERNS):
        if pattern.search(working):
            token = _PROTECT_FMT.format(i)
            working = pattern.sub(token, working)
            restore[token] = name

    # 3. Hide parentheticals that are part of the name rather than a separator.
    def _hide_paren(match):
        text = match.group(0)
        if FEATURE_PATTERN.search(text):
            return text  # "(Feat. Mase)" really does introduce another artist
        token = _PAREN_FMT.format(len(restore))
        restore[token] = text
        return token

    working = PAREN_GROUP.sub(_hide_paren, working)

    # 4. Hide the comma in name suffixes (", Jr." etc).
    working = NAME_SUFFIX.sub(lambda m: _SUFFIX_MARK + m.group(0)[1:], working)

    # 5. Split, then put everything back.
    parts = []
    for part in CREDIT_SPLIT.split(working):
        part = part.replace(_SUFFIX_MARK, ",")
        for token, name in restore.items():
            part = part.replace(token, name)
        part = _tidy(part)
        if part:
            parts.append(part)

    # A supergroup member list can repeat the project name; keep first mention.
    out = []
    for name in leading + parts:
        if name.lower() not in {n.lower() for n in out}:
            out.append(name)
    return out


def extract_primary_artist(credit: str) -> str:
    """The lead artist: the first name in the credit."""
    parts = split_credit(credit)
    return parts[0] if parts else ""


def artist_similarity(candidate_artist: str, billboard_credit: str) -> int:
    """How well an external artist name matches a Billboard credit, 0-100.

    Scores against the whole credit AND each individual artist in it, keeping
    the best: a lyrics or metadata provider may credit only the lead artist, or
    all of them, and either is a good match.

    Raw strings are compared alongside normalized ones because normalization
    strips punctuation, which erases names made entirely of symbols -- "¥$"
    normalizes to an empty string, and comparing empty strings would score 0
    for what is actually an exact match.
    """
    cand_raw = cell_text(candidate_artist).lower()
    if not cand_raw:
        return 0
    cand_norm = normalize_text(candidate_artist)

    credit = cell_text(billboard_credit)
    options = [credit] + split_credit(credit)

    best = 0
    for option in options:
        raw = option.strip().lower()
        if raw:
            best = max(best, int(fuzz.token_sort_ratio(raw, cand_raw)))
        norm = normalize_text(option)
        if norm and cand_norm:
            best = max(best, int(fuzz.token_sort_ratio(norm, cand_norm)))
    return best


def has_feature(credit: str) -> bool:
    """True when the credit carries a 'Featuring'/'feat.' billing."""
    return bool(FEATURE_PATTERN.search(credit)) if isinstance(credit, str) else False
