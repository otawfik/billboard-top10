"""Shared logic for collapsing one song that Billboard listed under two credits.

Billboard sometimes changes a chart entry's credit or title mid-run: "Kanye
West" became "Ye", "Charli XCX" became "Charli xcx", "Taylor Swift" became
"Taylor Swift Featuring Ice Spice". Each version becomes its own row, which
splits the song's weeks in two and understates its peak.

find_merge_groups() identifies those pairs. Both scripts 01 and 04 use it so the
song table and the artist-history index agree on what counts as one song.

A pair is merged only when ALL of these hold:
  1. same normalized title
  2. their credits share an artist, or are a known rename
  3. their chart runs overlap or sit within MERGE_GAP_DAYS
  4. they never appear on the same chart week -- one song cannot hold two
     positions at once, so a shared week proves they are different songs
  5. if their raw titles differ, neither carries a version marker: "(Interlude)",
     "(Taylor's Version)", "(1947)" mark a genuinely different recording, while
     "(Do It)" is just a longer form of the same title
"""

import re
from itertools import combinations

from artist_utils import normalize_text, split_credit

MERGE_GAP_DAYS = 35

# A real single runs somewhere in this range. Outside it, a duration belongs to
# an album upload or is a placeholder. Shared so script 03 backfills the
# MusicBrainz length for exactly the rows script 06 would need it for.
MIN_PLAUSIBLE_DURATION = 30
MAX_PLAUSIBLE_DURATION = 600

# Renames where the old and new credits share no artist name at all.
KNOWN_RENAMES = [
    {"kanye west", "ye"},
    {"puff daddy", "p diddy", "diddy"},
    {"tekashi 6ix9ine", "6ix9ine"},
    {"machine gun kelly", "mgk"},
]

# A parenthetical that marks a different recording rather than a longer title.
VERSION_MARKER = re.compile(
    r"\(\s*(?:\d{4}"
    r"|[^)]*\b(?:version|remix|interlude|poem|live|acoustic|remaster(?:ed)?"
    r"|edit|mix|demo|instrumental|reprise|radio|extended|sped\s*up|slowed)\b[^)]*)"
    r"\s*\)",
    re.IGNORECASE,
)

# Pairs that pass every rule but are, on inspection, different songs.
# Stored as (normalized title, frozenset of the two raw credits).
EXCLUDED_MERGES = {
    # SZA's "Special" (SOS) vs Lizzo's "Special" featuring SZA.
    ("special", frozenset({"SZA", "Lizzo Featuring SZA"})),
}


def _artist_parts(credit: str) -> set[str]:
    return {normalize_text(p) for p in split_credit(credit)} - {""}


def _linked(parts_a: set[str], parts_b: set[str]) -> bool:
    if parts_a & parts_b:
        return True
    return any(parts_a & group and parts_b & group for group in KNOWN_RENAMES)


def _mergeable(a, b, chart_dates) -> bool:
    """a and b are namedtuples with title, artist, debut_date, last_date."""
    if ("special", frozenset({a.artist, b.artist})) in EXCLUDED_MERGES:
        return False
    key = (normalize_text(a.title), frozenset({a.artist, b.artist}))
    if key in EXCLUDED_MERGES:
        return False

    if not _linked(_artist_parts(a.artist), _artist_parts(b.artist)):
        return False

    latest_start = max(a.debut_date, b.debut_date)
    earliest_end = min(a.last_date, b.last_date)
    if latest_start > earliest_end and (latest_start - earliest_end).days > MERGE_GAP_DAYS:
        return False

    # One song cannot occupy two chart positions in the same week.
    if chart_dates.get((a.title, a.artist), set()) & chart_dates.get((b.title, b.artist), set()):
        return False

    # Different titles are fine only when neither is a marked version.
    if a.title != b.title and (VERSION_MARKER.search(a.title) or VERSION_MARKER.search(b.title)):
        return False

    return True


def find_merge_groups(songs, chart_dates) -> list[list[int]]:
    """Group row positions of `songs` that are really the same song.

    `songs` is a DataFrame with title, artist, debut_date, last_date columns.
    `chart_dates` maps (title, artist) to the set of dates it charted.
    Returns only groups of two or more, as lists of positional indices.
    """
    songs = songs.reset_index(drop=True)
    parent = list(range(len(songs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_title = {}
    for row in songs.itertuples():
        by_title.setdefault(normalize_text(row.title), []).append(row)

    for group in by_title.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            if _mergeable(a, b, chart_dates):
                union(a.Index, b.Index)

    clusters = {}
    for i in range(len(songs)):
        clusters.setdefault(find(i), []).append(i)
    return [sorted(members) for members in clusters.values() if len(members) > 1]
