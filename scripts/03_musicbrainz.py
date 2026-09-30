"""
Step 3: Pull song and artist metadata from MusicBrainz.

Part A (data/raw/mb_songs.csv): for every song in billboard_songs.csv, search
MusicBrainz recordings, score the candidates with rapidfuzz, and for the best
match record the recording MBID, earliest release date, release type, genres,
tags, and the MBIDs of every credited artist.

Part B (data/raw/mb_artists.csv): for every unique artist MBID collected in
Part A, record name, type, country, gender, begin date, genres and tags. Each
artist is looked up only once.

MusicBrainz allows at most 1 request per second and requires a descriptive
User-Agent, so every request goes through a global throttle. Progress is saved
every 50 rows and reruns skip work that is already on disk.

Run:
    python scripts/03_musicbrainz.py --limit 50    # test on the first 50 songs
    python scripts/03_musicbrainz.py               # full list
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from rapidfuzz import fuzz

from song_utils import MAX_PLAUSIBLE_DURATION, MIN_PLAUSIBLE_DURATION
from artist_utils import (
    MATCH_RULES_VERSION,
    MATCH_SCORE_THRESHOLD,
    SEPARATOR_VARIANTS,
    artist_similarity,
    cell_text,
    extract_primary_artist,
    normalize_text,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "clean" / "billboard_songs.csv"
SONGS_OUT = PROJECT_ROOT / "data" / "raw" / "mb_songs.csv"
LYRICS_PATH = PROJECT_ROOT / "data" / "raw" / "lyrics.csv"
ARTISTS_OUT = PROJECT_ROOT / "data" / "raw" / "mb_artists.csv"

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "BillboardHitsProject/1.0 (mariofarouk1@gmail.com)"

# MusicBrainz rate limit: 1 request per second. Use slightly more than a
# second so clock jitter can never push us over the limit.
MIN_REQUEST_INTERVAL = 1.05
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
BATCH_SIZE = 50
SEARCH_LIMIT = 10

# MusicBrainz is full of karaoke/tribute re-recordings that share a title with
# the real song. They score well on title alone, so push them down.
NON_ORIGINAL_PATTERN = re.compile(
    r"karaoke|tribute|instrumental|made popular by|originally performed", re.IGNORECASE
)
NON_ORIGINAL_PENALTY = 25

# The title-only search drops the artist from the query, so it can return a
# different act's recording of the same name. Same floor as scripts/02.
TITLE_ONLY_ARTIST_FLOOR = 70

SONG_COLUMNS = [
    "title",
    "artist",
    "recording_mbid",
    "mb_title",
    "mb_artist",
    "first_release_date",
    "release_type",
    "recording_length_ms",
    "genres",
    "genre_counts",
    "tags",
    "tag_counts",
    "artist_mbids",
    "genre_source",
    "match_score",
    "match_method",
    "match_rules_version",
]

# --retry re-fetches rows that are unmatched, scored below this, or were
# written before the current credit-splitting rules.
RETRY_SCORE_THRESHOLD = 90
RETRY_CREDIT_MARKERS = re.compile(r"&|,|\bfeat\b\.?|\bduet\s+with\b", re.IGNORECASE)

ARTIST_COLUMNS = [
    "artist_mbid",
    "artist_name",
    "artist_type",
    "country",
    "gender",
    "begin_date",
    "genres",
    "genre_counts",
    "tags",
    "tag_counts",
]

# ---------------------------------------------------------------------------
# Throttled HTTP with retry/backoff
# ---------------------------------------------------------------------------
_last_request_time = 0.0


def _throttle():
    """Block until at least MIN_REQUEST_INTERVAL has passed since the last call."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def mb_get(path: str, params: dict) -> dict | None:
    """GET a MusicBrainz endpoint, retrying with backoff on 503s and timeouts."""
    url = f"{MB_BASE}/{path}?{urlencode({**params, 'fmt': 'json'})}"
    req = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 404:
                return None  # genuinely missing, retrying will not help
            wait = BACKOFF_BASE_SECONDS * (2**attempt)
            print(f"  [retry {attempt + 1}] HTTP {e.code} on {path} -- waiting {wait}s")
            time.sleep(wait)
        # OSError covers URLError, TimeoutError and the bare socket errors that
        # urllib does not wrap -- a ConnectionResetError from the server killed
        # an earlier run outright instead of being retried.
        except (OSError, json.JSONDecodeError) as e:
            wait = BACKOFF_BASE_SECONDS * (2**attempt)
            print(f"  [retry {attempt + 1}] {e} on {path} -- waiting {wait}s")
            time.sleep(wait)

    print(f"  [give up] {path} after {MAX_RETRIES} attempts")
    return None


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------
def credit_string(artist_credit: list | None) -> str:
    """Rebuild the full credit line, e.g. 'Bruno Mars & Cardi B'."""
    parts = []
    for ac in artist_credit or []:
        parts.append(ac.get("name", ""))
        parts.append(ac.get("joinphrase", ""))
    return "".join(parts)


def score_candidate(candidate: dict, norm_title: str, norm_artist: str, norm_primary: str) -> int:
    cand_title_raw = candidate.get("title", "")
    cand_artist_raw = credit_string(candidate.get("artist-credit"))

    title_score = fuzz.token_sort_ratio(norm_title, normalize_text(cand_title_raw))
    artist_score = max(
        fuzz.token_sort_ratio(norm_artist, normalize_text(cand_artist_raw)),
        fuzz.token_sort_ratio(norm_primary, normalize_text(cand_artist_raw)),
    )
    score = 0.6 * title_score + 0.4 * artist_score

    haystack = f"{cand_title_raw} {candidate.get('disambiguation', '')}"
    if NON_ORIGINAL_PATTERN.search(haystack):
        score -= NON_ORIGINAL_PENALTY

    return round(max(score, 0))


def pick_best_candidate(candidates: list[dict], norm_title: str, norm_artist: str, norm_primary: str):
    """Highest score wins.

    Ties break first on how closely the candidate matches the FULL Billboard
    credit, then on the earliest release. The full-credit step matters because
    the charting version of a collaboration is often a remix: "Finesse" by
    "Bruno Mars & Cardi B" must pick the Cardi B remix, not the earlier solo
    album track, even though both score the same. The date step then prefers an
    original over a later reissue of the same recording.
    """
    def sort_key(c):
        score = score_candidate(c, norm_title, norm_artist, norm_primary)
        full_match = fuzz.token_sort_ratio(
            norm_artist, normalize_text(credit_string(c.get("artist-credit")))
        )
        date = c.get("first-release-date") or "9999"
        return (-score, -full_match, date)

    best = min(candidates, key=sort_key)
    return best, score_candidate(best, norm_title, norm_artist, norm_primary)


# ---------------------------------------------------------------------------
# Part A: songs
# ---------------------------------------------------------------------------
def pick_earliest_release(recording: dict) -> tuple[str | None, str | None, str | None]:
    """Return (release_type, release_group_mbid, date) for the earliest release."""
    releases = recording.get("releases") or []
    if not releases:
        return None, None, None

    first_date = recording.get("first-release-date")
    chosen = None
    if first_date:
        chosen = next((r for r in releases if r.get("date") == first_date), None)
    if chosen is None:
        dated = [r for r in releases if r.get("date")]
        chosen = min(dated, key=lambda r: r["date"]) if dated else releases[0]

    rg = chosen.get("release-group") or {}
    return rg.get("primary-type"), rg.get("id"), chosen.get("date")


def names_and_counts(items: list[dict] | None) -> tuple[str, str]:
    """Names and their vote counts, most-voted first: ('hip hop|pop rap', '19|3').

    MusicBrainz returns genres in ALPHABETICAL order, not by popularity, so the
    sort here is what puts the most-voted genre first. The counts are stored
    alongside because the order alone cannot express ties: a recording whose
    genres are all voted once is stored in alphabetical order, which would
    otherwise look like a ranking.
    """
    if not items:
        return "", ""
    ordered = sorted(items, key=lambda g: -(g.get("count") or 0))
    return (
        "|".join(g["name"] for g in ordered),
        "|".join(str(g.get("count") or 0) for g in ordered),
    )


def search_recording(title: str, artist: str) -> tuple[dict | None, int, str]:
    """Run the query cascade. Returns (best candidate, score, match_method)."""
    norm_title = normalize_text(title)
    norm_artist = normalize_text(artist)
    norm_primary = normalize_text(extract_primary_artist(artist))

    if not norm_title:
        return None, 0, "none"

    # Unquoted terms: MusicBrainz phrase-matches quoted strings against its own
    # tokenization, so quoting normalized text returns nothing.
    stages = [
        ("title_primary_artist", f"recording:({norm_title}) AND artist:({norm_primary})"),
        ("title_full_artist", f"recording:({norm_title}) AND artist:({norm_artist})"),
        ("title_only", f"recording:({norm_title})"),
    ]

    for method, query in stages:
        if method == "title_primary_artist" and not norm_primary:
            continue
        if method == "title_full_artist" and norm_artist == norm_primary:
            continue  # identical to the query we already ran

        result = mb_get("recording", {"query": query, "limit": SEARCH_LIMIT})
        candidates = (result or {}).get("recordings") or []

        if method == "title_only":
            candidates = [
                c for c in candidates
                if artist_similarity(credit_string(c.get("artist-credit")), artist)
                >= TITLE_ONLY_ARTIST_FLOOR
            ]

        if not candidates:
            continue

        best, score = pick_best_candidate(candidates, norm_title, norm_artist, norm_primary)
        return best, score, method

    return None, 0, "none"


def fetch_song_details(mbid: str) -> dict:
    """Recording lookup, falling back to release-group genres when the recording has none."""
    recording = mb_get(
        f"recording/{mbid}",
        {"inc": "genres+tags+artist-credits+releases+release-groups"},
    )
    if recording is None:
        return {}

    genres, genre_counts = names_and_counts(recording.get("genres"))
    tags, tag_counts = names_and_counts(recording.get("tags"))
    release_type, release_group_mbid, release_date = pick_earliest_release(recording)
    genre_source = "recording" if genres else "none"

    # Remix and other secondary recordings often carry no first-release-date of
    # their own, so fall back to the date of the release they appear on.
    first_release_date = recording.get("first-release-date") or release_date

    if not genres and release_group_mbid:
        rg = mb_get(f"release-group/{release_group_mbid}", {"inc": "genres+tags"})
        if rg:
            genres, genre_counts = names_and_counts(rg.get("genres"))
            if genres:
                genre_source = "release-group"
            if not tags:
                tags, tag_counts = names_and_counts(rg.get("tags"))

    artist_mbids = [
        ac["artist"]["id"]
        for ac in recording.get("artist-credit") or []
        if ac.get("artist", {}).get("id")
    ]

    return {
        "mb_title": recording.get("title"),
        "mb_artist": credit_string(recording.get("artist-credit")),
        "first_release_date": first_release_date,
        "release_type": release_type,
        # LRCLIB durations are sometimes an album's or a placeholder, so script
        # 06 falls back to this.
        "recording_length_ms": recording.get("length"),
        "genres": genres,
        "genre_counts": genre_counts,
        "tags": tags,
        "tag_counts": tag_counts,
        "artist_mbids": "|".join(artist_mbids),
        "genre_source": genre_source,
    }


def build_song_row(title: str, artist: str) -> dict:
    blank = {col: None for col in SONG_COLUMNS}
    blank.update({"title": title, "artist": artist, "match_score": 0,
                  "match_method": "none", "match_rules_version": MATCH_RULES_VERSION})

    best, score, method = search_recording(title, artist)
    if best is None:
        return blank

    row = {
        **blank,
        "recording_mbid": best.get("id"),
        "mb_title": best.get("title"),
        "mb_artist": credit_string(best.get("artist-credit")),
        "first_release_date": best.get("first-release-date"),
        "match_score": score,
        "match_method": method,
    }

    # Only spend lookup requests on matches we actually trust. A wrong match's
    # genres would pollute the dataset, and the search fields above are still
    # written out so low-score rows can be reviewed by eye.
    if score >= MATCH_SCORE_THRESHOLD:
        row.update(fetch_song_details(best["id"]))
    else:
        row["genre_source"] = "not_looked_up"

    return row


# ---------------------------------------------------------------------------
# Part B: artists
# ---------------------------------------------------------------------------
def build_artist_row(artist_mbid: str) -> dict:
    artist = mb_get(f"artist/{artist_mbid}", {"inc": "genres+tags"})
    if artist is None:
        return {**{col: None for col in ARTIST_COLUMNS}, "artist_mbid": artist_mbid}

    genres, genre_counts = names_and_counts(artist.get("genres"))
    tags, tag_counts = names_and_counts(artist.get("tags"))
    return {
        "artist_mbid": artist_mbid,
        "artist_name": artist.get("name"),
        "artist_type": artist.get("type"),
        "country": artist.get("country"),
        "gender": artist.get("gender"),
        "begin_date": (artist.get("life-span") or {}).get("begin"),
        "genres": genres,
        "genre_counts": genre_counts,
        "tags": tags,
        "tag_counts": tag_counts,
    }


# ---------------------------------------------------------------------------
# Resuming / batch writing
# ---------------------------------------------------------------------------
def append_batch(path: Path, columns: list[str], rows: list[dict]):
    file_exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Retry mode
# ---------------------------------------------------------------------------
def needs_retry(row: dict) -> str | None:
    """Why this row should be re-fetched, or None to leave it alone."""
    if row.get("match_method") == "none":
        return "unmatched"

    score = row.get("match_score")
    if pd.isna(score) or score < RETRY_SCORE_THRESHOLD:
        return f"score<{RETRY_SCORE_THRESHOLD}"

    # Title-only hits stored before the artist floor existed may be a different
    # act's recording of the same name.
    if cell_text(row.get("match_method")) == "title_only":
        if artist_similarity(row.get("mb_artist"), row.get("artist")) < TITLE_ONLY_ARTIST_FLOOR:
            return "weak title-only artist"

    # Credits using a Billboard typo or abbreviation ("Feauring", "Ft.") were
    # split wrong before those spellings were added to the splitter.
    if SEPARATOR_VARIANTS.search(cell_text(row.get("artist"))):
        return "separator variant"

    version = row.get("match_rules_version")
    stale = pd.isna(version) or int(version) < MATCH_RULES_VERSION
    if stale and RETRY_CREDIT_MARKERS.search(cell_text(row.get("artist"))):
        return "old splitter"
    return None


def rewrite_all(path: Path, columns: list[str], rows: list[dict]):
    """Replace the whole file. Retry updates rows in place, so appending would
    duplicate them. Writes to a temp file first so an interrupt cannot truncate
    the real one."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def songs_needing_length() -> set[tuple[str, str]]:
    """Songs whose LRCLIB duration is unusable, so script 06 needs our length.

    Every other row already has a trustworthy duration, so fetching a length
    for it would spend a request on data nothing reads.
    """
    if not LYRICS_PATH.exists():
        return set()
    lyrics = pd.read_csv(LYRICS_PATH)
    usable = lyrics["duration"].between(
        MIN_PLAUSIBLE_DURATION, MAX_PLAUSIBLE_DURATION
    ).fillna(False)
    unusable = lyrics[~usable]
    return set(zip(unusable["title"], unusable["artist"]))


def missing_counts(row: dict, length_needed: set | None = None) -> bool:
    """Row that the backfill should re-fetch."""
    if bool(cell_text(row.get("genres"))) and not cell_text(row.get("genre_counts")):
        return True
    if length_needed is None:
        return False
    return (
        (row.get("title"), row.get("artist")) in length_needed
        and bool(cell_text(row.get("recording_mbid")))
        and not cell_text(row.get("recording_length_ms"))
    )


def run_backfill_counts():
    """Add vote counts to rows fetched before counts were stored.

    Only rows missing counts are re-fetched, so this costs far less than
    re-running everything. Songs whose genres came from a release-group need a
    second request, because script 03 never stored the release-group MBID and
    fetch_song_details has to rediscover it from the recording.
    """
    for path, columns, kind in (
        (SONGS_OUT, SONG_COLUMNS, "songs"),
        (ARTISTS_OUT, ARTIST_COLUMNS, "artists"),
    ):
        if not path.exists():
            print(f"Backfill {kind}: no {path.name} yet, skipping.")
            continue

        existing = pd.read_csv(path)
        for column in ("genre_counts", "tag_counts"):
            if column not in existing.columns:
                existing[column] = ""
        rows = existing.to_dict("records")

        length_needed = songs_needing_length() if kind == "songs" else set()
        targets = [i for i, row in enumerate(rows) if missing_counts(row, length_needed)]
        print(f"\nBackfill {kind}: {len(rows)} rows on disk, {len(targets)} missing counts.")
        if not targets:
            continue

        filled = failed = 0
        for n, i in enumerate(targets, start=1):
            row = rows[i]
            if kind == "songs":
                mbid = cell_text(row.get("recording_mbid"))
                details = fetch_song_details(mbid) if mbid else {}
                if details.get("genre_counts"):
                    rows[i] = {**row, **details}
                    filled += 1
                else:
                    failed += 1
                name = row["title"]
            else:
                fresh = build_artist_row(row["artist_mbid"])
                if fresh.get("genre_counts"):
                    rows[i] = fresh
                    filled += 1
                else:
                    failed += 1
                name = row.get("artist_name")

            print(f"[{n}/{len(targets)}] {name}: {rows[i].get('genre_counts') or 'no counts'}")

            if n % BATCH_SIZE == 0:
                rewrite_all(path, columns, rows)
                print(f"  -> saved ({n}/{len(targets)})")

        rewrite_all(path, columns, rows)
        print(f"Backfill {kind}: {filled} rows given counts, {failed} left without "
              f"(MusicBrainz has no genres for them any more).")


def run_retry():
    if not SONGS_OUT.exists():
        print("Nothing to retry: no mb_songs.csv yet.")
        return

    existing = pd.read_csv(SONGS_OUT)
    if "match_rules_version" not in existing.columns:
        existing["match_rules_version"] = pd.NA
    rows = existing.to_dict("records")

    reasons = {i: needs_retry(r) for i, r in enumerate(rows)}
    targets = [i for i, why in reasons.items() if why]

    print(f"{len(rows)} rows on disk, {len(targets)} to retry:")
    for label in ("unmatched", f"score<{RETRY_SCORE_THRESHOLD}",
                  "weak title-only artist", "separator variant", "old splitter"):
        count = sum(1 for why in reasons.values() if why == label)
        print(f"  {label}: {count}")

    changed = improved = newly_matched = 0
    for n, i in enumerate(targets, start=1):
        old = rows[i]
        new = build_song_row(old["title"], old["artist"])

        old_score = 0 if pd.isna(old.get("match_score")) else old["match_score"]
        if (new["recording_mbid"] != old.get("recording_mbid")
                or new["match_score"] != old_score
                or new["match_method"] != old.get("match_method")):
            changed += 1
            if new["match_score"] > old_score:
                improved += 1
            if old_score < MATCH_SCORE_THRESHOLD <= new["match_score"]:
                newly_matched += 1

        rows[i] = new
        print(f"[{n}/{len(targets)}] {old['title']} - {old['artist']}: "
              f"{old_score} -> {new['match_score']} ({new['match_method']})")

        if n % BATCH_SIZE == 0:
            rewrite_all(SONGS_OUT, SONG_COLUMNS, rows)
            print(f"  -> saved ({n}/{len(targets)})")

    rewrite_all(SONGS_OUT, SONG_COLUMNS, rows)
    print(f"\nRetried {len(targets)} rows: {changed} changed, {improved} improved, "
          f"{newly_matched} newly above the match threshold.")


def run_part_a(limit: int | None):
    songs = pd.read_csv(INPUT_PATH)
    if limit:
        songs = songs.head(limit)

    done = set()
    if SONGS_OUT.exists():
        existing = pd.read_csv(SONGS_OUT)
        done = set(zip(existing["title"], existing["artist"]))

    todo = [(r.title, r.artist) for r in songs.itertuples() if (r.title, r.artist) not in done]
    print(f"Part A: {len(done)} songs already done, {len(todo)} to fetch.")

    batch = []
    for i, (title, artist) in enumerate(todo, start=1):
        row = build_song_row(title, artist)
        print(f"[{i}/{len(todo)}] {title} - {artist} -> {row['match_score']} ({row['match_method']})")
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            append_batch(SONGS_OUT, SONG_COLUMNS, batch)
            print(f"  -> saved batch of {len(batch)} ({i}/{len(todo)})")
            batch = []

    if batch:
        append_batch(SONGS_OUT, SONG_COLUMNS, batch)
        print(f"  -> saved final batch of {len(batch)}")


def run_part_b():
    if not SONGS_OUT.exists():
        print("Part B: no mb_songs.csv yet, skipping.")
        return

    songs = pd.read_csv(SONGS_OUT)
    trusted = songs[songs["match_score"].fillna(0) >= MATCH_SCORE_THRESHOLD]

    wanted = []
    for value in trusted["artist_mbids"].dropna():
        wanted.extend(m for m in cell_text(value).split("|") if m)
    wanted = list(dict.fromkeys(wanted))  # unique, order preserved

    done = set()
    if ARTISTS_OUT.exists():
        done = set(pd.read_csv(ARTISTS_OUT)["artist_mbid"])

    todo = [m for m in wanted if m not in done]
    print(f"\nPart B: {len(done)} artists already done, {len(todo)} to fetch.")

    batch = []
    for i, mbid in enumerate(todo, start=1):
        row = build_artist_row(mbid)
        print(f"[{i}/{len(todo)}] {row.get('artist_name')} ({row.get('artist_type')})")
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            append_batch(ARTISTS_OUT, ARTIST_COLUMNS, batch)
            print(f"  -> saved batch of {len(batch)} ({i}/{len(todo)})")
            batch = []

    if batch:
        append_batch(ARTISTS_OUT, ARTIST_COLUMNS, batch)
        print(f"  -> saved final batch of {len(batch)}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(limit: int | None):
    if not SONGS_OUT.exists():
        print("No mb_songs.csv to report on.")
        return

    songs = pd.read_csv(INPUT_PATH)
    if limit:
        songs = songs.head(limit)
    mb = pd.read_csv(SONGS_OUT)
    merged = songs.merge(mb, on=["title", "artist"], how="inner")
    merged["matched"] = merged["match_score"].fillna(0) >= MATCH_SCORE_THRESHOLD

    print("\n--- MusicBrainz match report ---")
    print(f"Threshold for counting as matched: match_score >= {MATCH_SCORE_THRESHOLD}")
    print(f"Overall match rate: {merged['matched'].mean() * 100:.1f}% "
          f"({merged['matched'].sum()}/{len(merged)})")

    for label, subset in [("Top 10", merged[merged["top10"]]), ("Non-Top 10", merged[~merged["top10"]])]:
        if len(subset):
            print(f"  {label}: {subset['matched'].mean() * 100:.1f}% "
                  f"({subset['matched'].sum()}/{len(subset)})")

    # Genre coverage, song level vs artist level.
    matched = merged[merged["matched"]]
    if len(matched):
        song_level = matched["genres"].fillna("").str.len().gt(0)
        print(f"\nSongs with >=1 genre (song level): {song_level.mean() * 100:.1f}% "
              f"({song_level.sum()}/{len(matched)})")
        print("  genre_source breakdown:")
        print(matched["genre_source"].fillna("none").value_counts().to_string())

        if ARTISTS_OUT.exists():
            artists = pd.read_csv(ARTISTS_OUT)
            with_genres = set(artists.loc[artists["genres"].fillna("").str.len().gt(0), "artist_mbid"])

            def any_artist_genre(mbids):
                if not isinstance(mbids, str):
                    return False
                return any(m in with_genres for m in mbids.split("|") if m)

            artist_level = matched["artist_mbids"].apply(any_artist_genre)
            print(f"Songs with >=1 genre (artist level): {artist_level.mean() * 100:.1f}% "
                  f"({artist_level.sum()}/{len(matched)})")
            either = song_level | artist_level
            print(f"Songs with >=1 genre (song or artist level): {either.mean() * 100:.1f}%")
        else:
            print("Songs with >=1 genre (artist level): no mb_artists.csv yet.")

    low = merged[(merged["match_score"].fillna(0) > 0) & (~merged["matched"])]
    print(f"\n10 random low-score matches (0 < score < {MATCH_SCORE_THRESHOLD}) for manual review:")
    if not len(low):
        print("  (none found)")
    else:
        cols = ["title", "artist", "mb_title", "mb_artist", "match_score", "match_method"]
        print(low.sample(min(10, len(low)), random_state=42)[cols].to_string(index=False))

    print("\nmatch_score distribution:")
    print(merged["match_score"].fillna(0).describe().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N songs.")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all", help="Which part to run.")
    parser.add_argument("--retry", action="store_true",
                        help="Re-fetch unmatched, low-score, or stale-splitter song rows instead of fetching new ones.")
    parser.add_argument("--backfill-counts", action="store_true",
                        help="Add genre/tag vote counts to song and artist rows fetched before counts were stored.")
    args = parser.parse_args()

    if args.backfill_counts:
        run_backfill_counts()
        return

    if args.retry:
        run_retry()
        run_part_b()  # pick up artists referenced by the corrected matches
        print_report(args.limit)
        return

    if args.part in ("a", "all"):
        run_part_a(args.limit)
    if args.part in ("b", "all"):
        run_part_b()

    print_report(args.limit)


if __name__ == "__main__":
    main()
