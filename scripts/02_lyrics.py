"""
Step 2: Pull lyrics for every song in data/clean/billboard_songs.csv from LRCLIB.

For each song we try up to three LRCLIB searches, falling back only when the
previous search returned zero results:
  1. title + full artist string
  2. title + primary artist only (text before "featuring"/"&"/"x"/","/"with")
  3. title alone

Whichever stage returns candidates, we score every candidate against the
Billboard title/artist with rapidfuzz and keep the best one. match_score and
match_method are always recorded (even for low-confidence matches) so results
can be reviewed by eye; a score threshold is applied only when computing
match-rate statistics, not to blank out data.

Progress is appended to data/raw/lyrics.csv every 50 songs. Reruns skip any
(title, artist) pair already present in that file.
"""

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import pandas as pd
from rapidfuzz import fuzz

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
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "lyrics.csv"

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
USER_AGENT = "BillboardHitsProject/1.0 (mariofarouk1@gmail.com)"

REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
BATCH_SIZE = 50

OUTPUT_COLUMNS = [
    "title",
    "artist",
    "lrclib_id",
    "lrclib_title",
    "lrclib_artist",
    "plain_lyrics",
    "instrumental",
    "duration",
    "match_score",
    "match_method",
    "match_rules_version",
]

# --retry re-fetches rows that are unmatched, scored below this, or were
# written before the current credit-splitting rules.
RETRY_SCORE_THRESHOLD = 90
RETRY_CREDIT_MARKERS = re.compile(r"&|,|\bfeat\b\.?|\bduet\s+with\b", re.IGNORECASE)

# LRCLIB holds instrumental uploads of songs that plainly have vocals, and they
# score identically to the real entry on title and artist. An instrumental is
# therefore a last resort: a candidate carrying actual lyrics wins as long as it
# scores within this many points of the best candidate and still clears the
# match threshold.
INSTRUMENTAL_TOLERANCE = 15

# The title-only search ignores the artist entirely, so it happily returns a
# different act's song of the same name. Require the candidate's artist to
# resemble the Billboard credit before trusting a title-only hit; below this,
# the song is left unmatched rather than given someone else's lyrics.
TITLE_ONLY_ARTIST_FLOOR = 70

# ---------------------------------------------------------------------------
# LRCLIB search with retry/backoff
# ---------------------------------------------------------------------------
def lrclib_search(track_name: str, artist_name: str | None = None) -> list[dict]:
    """Call /api/search, retrying with exponential backoff on errors/timeouts."""
    params = {"track_name": track_name}
    if artist_name:
        params["artist_name"] = artist_name
    url = f"{LRCLIB_SEARCH_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        # OSError covers HTTPError, URLError, TimeoutError and the bare socket
        # errors urllib does not wrap, such as ConnectionResetError.
        except (OSError, json.JSONDecodeError) as e:
            wait = BACKOFF_BASE_SECONDS * (2**attempt)
            print(f"  [retry] {track_name!r} / {artist_name!r}: {e} -- waiting {wait}s")
            time.sleep(wait)
    print(f"  [give up] {track_name!r} / {artist_name!r} after {MAX_RETRIES} attempts")
    return []


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def score_candidate(candidate: dict, norm_title: str, norm_artist: str, norm_primary: str) -> float:
    cand_title = normalize_text(candidate.get("trackName", ""))
    cand_artist = normalize_text(candidate.get("artistName", ""))

    title_score = fuzz.token_sort_ratio(norm_title, cand_title)
    artist_score = max(
        fuzz.token_sort_ratio(norm_artist, cand_artist),
        fuzz.token_sort_ratio(norm_primary, cand_artist),
    )
    return round(0.6 * title_score + 0.4 * artist_score)


def match_song(title: str, artist: str) -> dict:
    norm_title = normalize_text(title)
    norm_artist = normalize_text(artist)
    primary_artist = extract_primary_artist(artist)
    norm_primary = normalize_text(primary_artist)

    stages = [
        ("full", title, artist),
        ("primary_artist", title, primary_artist),
        ("title_only", title, None),
    ]

    # Best instrumental-only result seen so far, used if no stage yields lyrics.
    fallback = None

    for method, q_title, q_artist in stages:
        # Skip the primary_artist stage if it's identical to the full artist
        # (no point re-querying the same thing).
        if method == "primary_artist" and q_artist.strip().lower() == artist.strip().lower():
            continue

        results = lrclib_search(q_title, q_artist)
        time.sleep(REQUEST_DELAY_SECONDS)

        if method == "title_only":
            results = [
                c for c in results
                if artist_similarity(c.get("artistName"), artist) >= TITLE_ONLY_ARTIST_FLOOR
            ]

        if not results:
            continue

        # Ties break on how closely the candidate matches the FULL Billboard
        # credit, so a collaboration picks the version that actually charted
        # (the Cardi B remix of "Finesse", not the earlier solo album track).
        def rank(c):
            return (
                score_candidate(c, norm_title, norm_artist, norm_primary),
                fuzz.token_sort_ratio(norm_artist, normalize_text(c.get("artistName", ""))),
            )

        top_score = max(score_candidate(c, norm_title, norm_artist, norm_primary) for c in results)
        with_lyrics = [
            c for c in results
            if not c.get("instrumental")
            and str(c.get("plainLyrics") or "").strip()
            and score_candidate(c, norm_title, norm_artist, norm_primary)
            >= max(top_score - INSTRUMENTAL_TOLERANCE, MATCH_SCORE_THRESHOLD)
        ]

        best = max(with_lyrics or results, key=rank)
        best_score = score_candidate(best, norm_title, norm_artist, norm_primary)

        # Nothing here has lyrics, so keep the best of them and try the next
        # stage before settling for an instrumental.
        if not with_lyrics:
            if fallback is None:
                fallback = (best, best_score, method)
            continue

        return {
            "title": title,
            "artist": artist,
            "lrclib_id": best.get("id"),
            "lrclib_title": best.get("trackName"),
            "lrclib_artist": best.get("artistName"),
            "plain_lyrics": best.get("plainLyrics"),
            "instrumental": best.get("instrumental"),
            "duration": best.get("duration"),
            "match_score": best_score,
            "match_method": method,
            "match_rules_version": MATCH_RULES_VERSION,
        }

    # No stage produced lyrics; fall back to the best instrumental if there was one.
    if fallback is not None:
        best, best_score, method = fallback
        return {
            "title": title,
            "artist": artist,
            "lrclib_id": best.get("id"),
            "lrclib_title": best.get("trackName"),
            "lrclib_artist": best.get("artistName"),
            "plain_lyrics": best.get("plainLyrics"),
            "instrumental": best.get("instrumental"),
            "duration": best.get("duration"),
            "match_score": best_score,
            "match_method": method,
            "match_rules_version": MATCH_RULES_VERSION,
        }

    # No results from any stage.
    return {
        "title": title,
        "artist": artist,
        "lrclib_id": None,
        "lrclib_title": None,
        "lrclib_artist": None,
        "plain_lyrics": None,
        "instrumental": None,
        "duration": None,
        "match_score": 0,
        "match_method": "none",
        "match_rules_version": MATCH_RULES_VERSION,
    }


# ---------------------------------------------------------------------------
# Resuming / batch writing
# ---------------------------------------------------------------------------
def load_already_done() -> set[tuple[str, str]]:
    if not OUTPUT_PATH.exists():
        return set()
    existing = pd.read_csv(OUTPUT_PATH)
    return set(zip(existing["title"], existing["artist"]))


def append_batch(rows: list[dict]):
    file_exists = OUTPUT_PATH.exists()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    # A match with no lyrics text is worth re-checking: it is usually an
    # instrumental upload standing in for a song that does have vocals.
    if not cell_text(row.get("plain_lyrics")):
        return "no lyrics"

    # Title-only hits stored before the artist floor existed may be a different
    # act's song of the same name.
    if cell_text(row.get("match_method")) == "title_only":
        if artist_similarity(row.get("lrclib_artist"), row.get("artist")) < TITLE_ONLY_ARTIST_FLOOR:
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


def rewrite_all(rows: list[dict]):
    """Replace the whole file. Retry updates rows in place, so appending would
    duplicate them. Writes to a temp file first so an interrupt cannot truncate
    the real one."""
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, OUTPUT_PATH)


def run_retry():
    if not OUTPUT_PATH.exists():
        print("Nothing to retry: no lyrics.csv yet.")
        return

    existing = pd.read_csv(OUTPUT_PATH)
    if "match_rules_version" not in existing.columns:
        existing["match_rules_version"] = pd.NA
    rows = existing.to_dict("records")

    reasons = {i: needs_retry(r) for i, r in enumerate(rows)}
    targets = [i for i, why in reasons.items() if why]

    print(f"{len(rows)} rows on disk, {len(targets)} to retry:")
    for label in ("unmatched", f"score<{RETRY_SCORE_THRESHOLD}", "no lyrics",
                  "weak title-only artist", "separator variant", "old splitter"):
        count = sum(1 for why in reasons.values() if why == label)
        print(f"  {label}: {count}")

    changed = improved = newly_matched = 0
    for n, i in enumerate(targets, start=1):
        old = rows[i]
        new = match_song(old["title"], old["artist"])

        old_score = 0 if pd.isna(old.get("match_score")) else old["match_score"]
        if (new["lrclib_id"] != old.get("lrclib_id")
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
            rewrite_all(rows)
            print(f"  -> saved ({n}/{len(targets)})")

    rewrite_all(rows)
    print(f"\nRetried {len(targets)} rows: {changed} changed, {improved} improved, "
          f"{newly_matched} newly above the match threshold.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(limit: int | None):
    results = pd.read_csv(OUTPUT_PATH)
    songs = pd.read_csv(INPUT_PATH)
    if limit:
        songs = songs.head(limit)
    merged = songs.merge(results, on=["title", "artist"], how="left")

    merged["matched"] = merged["match_score"].fillna(0) >= MATCH_SCORE_THRESHOLD

    print("\n--- Lyrics match report ---")
    print(f"Threshold for counting as matched: match_score >= {MATCH_SCORE_THRESHOLD}")
    overall_rate = merged["matched"].mean() * 100
    print(f"Overall match rate: {overall_rate:.1f}% ({merged['matched'].sum()}/{len(merged)})")

    for label, subset in [("Top 10", merged[merged["top10"]]), ("Non-Top 10", merged[~merged["top10"]])]:
        if len(subset) == 0:
            continue
        rate = subset["matched"].mean() * 100
        print(f"  {label} match rate: {rate:.1f}% ({subset['matched'].sum()}/{len(subset)})")

    low_score = merged[(merged["match_score"].fillna(0) > 0) & (merged["match_score"] < MATCH_SCORE_THRESHOLD)]
    print(f"\n10 random low-score matches (0 < score < {MATCH_SCORE_THRESHOLD}) for manual review:")
    if len(low_score) == 0:
        print("  (none found)")
    else:
        sample = low_score.sample(min(10, len(low_score)), random_state=42)
        cols = ["title", "artist", "lrclib_title", "lrclib_artist", "match_score", "match_method"]
        print(sample[cols].to_string(index=False))

    print("\nmatch_score distribution:")
    print(merged["match_score"].fillna(0).describe().to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N songs (for testing).")
    parser.add_argument("--retry", action="store_true",
                        help="Re-fetch unmatched, low-score, or stale-splitter rows instead of fetching new ones.")
    args = parser.parse_args()

    if args.retry:
        run_retry()
        print_report(args.limit)
        return

    songs = pd.read_csv(INPUT_PATH)
    if args.limit:
        songs = songs.head(args.limit)

    done = load_already_done()
    todo = [
        (row.title, row.artist)
        for row in songs.itertuples()
        if (row.title, row.artist) not in done
    ]
    print(f"{len(done)} songs already done, {len(todo)} remaining to fetch.")

    batch = []
    for i, (title, artist) in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {title} - {artist}")
        row = match_song(title, artist)
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            append_batch(batch)
            print(f"  -> saved batch of {len(batch)} (progress: {i}/{len(todo)})")
            batch = []

    if batch:
        append_batch(batch)
        print(f"  -> saved final batch of {len(batch)}")

    print_report(args.limit)


if __name__ == "__main__":
    main()
