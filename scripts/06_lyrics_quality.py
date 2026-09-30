"""
Step 6: Flag lyrics that look partial, and try to replace them with a fuller version.

A truncated LRCLIB upload still matches on title and artist, so it cannot be
caught by match_score. What gives it away is how little text it holds for how
long the song runs: characters per second of duration.

Two problems have to be kept apart, because LRCLIB duration is not always the
song's:
  * duration between 30s and 600s -> trustworthy, so characters per second is
    meaningful and a song far below its genre's normal rate is suspect.
  * duration outside that range -> the entry is an album upload or a placeholder
    (0-2 seconds). Characters per second is meaningless there, so those rows are
    judged on lyrics length alone and marked duration_reliable = False.

Flagged rows are re-searched for a fuller version of the same song. Anything
that stays flagged keeps its lyrics and carries lyrics_suspect_short = True, so
it can be excluded or discussed later rather than silently trusted.

Run:
    python scripts/06_lyrics_quality.py --check-only   # report, no API calls
    python scripts/06_lyrics_quality.py                # re-search flagged rows
"""

import argparse
import importlib.util
import time
from pathlib import Path

import pandas as pd

from artist_utils import MATCH_SCORE_THRESHOLD, artist_similarity, cell_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LYRICS_PATH = PROJECT_ROOT / "data" / "raw" / "lyrics.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "clean" / "lyrics_quality.csv"

# A real single runs somewhere in this range. Outside it, the duration belongs
# to an album upload or is a placeholder, not to the song.
MIN_PLAUSIBLE_DURATION = 30
MAX_PLAUSIBLE_DURATION = 600

# Flag the slowest this share of songs within each genre.
SUSPECT_PERCENTILE = 0.02
# A genre needs at least this many songs before its own cutoff is trustworthy.
MIN_GENRE_SIZE = 30
# Replace the stored lyrics only if the new version is meaningfully longer.
IMPROVEMENT_RATIO = 1.2


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).parent / path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_best_lyrics(songs: pd.DataFrame, lyrics: pd.DataFrame,
                     debut_rejected: set | None = None) -> pd.DataFrame:
    """One lyrics row per song, preferring the credit the song debuted under.

    A merged song was fetched once per credit, and a later credit often matches
    "better" precisely because it is a fuller, later version: LRCLIB's entry for
    "Taylor Swift Featuring Ice Spice" contains a verse that did not exist when
    "Karma" entered the chart. Using it would feed post-debut text into features
    meant to describe the song at debut, the same leakage rule that governs
    artist history.

    So the debut credit wins whenever its match is good: at or above the match
    threshold, with real lyrics, and not flagged as suspiciously short.
    `debut_rejected` carries the songs whose debut lyrics were flagged on a
    previous pass, which is the one case where a later credit is preferred.
    """
    lut = {(r["title"], r["artist"]): r for r in lyrics.to_dict("records")}
    debut_rejected = debut_rejected or set()

    def score_of(row):
        value = row.get("match_score")
        return float(value) if pd.notna(value) else 0.0

    def usable(row):
        return (
            score_of(row) >= MATCH_SCORE_THRESHOLD
            and bool(cell_text(row.get("plain_lyrics")))
            and not bool(row.get("instrumental"))
        )

    chosen = []
    for song in songs.itertuples():
        debut_key = (song.title, song.artist)
        keys = [debut_key]
        for part in cell_text(getattr(song, "merged_from", "")).split("|"):
            if " :: " in part:
                title, artist = part.split(" :: ", 1)
                keys.append((title.strip(), artist.strip()))

        candidates = [lut[k] for k in dict.fromkeys(keys) if k in lut]
        if not candidates:
            chosen.append({"title": song.title, "artist": song.artist})
            continue

        debut = lut.get(debut_key)
        if debut is not None and usable(debut) and debut_key not in debut_rejected:
            best = debut
        else:
            best = max(candidates, key=lambda r: (score_of(r), len(cell_text(r.get("plain_lyrics")))))

        row = dict(best)
        # Re-key to the canonical credit so it joins to the song table.
        row["title"], row["artist"] = song.title, song.artist
        row["lyrics_from_credit"] = best["artist"]
        chosen.append(row)

    return pd.DataFrame(chosen)


def build_table(debut_rejected: set | None = None) -> pd.DataFrame:
    """Lyrics rows with length, rate, and the genre each is judged against."""
    genres_mod = _load("genres_mod", "05_genres.py")
    songs, mb_songs, mb_artists = genres_mod.load_inputs()
    genres = genres_mod.build_genres(songs, mb_songs, mb_artists)[
        ["title", "artist", "primary_genre"]
    ]

    # Driven by the song table, not by lyrics.csv: script 01 merges songs that
    # Billboard listed under two credits, and the dropped credit still has a
    # row in the lyrics fetch cache. Starting from the song table keeps this
    # report to songs that actually exist.
    lyrics = pd.read_csv(LYRICS_PATH)
    df = genres.merge(
        pick_best_lyrics(songs, lyrics, debut_rejected), on=["title", "artist"], how="left"
    )

    df["lyrics_chars"] = [len(cell_text(v)) for v in df["plain_lyrics"]]
    df["instrumental"] = df["instrumental"].fillna(False).astype(bool)
    # A song with no lyrics row at all is unmatched, same as an explicit "none".
    df["matched"] = df["match_method"].apply(lambda v: cell_text(v) not in ("", "none"))

    # LRCLIB duration is sometimes an album's or a 0-2s placeholder. Fall back
    # to the MusicBrainz recording length where we have it.
    mb_length = mb_songs.set_index(["title", "artist"])["recording_length_ms"].to_dict() \
        if "recording_length_ms" in mb_songs.columns else {}
    # pd.notna, not truthiness: a NaN length is truthy and would look like a
    # real fallback value.
    raw_ms = [mb_length.get((t, a)) for t, a in zip(df["title"], df["artist"])]
    df["mb_duration"] = [ms / 1000 if pd.notna(ms) and ms else None for ms in raw_ms]

    lrclib_ok = df["duration"].between(MIN_PLAUSIBLE_DURATION, MAX_PLAUSIBLE_DURATION).fillna(False)
    df["duration_used"] = df["duration"].where(lrclib_ok, df["mb_duration"])
    df["duration_source"] = [
        "lrclib" if ok else ("musicbrainz" if pd.notna(mb) else "none")
        for ok, mb in zip(lrclib_ok, df["mb_duration"])
    ]

    df["duration_reliable"] = (
        df["duration_used"].between(MIN_PLAUSIBLE_DURATION, MAX_PLAUSIBLE_DURATION).fillna(False)
    )
    df["chars_per_sec"] = (df["lyrics_chars"] / df["duration_used"]).where(df["duration_reliable"])
    return df


def flag_suspects(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose lyrics look too short to be the whole song."""
    judged = df["matched"] & ~df["instrumental"] & (df["lyrics_chars"] > 0)

    # Cutoff per genre, computed only from rows with a trustworthy duration.
    rated = df[judged & df["duration_reliable"]]
    global_cut = rated["chars_per_sec"].quantile(SUSPECT_PERCENTILE)
    cutoffs = {}
    for genre, group in rated.dropna(subset=["primary_genre"]).groupby("primary_genre"):
        if len(group) >= MIN_GENRE_SIZE:
            cutoffs[genre] = group["chars_per_sec"].quantile(SUSPECT_PERCENTILE)

    df["cutoff_used"] = [
        cutoffs.get(g, global_cut) if judged.iloc[i] else None
        for i, g in enumerate(df["primary_genre"])
    ]

    # Rows whose duration cannot be trusted are judged on length alone.
    length_cut = rated["lyrics_chars"].quantile(SUSPECT_PERCENTILE)
    df["length_cutoff"] = length_cut

    suspect = []
    for i, row in enumerate(df.itertuples()):
        if not judged.iloc[i]:
            suspect.append(False)
        elif row.duration_reliable:
            suspect.append(bool(row.chars_per_sec < df["cutoff_used"].iloc[i]))
        else:
            suspect.append(bool(row.lyrics_chars < length_cut))
    df["lyrics_suspect_short"] = suspect
    return df


def find_fuller_version(lyrics_mod, title: str, artist: str, current_chars: int):
    """Search LRCLIB for a version of this song carrying more lyrics."""
    best = None
    for q_artist in (artist, None):
        results = lyrics_mod.lrclib_search(title, q_artist)
        time.sleep(lyrics_mod.REQUEST_DELAY_SECONDS)  # same courtesy delay as script 02
        for candidate in results or []:
            text = str(candidate.get("plainLyrics") or "").strip()
            if candidate.get("instrumental") or not text:
                continue
            if artist_similarity(candidate.get("artistName"), artist) < lyrics_mod.TITLE_ONLY_ARTIST_FLOOR:
                continue
            if best is None or len(text) > len(best[1]):
                best = (candidate, text)
        if best and len(best[1]) >= current_chars * IMPROVEMENT_RATIO:
            break
    if best and len(best[1]) >= current_chars * IMPROVEMENT_RATIO:
        return best
    return None


def refetch_suspects(df: pd.DataFrame) -> int:
    """Try to replace each flagged row's lyrics with a fuller version."""
    lyrics_mod = _load("lyrics_mod", "02_lyrics.py")
    stored = pd.read_csv(LYRICS_PATH)
    rows = stored.to_dict("records")
    index = {(r["title"], r["artist"]): i for i, r in enumerate(rows)}

    targets = df[df["lyrics_suspect_short"]]
    print(f"\nRe-searching {len(targets)} flagged rows for a fuller version...\n")

    replaced = 0
    for n, row in enumerate(targets.itertuples(), start=1):
        found = find_fuller_version(lyrics_mod, row.title, row.artist, row.lyrics_chars)
        label = f"[{n}/{len(targets)}] {row.title} - {row.artist}"
        if found:
            candidate, text = found
            i = index[(row.title, row.artist)]
            rows[i].update({
                "lrclib_id": candidate.get("id"),
                "lrclib_title": candidate.get("trackName"),
                "lrclib_artist": candidate.get("artistName"),
                "plain_lyrics": text,
                "instrumental": candidate.get("instrumental"),
                "duration": candidate.get("duration"),
            })
            replaced += 1
            print(f"{label}: {row.lyrics_chars} -> {len(text)} chars  FULLER VERSION FOUND")
        else:
            print(f"{label}: {row.lyrics_chars} chars, nothing fuller found")

    if replaced:
        lyrics_mod.rewrite_all(rows)
    print(f"\n{replaced} of {len(targets)} rows replaced with a fuller version.")
    return replaced


def print_report(df: pd.DataFrame):
    judged = df["matched"] & ~df["instrumental"] & (df["lyrics_chars"] > 0)
    print(f"\n{'=' * 78}\nLYRICS LENGTH CHECK\n{'=' * 78}")
    print(f"rows: {len(df)} | judged: {int(judged.sum())} | "
          f"instrumental: {int(df['instrumental'].sum())} | unmatched: {int((~df['matched']).sum())}")
    print(f"duration unusable (outside {MIN_PLAUSIBLE_DURATION}-{MAX_PLAUSIBLE_DURATION}s): "
          f"{int((judged & ~df['duration_reliable']).sum())}")
    print("duration source: " + ", ".join(
        f"{k}={v}" for k, v in df.loc[judged, "duration_source"].value_counts().items()))

    print("\nchars-per-second cutoffs (bottom "
          f"{SUSPECT_PERCENTILE:.0%} within each genre):")
    rated = df[judged & df["duration_reliable"]]
    for genre, group in rated.dropna(subset=["primary_genre"]).groupby("primary_genre"):
        mark = "" if len(group) >= MIN_GENRE_SIZE else "  (too few, uses global cutoff)"
        print(f"  {genre:20} n={len(group):5}  median={group['chars_per_sec'].median():6.2f}  "
              f"cutoff={group['chars_per_sec'].quantile(SUSPECT_PERCENTILE):6.2f}{mark}")

    flagged = df[df["lyrics_suspect_short"]]
    print(f"\nlyrics_suspect_short: {len(flagged)} rows ({len(flagged) / max(int(judged.sum()), 1) * 100:.1f}% of judged)")
    if len(flagged):
        cols = ["title", "artist", "primary_genre", "duration", "lyrics_chars",
                "chars_per_sec", "duration_reliable"]
        print(flagged.sort_values("lyrics_chars")[cols].round(2)
              .to_string(index=False, max_colwidth=26))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Report without calling LRCLIB or writing lyrics back.")
    args = parser.parse_args()

    df = flag_suspects(build_table())

    # Second pass: a song whose DEBUT lyrics came back suspiciously short is the
    # one case where a later credit is allowed to win, so re-pick just those.
    rejected = {
        (row.title, row.artist)
        for row in df.itertuples()
        if row.lyrics_suspect_short and row.lyrics_from_credit == row.artist
    }
    if rejected:
        retried = flag_suspects(build_table(rejected))
        switched = int((retried["lyrics_from_credit"] != retried["artist"]).sum()
                       - (df["lyrics_from_credit"] != df["artist"]).sum())
        print(f"\n{len(rejected)} songs had short debut lyrics; "
              f"{max(switched, 0)} switched to a later credit.")
        df = retried

    print_report(df)

    if not args.check_only:
        if refetch_suspects(df):
            df = flag_suspects(build_table())  # recompute after replacements
            print("\nAfter re-search:")
            print_report(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df[["title", "artist", "lrclib_id", "duration", "mb_duration",
              "duration_used", "duration_source", "duration_reliable",
              "lyrics_chars", "chars_per_sec", "primary_genre", "instrumental",
              "lyrics_suspect_short"]]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
