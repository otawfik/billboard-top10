"""
Step 7: Join every per-song table into one analysis file.

Sources, all keyed on the canonical (title, artist) from script 01:
  data/clean/billboard_songs.csv   chart performance and the target, top10
  data/clean/artist_history.csv    what the artist had done before this debut
  data/clean/genres.csv            broad genre
  data/clean/lyrics_quality.csv    lyrics length and the suspect-short flag
  data/raw/lyrics.csv              the lyrics text itself
  data/raw/mb_songs.csv            release date, release type, MusicBrainz genres

Every join is a LEFT join from the song table, so the output has exactly one
row per song and a missing source shows up as an empty column rather than a
dropped song.

Output: data/clean/songs_merged.csv
"""

from pathlib import Path

import pandas as pd

from artist_utils import cell_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = PROJECT_ROOT / "data" / "clean"
RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = CLEAN / "songs_merged.csv"

KEY = ["title", "artist"]


def load(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    # A source may hold several rows for one key if a credit was re-fetched.
    return df.drop_duplicates(subset=KEY, keep="first")


def main():
    songs = load(CLEAN / "billboard_songs.csv")
    print(f"billboard_songs: {len(songs)} rows")

    history = load(CLEAN / "artist_history.csv", KEY + [
        "primary_artist", "all_credited_artists", "n_credited_artists", "has_feature",
        "prior_entries", "prior_top10s", "prior_number1s", "weeks_since_last_entry",
        "is_first_entry", "best_prior_top10s",
    ])
    genres = load(CLEAN / "genres.csv", KEY + [
        "primary_genre", "multi_genre", "n_broad_genres", "genre_source",
        "genre_rule_step", "mb_labels",
        "is_hiphop", "is_pop", "is_rnb", "is_country",
        "is_rock", "is_latin", "is_dance", "is_other",
    ])
    quality = load(CLEAN / "lyrics_quality.csv", KEY + [
        "duration_used", "duration_source", "duration_reliable",
        "lyrics_chars", "chars_per_sec", "lyrics_suspect_short",
    ])
    lyrics = load(RAW / "lyrics.csv", KEY + [
        "plain_lyrics", "instrumental", "match_score", "match_method",
    ]).rename(columns={"match_score": "lyrics_match_score",
                       "match_method": "lyrics_match_method"})
    mb = load(RAW / "mb_songs.csv", KEY + [
        "recording_mbid", "first_release_date", "release_type", "recording_length_ms",
        "genres", "genre_counts", "artist_mbids", "match_score", "match_method",
    ]).rename(columns={"genres": "mb_genres", "genre_counts": "mb_genre_counts",
                       "match_score": "mb_match_score", "match_method": "mb_match_method"})

    merged = songs
    for name, frame in [("artist_history", history), ("genres", genres),
                        ("lyrics_quality", quality), ("lyrics", lyrics),
                        ("musicbrainz", mb)]:
        before = len(merged)
        merged = merged.merge(frame, on=KEY, how="left")
        print(f"  + {name:15} {len(frame):5} rows -> {len(merged)} (was {before})")
        assert len(merged) == before, f"{name} join duplicated rows"

    CLEAN.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(merged)} rows x {len(merged.columns)} columns to {OUTPUT_PATH}")
    report(merged)


def report(df: pd.DataFrame):
    print(f"\n{'=' * 72}\nMISSING VALUES PER COLUMN\n{'=' * 72}")
    rows = []
    for column in df.columns:
        filled = df[column].apply(lambda v: bool(cell_text(v))).sum()
        missing = len(df) - filled
        rows.append((column, missing, missing / len(df) * 100))
    for column, missing, pct in sorted(rows, key=lambda r: -r[1]):
        bar = "#" * int(pct / 4)
        print(f"  {column:26} {missing:5}  {pct:5.1f}%  {bar}")

    print(f"\n{'=' * 72}\nTOP 10 vs NON-TOP 10\n{'=' * 72}")
    top = df["top10"].astype(bool)
    print(f"  Top 10:     {int(top.sum()):5}  ({top.mean() * 100:.1f}%)")
    print(f"  non-Top 10: {int((~top).sum()):5}  ({(~top).mean() * 100:.1f}%)")

    print("\n  coverage within each group:")
    for label, subset in [("Top 10", df[top]), ("non-Top 10", df[~top])]:
        lyrics = subset["plain_lyrics"].apply(lambda v: bool(cell_text(v))).mean() * 100
        genre = subset["primary_genre"].apply(lambda v: bool(cell_text(v))).mean() * 100
        mbid = subset["recording_mbid"].apply(lambda v: bool(cell_text(v))).mean() * 100
        print(f"    {label:11} lyrics {lyrics:5.1f}%   genre {genre:5.1f}%   musicbrainz {mbid:5.1f}%")

    print("\n  primary_genre by group (% within group):")
    share = pd.crosstab(df["primary_genre"], top, normalize="columns") * 100
    share.columns = ["non-Top 10", "Top 10"]
    print(share.round(1).sort_values("Top 10", ascending=False).to_string())


if __name__ == "__main__":
    main()
