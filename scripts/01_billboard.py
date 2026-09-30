"""
Step 1: Build the Billboard Hot 100 "songs" table for songs that debuted in 2018 or later.

Downloads the full Billboard Hot 100 weekly-chart archive (1958-present) and, for
each song, computes peak position / weeks on chart using EVERY week the song ever
charted (not just weeks from 2018 onward). This matters because a song that first
charted in, say, late 2017 could still climb the chart into 2018 -- if we threw
away its pre-2018 weeks before computing peak position, we'd get the wrong number.
Instead we aggregate across the whole archive first, and only filter down to
"debuted in 2018+" as the very last step.

Output: data/clean/billboard_songs.csv, one row per song.
"""

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from song_utils import find_merge_groups

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
ARCHIVE_PATH = RAW_DIR / "all.json"
OUTPUT_PATH = CLEAN_DIR / "billboard_songs.csv"

ARCHIVE_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"
CUTOFF_DATE = pd.Timestamp("2018-01-01")


def download_archive():
    """Download the full Billboard Hot 100 archive, unless we already have it."""
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists at {ARCHIVE_PATH}, skipping download.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading archive from {ARCHIVE_URL} ...")
    urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)
    print(f"Saved to {ARCHIVE_PATH}")


def load_all_weeks() -> pd.DataFrame:
    """Flatten the archive (list of {date, data: [...]}) into one row per
    (date, song, artist, this_week) across the entire chart history."""
    with open(ARCHIVE_PATH, "r") as f:
        archive = json.load(f)

    rows = []
    for week in archive:
        date = week["date"]
        for entry in week["data"]:
            # Some historical rows can have a null this_week/position; skip those.
            if entry.get("this_week") is None:
                continue
            rows.append(
                {
                    "date": date,
                    "song": entry["song"],
                    "artist": entry["artist"],
                    "this_week": entry["this_week"],
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def merge_duplicate_credits(songs: pd.DataFrame, weeks_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that are one song Billboard listed under two credits.

    The credit from the DEBUT week becomes the canonical artist, because the
    predictor features in script 04 describe what was known when the song
    entered the chart: if Ice Spice joins "Karma" in week 30, her chart history
    must not be attributed to the song's debut. The last credit is kept in
    final_artist for reference, and merged_from records both source rows.
    """
    chart_dates = (
        weeks_df.groupby(["song", "artist"])["date"]
        .apply(lambda s: set(s.dt.strftime("%Y-%m-%d")))
        .to_dict()
    )

    songs = songs.reset_index(drop=True)
    candidates = songs.rename(
        columns={"song": "title", "first_chart_date": "debut_date", "last_chart_date": "last_date"}
    )[["title", "artist", "debut_date", "last_date"]]

    groups = find_merge_groups(candidates, chart_dates)
    if not groups:
        songs["final_artist"] = songs["artist"]
        songs["merged_from"] = ""
        return songs

    merged_rows = []
    drop = set()
    for members in groups:
        rows = songs.loc[members].sort_values("first_chart_date")
        debut, final = rows.iloc[0], rows.iloc[-1]

        # Weeks must be distinct chart dates, not the sum of both runs.
        dates = set()
        for row in rows.itertuples():
            dates |= chart_dates.get((row.song, row.artist), set())

        merged_rows.append({
            "song": debut.song,
            "artist": debut.artist,
            "first_chart_date": rows["first_chart_date"].min(),
            "last_chart_date": rows["last_chart_date"].max(),
            "peak_position": int(rows["peak_position"].min()),
            "weeks_on_chart": len(dates),
            "debut_position": int(debut.debut_position),
            "final_artist": final.artist,
            "merged_from": "|".join(f"{r.song} :: {r.artist}" for r in rows.itertuples()),
        })
        drop.update(members)

    kept = songs.drop(index=list(drop)).copy()
    kept["final_artist"] = kept["artist"]
    kept["merged_from"] = ""

    out = pd.concat([kept, pd.DataFrame(merged_rows)], ignore_index=True)
    print(f"  merged {len(drop)} rows into {len(groups)} songs "
          f"({len(drop) - len(groups)} rows removed)")
    return out


def build_song_table(weeks_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every chart week into one row per song, using the FULL history
    of each song (including any weeks before 2018) so peak position and weeks
    on chart are correct."""
    most_recent_chart_date = weeks_df["date"].max()

    # Sort so the first row within each song group is its earliest chart week --
    # needed to read off debut_position correctly.
    weeks_df = weeks_df.sort_values("date")

    grouped = weeks_df.groupby(["song", "artist"], sort=False)

    songs = grouped.agg(
        first_chart_date=("date", "min"),
        last_chart_date=("date", "max"),
        peak_position=("this_week", "min"),
        weeks_on_chart=("date", "nunique"),
    ).reset_index()

    # debut_position = this_week on the song's first chart date.
    debut_rows = weeks_df.drop_duplicates(subset=["song", "artist"], keep="first")
    debut_rows = debut_rows.rename(columns={"this_week": "debut_position"})[
        ["song", "artist", "debut_position"]
    ]
    songs = songs.merge(debut_rows, on=["song", "artist"], how="left")

    # Keep only songs whose first chart appearance is 2018-01-01 or later.
    songs = songs[songs["first_chart_date"] >= CUTOFF_DATE].copy()

    songs = merge_duplicate_credits(songs, weeks_df)

    songs["top10"] = songs["peak_position"] <= 10
    songs["debut_year"] = songs["first_chart_date"].dt.year
    songs["still_charting"] = songs["last_chart_date"] == most_recent_chart_date

    songs = songs.rename(columns={"song": "title"})
    songs = songs[
        [
            "title",
            "artist",
            "final_artist",
            "merged_from",
            "first_chart_date",
            "last_chart_date",
            "peak_position",
            "weeks_on_chart",
            "debut_position",
            "top10",
            "debut_year",
            "still_charting",
        ]
    ]
    songs = songs.sort_values("first_chart_date").reset_index(drop=True)
    return songs


def print_summary(songs: pd.DataFrame):
    total = len(songs)
    top10_count = int(songs["top10"].sum())
    print("\n--- Summary ---")
    print(f"Total songs (debuted 2018+): {total}")
    print(f"Top 10 songs: {top10_count}")
    print("\nSongs per debut year:")
    print(songs["debut_year"].value_counts().sort_index().to_string())


def main():
    download_archive()
    weeks_df = load_all_weeks()
    songs = build_song_table(weeks_df)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    songs.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(songs)} songs to {OUTPUT_PATH}")

    print_summary(songs)


if __name__ == "__main__":
    main()
