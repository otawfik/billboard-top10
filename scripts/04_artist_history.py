"""
Step 4: Build "what had this artist done before?" features for every song.

Reads data/raw/all.json (the full 1958-2026 archive) and data/clean/billboard_songs.csv.
Makes no API calls.

The point of this script is to describe an artist's track record AT THE MOMENT a
song debuted, so the features can be used to predict whether that song reaches
the Top 10. That means nothing from the debut week or later may be counted. Every
lookup counts only songs that debuted STRICTLY BEFORE this song's debut date,
which also excludes the song itself and any sibling song that debuted the same
week. verify_no_leakage() re-checks this with an independent method and stops the
script if it ever fails.

Output: data/clean/artist_history.csv, one row per song, joinable on title + artist.
"""

import json
import random
from bisect import bisect_left
from pathlib import Path

import pandas as pd

from artist_utils import has_feature, normalize_text, split_credit
from song_utils import find_merge_groups

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "all.json"
SONGS_PATH = PROJECT_ROOT / "data" / "clean" / "billboard_songs.csv"

# Matches script 01: duplicate credits are merged only for the analysis window,
# since those are the pairs that were reviewed.
CUTOFF_DATE = pd.Timestamp("2018-01-01")
OUTPUT_PATH = PROJECT_ROOT / "data" / "clean" / "artist_history.csv"

VERIFY_SAMPLE_SIZE = 250
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Archive -> one row per song, across ALL of chart history
# ---------------------------------------------------------------------------
def load_archive_songs() -> pd.DataFrame:
    """Every song in the 1958-2026 archive with its debut date and peak position."""
    with open(ARCHIVE_PATH, "r") as f:
        archive = json.load(f)

    rows = []
    for week in archive:
        date = week["date"]
        for entry in week["data"]:
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

    weeks = pd.DataFrame(rows)
    weeks["date"] = pd.to_datetime(weeks["date"])

    songs = (
        weeks.groupby(["song", "artist"], sort=False)
        .agg(
            debut_date=("date", "min"),
            last_date=("date", "max"),
            peak_position=("this_week", "min"),
        )
        .reset_index()
    )

    chart_dates = (
        weeks.groupby(["song", "artist"])["date"]
        .apply(lambda s: set(s.dt.strftime("%Y-%m-%d")))
        .to_dict()
    )
    return collapse_duplicate_credits(songs, chart_dates)


def collapse_duplicate_credits(songs: pd.DataFrame, chart_dates: dict) -> pd.DataFrame:
    """Apply the same merge script 01 applies, so one song is one chart entry.

    Without this the history index would count a re-credited song twice, and an
    artist like Taylor Swift would gain a phantom prior entry from "Karma"
    appearing under two credits. Only songs debuting in the analysis window are
    merged, matching the pairs that were reviewed.
    """
    recent = songs[songs["debut_date"] >= CUTOFF_DATE].copy()
    candidates = recent.rename(columns={"song": "title"})[
        ["title", "artist", "debut_date", "last_date"]
    ]

    groups = find_merge_groups(candidates, chart_dates)
    if not groups:
        return songs

    positions = recent.index.to_list()
    drop = []
    for members in groups:
        rows = recent.loc[[positions[i] for i in members]].sort_values("debut_date")
        keep = rows.index[0]
        songs.loc[keep, "peak_position"] = int(rows["peak_position"].min())
        songs.loc[keep, "debut_date"] = rows["debut_date"].min()
        songs.loc[keep, "last_date"] = rows["last_date"].max()
        drop.extend(rows.index[1:])

    print(f"  collapsed {len(drop)} duplicate credit rows into {len(groups)} songs")
    return songs.drop(index=drop).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-artist history index
# ---------------------------------------------------------------------------
class ArtistHistory:
    """Debut dates and running Top 10 / #1 counts for one artist, sorted by date.

    prefix_top10[k] is the number of Top 10 songs among the artist's first k
    songs, so a single bisect gives every prior count at once.
    """

    def __init__(self, dates, top10_flags, number1_flags):
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        self.dates = [dates[i] for i in order]

        self.prefix_top10 = [0]
        self.prefix_number1 = [0]
        for i in order:
            self.prefix_top10.append(self.prefix_top10[-1] + top10_flags[i])
            self.prefix_number1.append(self.prefix_number1[-1] + number1_flags[i])

    def priors_before(self, date) -> dict:
        """Counts over songs that debuted STRICTLY BEFORE `date`."""
        k = bisect_left(self.dates, date)
        return {
            "prior_entries": k,
            "prior_top10s": self.prefix_top10[k],
            "prior_number1s": self.prefix_number1[k],
            "last_prior_debut": self.dates[k - 1] if k > 0 else None,
        }


def build_artist_index(archive_songs: pd.DataFrame) -> dict[str, ArtistHistory]:
    """Map normalized artist name -> ArtistHistory, using every credit in history.

    An artist's track record includes songs they were featured on, so a song is
    filed under every artist named in its credit, not just the primary one.
    """
    collected: dict[str, dict[str, list]] = {}

    for row in archive_songs.itertuples():
        top10 = 1 if row.peak_position <= 10 else 0
        number1 = 1 if row.peak_position == 1 else 0
        for artist in split_credit(row.artist):
            key = normalize_text(artist)
            if not key:
                continue
            bucket = collected.setdefault(key, {"dates": [], "top10": [], "number1": []})
            bucket["dates"].append(row.debut_date)
            bucket["top10"].append(top10)
            bucket["number1"].append(number1)

    return {
        key: ArtistHistory(b["dates"], b["top10"], b["number1"])
        for key, b in collected.items()
    }


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------
EMPTY_PRIORS = {
    "prior_entries": 0,
    "prior_top10s": 0,
    "prior_number1s": 0,
    "last_prior_debut": None,
}


def build_features(songs: pd.DataFrame, index: dict[str, ArtistHistory]) -> pd.DataFrame:
    rows = []

    for song in songs.itertuples():
        debut = song.first_chart_date
        credited = split_credit(song.artist)
        primary = credited[0] if credited else song.artist

        history = index.get(normalize_text(primary))
        priors = history.priors_before(debut) if history else dict(EMPTY_PRIORS)

        # Best track record among everyone credited on the song, not just the lead.
        best_prior_top10s = 0
        for artist in credited:
            other = index.get(normalize_text(artist))
            if other:
                best_prior_top10s = max(
                    best_prior_top10s, other.priors_before(debut)["prior_top10s"]
                )

        last_prior = priors["last_prior_debut"]
        weeks_since = (debut - last_prior).days // 7 if last_prior is not None else None

        rows.append(
            {
                "title": song.title,
                "artist": song.artist,
                "primary_artist": primary,
                "all_credited_artists": "|".join(credited),
                "n_credited_artists": len(credited),
                "has_feature": has_feature(song.artist),
                "prior_entries": priors["prior_entries"],
                "prior_top10s": priors["prior_top10s"],
                "prior_number1s": priors["prior_number1s"],
                "weeks_since_last_entry": weeks_since,
                "is_first_entry": priors["prior_entries"] == 0,
                "best_prior_top10s": best_prior_top10s,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------
def verify_no_leakage(features: pd.DataFrame, songs: pd.DataFrame, archive_songs: pd.DataFrame):
    """Prove that no information from a song's debut week or later was used.

    Check 1 re-derives the counts on a random sample with a plain list filter --
    a different algorithm from the bisect/prefix-sum path, so agreement is
    meaningful rather than circular.
    Check 2 asserts the most recent prior entry predates the song's own debut.
    Check 3 asserts first entries carry no history at all.
    """
    print("\n--- Leakage verification ---")

    merged = features.merge(
        songs[["title", "artist", "first_chart_date"]], on=["title", "artist"], how="left"
    )

    # Rebuild a plain per-artist list of (debut_date, peak) for brute-force checks.
    by_artist: dict[str, list[tuple]] = {}
    for row in archive_songs.itertuples():
        for artist in split_credit(row.artist):
            key = normalize_text(artist)
            if key:
                by_artist.setdefault(key, []).append((row.debut_date, row.peak_position))

    # Check 1: independent recomputation on a random sample.
    random.seed(RANDOM_SEED)
    sample = merged.sample(min(VERIFY_SAMPLE_SIZE, len(merged)), random_state=RANDOM_SEED)
    for row in sample.itertuples():
        debut = row.first_chart_date
        entries = by_artist.get(normalize_text(row.primary_artist), [])
        prior = [(d, p) for d, p in entries if d < debut]

        assert len(prior) == row.prior_entries, f"prior_entries mismatch: {row.title}"
        assert sum(p <= 10 for _, p in prior) == row.prior_top10s, f"prior_top10s mismatch: {row.title}"
        assert sum(p == 1 for _, p in prior) == row.prior_number1s, f"prior_number1s mismatch: {row.title}"

        if prior:
            expected_weeks = (debut - max(d for d, _ in prior)).days // 7
            assert expected_weeks == row.weeks_since_last_entry, f"recency mismatch: {row.title}"
    print(f"  [pass] {len(sample)} songs recomputed by an independent method, all counts agree")

    # Check 2: the most recent prior entry is strictly before this song's debut.
    with_history = merged[merged["weeks_since_last_entry"].notna()]
    assert (with_history["weeks_since_last_entry"] > 0).all(), \
        "a prior entry debuted on or after the song's own debut week"
    print(f"  [pass] all {len(with_history)} songs with history reference a strictly earlier debut "
          f"(min gap {int(with_history['weeks_since_last_entry'].min())} week(s))")

    # Check 3: first entries carry no history.
    firsts = merged[merged["is_first_entry"]]
    assert (firsts[["prior_entries", "prior_top10s", "prior_number1s"]] == 0).all().all(), \
        "a first entry has non-zero prior counts"
    assert firsts["weeks_since_last_entry"].isna().all(), \
        "a first entry has a recency value"
    print(f"  [pass] all {len(firsts)} first entries have zero history and no recency value")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(features: pd.DataFrame, songs: pd.DataFrame, archive_songs: pd.DataFrame):
    print("\n--- Summary ---")
    print(f"Songs: {len(features)}")
    print(f"First-time entries: {features['is_first_entry'].mean() * 100:.1f}% "
          f"({features['is_first_entry'].sum()})")
    print(f"Songs with a 'Featuring' credit: {features['has_feature'].mean() * 100:.1f}%")
    print(f"Mean credited artists per song: {features['n_credited_artists'].mean():.2f}")

    print("\nprior_entries (primary artist):")
    print(features["prior_entries"].describe().to_string())
    print("\nprior_top10s (primary artist):")
    print(features["prior_top10s"].value_counts().sort_index().head(8).to_string())
    print("\nweeks_since_last_entry (artists with history):")
    print(features["weeks_since_last_entry"].describe().to_string())

    # Confirm the history really reaches back past the 2018 cutoff.
    pre_2018 = archive_songs[archive_songs["debut_date"] < pd.Timestamp("2018-01-01")]
    print(f"\nArchive songs available as history: {len(archive_songs)} "
          f"({len(pre_2018)} debuted before 2018)")

    # Quick look at whether the features point the way we would expect.
    merged = features.merge(songs[["title", "artist", "top10"]], on=["title", "artist"], how="left")
    print("\nTop 10 rate by prior success (sanity check, not a model):")
    print(f"  first-time artists:        {merged[merged['is_first_entry']]['top10'].mean() * 100:.1f}%")
    print(f"  artists with 0 prior top10s: {merged[merged['prior_top10s'] == 0]['top10'].mean() * 100:.1f}%")
    print(f"  artists with 1-4 prior top10s: "
          f"{merged[merged['prior_top10s'].between(1, 4)]['top10'].mean() * 100:.1f}%")
    print(f"  artists with 5+ prior top10s: {merged[merged['prior_top10s'] >= 5]['top10'].mean() * 100:.1f}%")


def print_examples(features: pd.DataFrame):
    """Five rows chosen to be eyeballed: a superstar, a debut, a feature, extremes."""
    cols = [
        "title", "primary_artist", "n_credited_artists", "has_feature",
        "prior_entries", "prior_top10s", "prior_number1s",
        "weeks_since_last_entry", "is_first_entry", "best_prior_top10s",
    ]

    picks = []

    taylor = features[features["primary_artist"].str.lower() == "taylor swift"]
    if len(taylor):
        picks.append(("Big artist (Taylor Swift, latest song)",
                      taylor.sort_values("prior_entries").iloc[-1]))

    firsts = features[features["is_first_entry"]]
    if len(firsts):
        picks.append(("First-ever chart entry", firsts.iloc[0]))

    features_with_feat = features[features["has_feature"] & (features["n_credited_artists"] > 1)]
    if len(features_with_feat):
        picks.append(("Featured credit (best_prior_top10s spans all artists)",
                      features_with_feat.sort_values("best_prior_top10s").iloc[-1]))

    picks.append(("Most prior entries of any song", features.sort_values("prior_entries").iloc[-1]))

    gaps = features.dropna(subset=["weeks_since_last_entry"])
    if len(gaps):
        picks.append(("Longest gap since last entry",
                      gaps.sort_values("weeks_since_last_entry").iloc[-1]))

    print("\n--- Examples ---")
    for label, row in picks:
        print(f"\n{label}:")
        print(f"  {row['title']} - {row['primary_artist']}")
        for col in cols[2:]:
            print(f"    {col}: {row[col]}")


def main():
    print("Loading archive...")
    archive_songs = load_archive_songs()
    print(f"  {len(archive_songs)} songs across all of chart history")

    songs = pd.read_csv(SONGS_PATH, parse_dates=["first_chart_date", "last_chart_date"])
    print(f"  {len(songs)} songs debuting 2018 or later")

    print("Building artist index...")
    index = build_artist_index(archive_songs)
    print(f"  {len(index)} unique credited artists")

    print("Building features...")
    features = build_features(songs, index)

    verify_no_leakage(features, songs, archive_songs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(features)} rows to {OUTPUT_PATH}")

    print_summary(features, songs, archive_songs)
    print_examples(features)


if __name__ == "__main__":
    main()
