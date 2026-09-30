"""
Step 8: Clean the lyrics text and derive the analysis features.

Reads data/clean/songs_merged.csv and writes data/clean/songs_analysis.csv:
every merged column, plus cleaned lyrics and the features built from them.

The raw lyrics column is kept alongside the cleaned one, so any cleaning
decision can be checked or redone without re-fetching.

Features are left BLANK, never zero, when a song has no lyrics or is an
instrumental: a song with no lyrics did not score 0 words, we simply do not
know. Zero would be a real measurement and would drag every average down.

Run:
    python scripts/08_features.py --holiday-list   # just the is_holiday proposal
    python scripts/08_features.py                  # write songs_analysis.csv
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from artist_utils import cell_text, normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_PATH = PROJECT_ROOT / "data" / "clean" / "songs_merged.csv"
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "all.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "clean" / "songs_analysis.csv"

# ---------------------------------------------------------------------------
# Lyrics cleaning
# ---------------------------------------------------------------------------
# "(x2)", "[x3]", "(2x)", "x4" at the end of a line, and the × character.
REPEAT_MARKER = re.compile(
    r"\s*[\(\[]?\s*(?:[x×]\s*(\d{1,2})|(\d{1,2})\s*[x×])\s*[\)\]]?\s*$", re.IGNORECASE
)
LRC_TIMESTAMP = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
# LRC file metadata that LRCLIB sometimes leaves in the plain text.
LRC_METADATA = re.compile(r"\[(?:ti|ar|al|by|offset|length|re|ve)\s*:[^\]]*\]", re.IGNORECASE)
# "[Chorus]", "[Verse 2: Drake]", "[Outro: 21 Savage]", and the "[?]" / "[__]"
# placeholders LRCLIB uses for words the transcriber could not make out.
SECTION_LABEL = re.compile(r"\[[^\]]{0,60}\]")

# Strong profanity only, matched whole-word, with the asterisk-censored forms
# LRCLIB transcriptions often use. Mild words ("damn", "hell") are deliberately
# excluded -- they are common enough to flag most songs and say little.
PROFANITY = re.compile(
    r"\b(?:"
    r"f+u+c+k+\w*|f\*+c?k\w*|f\*{2,}\w*"
    r"|sh[i!\*]+t+\w*|s\*{2,}t?\w*"
    r"|b[i!\*]+t?ch\w*|b\*{2,}ch\w*"
    r"|n[i!\*]+gg[ae]r?\w*|n\*{2,}a\w*"
    r"|cunt\w*|pussy\w*|dick(?:head|s)?|cock(?:s|sucker)?"
    r"|motherfuck\w*|mothafuck\w*|asshole\w*|bastard\w*"
    r"|whore\w*|slut\w*|\bhoes?\b"
    r")\b",
    re.IGNORECASE,
)


def expand_repeats(line: str) -> list[str]:
    """Turn 'Say it again (x3)' into the line repeated three times."""
    match = REPEAT_MARKER.search(line)
    if not match:
        return [line]
    count = int(match.group(1) or match.group(2))
    body = line[: match.start()].strip()
    if not body or not 2 <= count <= 20:
        return [line]
    return [body] * count


def clean_lyrics(raw) -> str:
    """Strip structure markup, expand repeats, normalize whitespace."""
    text = cell_text(raw)
    if not text:
        return ""

    text = LRC_TIMESTAMP.sub(" ", text)
    text = LRC_METADATA.sub(" ", text)

    lines = []
    for line in text.split("\n"):
        for piece in expand_repeats(line):
            piece = SECTION_LABEL.sub(" ", piece)
            piece = re.sub(r"[ \t]+", " ", piece).strip()
            if piece:
                lines.append(piece)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lyric features
# ---------------------------------------------------------------------------
WORD = re.compile(r"[a-z0-9']+")


def lyric_features(cleaned: str) -> dict:
    lines = [line for line in cleaned.split("\n") if line.strip()]
    words = WORD.findall(cleaned.lower())
    if not words or not lines:
        return {}

    counts = Counter(lines)
    repeated_lines = sum(n for n in counts.values() if n > 1)

    return {
        "word_count": len(words),
        "unique_word_ratio": round(len(set(words)) / len(words), 4),
        "repetition_score": round(repeated_lines / len(lines), 4),
        "avg_line_length": round(len(words) / len(lines), 2),
        "has_profanity": bool(PROFANITY.search(cleaned)),
    }


# ---------------------------------------------------------------------------
# is_holiday -- PROPOSED RULE, under review
# ---------------------------------------------------------------------------
# A song is seasonal if its title names the season, OR if it RECURS: a chart run
# concentrated in December/January across two or more separate seasons.
#
# Recurrence is what makes the second half work. "Charted mostly in Dec/Jan" on
# its own flagged 475 songs, because any album track released in December has a
# one-week run that is trivially 100% seasonal -- it caught most of Meek Mill's
# "Championships" and half of XXXTENTACION's catalogue. A perennial comes back
# every year; an album track does not. Adding the two-season test cuts the flag
# to 36 songs, all genuinely seasonal.
#
# A December is paired with the January after it, so one season spanning the new
# year counts once rather than twice.
HOLIDAY_TITLE = re.compile(
    r"\bchristmas\b|\bxmas\b|\bsanta\b|\bholiday\b|\bjingle\b|\bsleigh\b"
    r"|\brudolph\b|\bfrosty\b|\bmistletoe\b|\bnoel\b|\bnavidad\b"
    r"|\bauld lang syne\b|\bwinter wonderland\b|\bsilent night\b"
    r"|\bdrummer boy\b|\bdeck the halls\b|\blet it snow\b|\bsnowman\b"
    r"|\bhanukkah\b|\bnew year\b|\bgrinch\b|\bscrooge\b|\bwonderful time\b",
    re.IGNORECASE,
)
HOLIDAY_SEASON_SHARE = 0.70
HOLIDAY_MIN_SEASONS = 2

# Titles that trip a keyword but are not seasonal songs. Both happen to have
# charted in December, so no rule based on timing can catch them.
HOLIDAY_EXCEPTIONS = {
    ("Santa Fe", "Zach Bryan"),        # the city, not Santa Claus
    ("Holiday Road", "Kesha"),         # the National Lampoon's Vacation song
}


def chart_season_stats() -> dict:
    """Per song: total chart weeks, Dec/Jan weeks, and distinct seasons."""
    with open(ARCHIVE_PATH) as f:
        archive = json.load(f)
    stats = {}
    for week in archive:
        year, month = int(week["date"][:4]), int(week["date"][5:7])
        seasonal = month in (12, 1)
        for entry in week["data"]:
            key = (entry["song"], entry["artist"])
            total, count, seasons = stats.get(key, (0, 0, set()))
            if seasonal:
                # December belongs to the season that ends the following January.
                seasons = seasons | {year if month == 12 else year - 1}
            stats[key] = (total + 1, count + seasonal, seasons)
    return stats


def add_holiday(df: pd.DataFrame) -> pd.DataFrame:
    stats = chart_season_stats()

    # A merged song's weeks are spread across its source credits, so pool them.
    def pooled(row):
        keys = [(row["title"], row["artist"])]
        for part in cell_text(row.get("merged_from")).split("|"):
            if " :: " in part:
                title, artist = part.split(" :: ", 1)
                keys.append((title.strip(), artist.strip()))
        total = count = 0
        seasons = set()
        for key in keys:
            if key in stats:
                t, c, s = stats[key]
                total, count, seasons = total + t, count + c, seasons | s
        if not total:
            return pd.Series({"dec_jan_share": None, "dec_jan_seasons": 0})
        return pd.Series({"dec_jan_share": round(count / total, 4),
                          "dec_jan_seasons": len(seasons)})

    df = pd.concat([df, df.apply(pooled, axis=1)], axis=1)

    # A title keyword alone is not enough: "santa" is Spanish for "saint"
    # ("La Santa"), and it turns up in place names ("7969 Santa"). A real
    # holiday song charts in the season at least once, so require both.
    title_hit = (
        df["title"].apply(lambda t: bool(HOLIDAY_TITLE.search(cell_text(t))))
        & (df["dec_jan_share"].fillna(0) > 0)
    )
    season_hit = ((df["dec_jan_share"].fillna(0) >= HOLIDAY_SEASON_SHARE)
                  & (df["dec_jan_seasons"] >= HOLIDAY_MIN_SEASONS))
    excepted = [
        (cell_text(a), cell_text(b)) in HOLIDAY_EXCEPTIONS
        for a, b in zip(df["title"], df["artist"])
    ]
    df["is_holiday"] = (title_hit | season_hit) & ~pd.Series(excepted, index=df.index)
    df["holiday_reason"] = [
        "title+season" if t and s else "title" if t else "season" if s else ""
        for t, s in zip(title_hit, season_hit)
    ]
    return df


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    df = pd.read_csv(MERGED_PATH)

    # Peak is not final for a song still on the chart, so it cannot be used as
    # a training label. It stays in the file for descriptive work.
    df["in_model"] = ~df["still_charting"].astype(bool)

    df["lyrics_clean"] = df["plain_lyrics"].apply(clean_lyrics)
    instrumental = df["instrumental"].fillna(False).astype(bool)
    df.loc[instrumental, "lyrics_clean"] = ""

    features = [lyric_features(text) if text else {} for text in df["lyrics_clean"]]
    for column in ["word_count", "unique_word_ratio", "repetition_score",
                   "avg_line_length", "has_profanity"]:
        df[column] = [f.get(column) for f in features]

    # Days between release and chart entry. Only full dates can answer this;
    # a year-only release date would have to guess a month and a day.
    release = pd.to_datetime(
        df["first_release_date"].where(
            df["first_release_date"].astype(str).str.len() == 10
        ),
        errors="coerce",
    )
    debut = pd.to_datetime(df["first_chart_date"], errors="coerce")
    df["days_release_to_chart"] = (debut - release).dt.days
    df["debut_month"] = debut.dt.month

    return add_model_split(add_album_features(add_holiday(df)))


# ---------------------------------------------------------------------------
# Album releases
# ---------------------------------------------------------------------------
# When an album drops, every track can enter the chart at once. Those songs
# compete with each other and mostly fall straight back out, so they behave
# differently from a single released on its own.
#
# Leakage-safe: both columns count only what is visible in the debut week
# itself -- how many songs by this artist entered the chart alongside it. No
# later week is consulted.
# The effect is U-shaped, so it is bucketed rather than thresholded. A boolean
# at any single cutoff merges two opposite effects: album filler drags the
# middle down, while only a superstar can place ten songs at once and those
# chart high. See the README.
ALBUM_BUCKETS = [0, 1, 4, 9, 10_000]
ALBUM_BUCKET_LABELS = ["1", "2-4", "5-9", "10+"]

# Songs from one album drop are not independent draws, so the split is by time,
# not at random: a random split would put tracks from the same release on both
# sides and leak.
TRAIN_YEARS = range(2018, 2025)
TEST_YEARS = range(2025, 2027)


def add_album_features(df: pd.DataFrame) -> pd.DataFrame:
    key = [
        (normalize_text(a), d)
        for a, d in zip(df["primary_artist"], df["first_chart_date"])
    ]
    counts = Counter(key)
    df["same_artist_debuts_that_week"] = [counts[k] for k in key]
    df["album_debut_bucket"] = pd.cut(
        df["same_artist_debuts_that_week"],
        ALBUM_BUCKETS, labels=ALBUM_BUCKET_LABELS,
    ).astype(str)
    return df


def add_model_split(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based train/test split, blank for songs that cannot be labelled."""
    split = []
    for year, usable in zip(df["debut_year"], df["in_model"]):
        if not usable:
            split.append("")            # peak not final, so no label to learn
        elif year in TRAIN_YEARS:
            split.append("train")
        elif year in TEST_YEARS:
            split.append("test")
        else:
            split.append("")
    df["model_split"] = split
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
NEW_COLUMNS = {
    "in_model": "False for songs still on the chart, whose peak is not final",
    "lyrics_clean": "lyrics with section labels, timestamps and LRC tags removed",
    "word_count": "words in the cleaned lyrics",
    "unique_word_ratio": "distinct words / total words; lower means more repetitive",
    "repetition_score": "share of lines that appear more than once",
    "avg_line_length": "average words per line",
    "has_profanity": "cleaned lyrics contain strong profanity",
    "days_release_to_chart": "days from first release to chart debut",
    "debut_month": "calendar month of the chart debut, 1-12",
    "dec_jan_share": "share of the song's chart weeks falling in December or January",
    "dec_jan_seasons": "how many separate Dec/Jan seasons the song charted in",
    "is_holiday": "seasonal song, by title keyword or December/January chart run",
    "holiday_reason": "which half of the is_holiday rule fired",
    "same_artist_debuts_that_week": "songs by the same primary artist entering the chart that same week",
    "album_debut_bucket": "that count as a band: 1, 2-4, 5-9, 10+",
    "model_split": "train (2018-2024), test (2025-2026), blank where in_model is False",
}


def print_album_report(df: pd.DataFrame):
    top = df["top10"].astype(bool)

    print(f"\n{'=' * 78}\nSAME-ARTIST DEBUTS THAT WEEK\n{'=' * 78}")
    print("  count  songs   share   Top 10 rate")
    dist = df["same_artist_debuts_that_week"].value_counts().sort_index()
    for value, n in dist.items():
        rate = top[df["same_artist_debuts_that_week"] == value].mean() * 100
        print(f"  {value:5}  {n:5}  {n / len(df) * 100:5.1f}%  {rate:10.1f}%")

    print("\n  album_debut_bucket:")
    for label in ALBUM_BUCKET_LABELS:
        group = df[df["album_debut_bucket"] == label]
        rate = group["top10"].astype(bool).mean() * 100
        print(f"    {label:6} {len(group):5} songs   Top 10 {rate:5.1f}%")

    print(f"\n{'=' * 78}\nDEBUT MONTH, 5+ ALBUM DEBUTS EXCLUDED\n{'=' * 78}")
    clean = df[df["album_debut_bucket"].isin(["1", "2-4"])]
    ctop = clean["top10"].astype(bool)
    all_share = pd.crosstab(df["debut_month"], top, normalize="columns") * 100
    cut_share = pd.crosstab(clean["debut_month"], ctop, normalize="columns") * 100
    print("  month   Top10 all   Top10 no-bomb   non-Top10 no-bomb")
    for month in range(1, 13):
        a = all_share.loc[month, True] if month in all_share.index else 0
        b = cut_share.loc[month, True] if month in cut_share.index else 0
        c = cut_share.loc[month, False] if month in cut_share.index else 0
        print(f"  {month:5}   {a:9.1f}%   {b:13.1f}%   {c:17.1f}%")

    print(f"\n{'=' * 78}\nBIGGEST ALBUM-BOMB WEEKS\n{'=' * 78}")
    bombs = df[df["album_debut_bucket"].isin(["5-9", "10+"])].copy()
    grouped = bombs.groupby([bombs["primary_artist"], bombs["first_chart_date"]]).agg(
        songs=("title", "size"),
        top10=("top10", lambda s: int(s.astype(bool).sum())),
        best_peak=("peak_position", "min"),
    ).sort_values("songs", ascending=False).head(10)
    print(f"  {'artist':28} {'debut week':12} {'songs':>6} {'Top 10':>7} {'best peak':>10}")
    for (artist, date), row in grouped.iterrows():
        print(f"  {str(artist)[:28]:28} {str(date)[:10]:12} {row.songs:6} "
              f"{row.top10:7} {row.best_peak:10}")


def print_holiday_list(df: pd.DataFrame):
    flagged = df[df["is_holiday"]].sort_values(
        ["holiday_reason", "dec_jan_share"], ascending=[True, False]
    )
    print(f"\n{'=' * 78}\nis_holiday -- PROPOSED, {len(flagged)} songs flagged\n{'=' * 78}")
    print(f"Rule: title keyword OR (at least {HOLIDAY_SEASON_SHARE:.0%} of chart weeks in "
          f"Dec/Jan AND at least {HOLIDAY_MIN_SEASONS} separate seasons)\n")
    for reason, group in flagged.groupby("holiday_reason"):
        print(f"-- matched by {reason} ({len(group)}) " + "-" * 40)
        for row in group.itertuples():
            share = "" if pd.isna(row.dec_jan_share) else f"{row.dec_jan_share:.0%}"
            print(f"   {row.title[:42]:42} {row.artist[:24]:24} dec/jan {share:>4}"
                  f"  seasons {int(row.dec_jan_seasons)}  peak {row.peak_position}")


def print_report(df: pd.DataFrame):
    print(f"\n{'=' * 78}\nDATA DICTIONARY -- new columns\n{'=' * 78}")
    for column, description in NEW_COLUMNS.items():
        filled = df[column].apply(lambda v: bool(cell_text(v))).sum()
        print(f"  {column:22} {filled:5}/{len(df)} filled   {description}")

    print(f"\n{'=' * 78}\nTOP 10 vs NON-TOP 10\n{'=' * 78}")
    top = df["top10"].astype(bool)
    print(f"  {'column':24} {'Top 10':>12} {'non-Top 10':>12}")
    for column in ["word_count", "unique_word_ratio", "repetition_score",
                   "avg_line_length", "days_release_to_chart"]:
        a, b = df.loc[top, column].median(), df.loc[~top, column].median()
        print(f"  {column:24} {a:12.3f} {b:12.3f}   (median)")
    for column in ["has_profanity", "is_holiday", "in_model"]:
        a = df.loc[top, column].fillna(False).mean() * 100
        b = df.loc[~top, column].fillna(False).mean() * 100
        print(f"  {column:24} {a:11.1f}% {b:11.1f}%")

    print("\n  debut_month, share of each group:")
    share = pd.crosstab(df["debut_month"], top, normalize="columns") * 100
    share.columns = ["non-Top 10", "Top 10"]
    print(share.round(1).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holiday-list", action="store_true",
                        help="Only show the is_holiday proposal; write nothing.")
    args = parser.parse_args()

    df = build()
    print_holiday_list(df)
    if args.holiday_list:
        print("\n[--holiday-list] nothing written.")
        return

    print_report(df)
    print_album_report(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
