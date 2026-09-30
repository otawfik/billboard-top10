"""
Step 5: Collapse MusicBrainz's many genre labels into a few broad genres.

MusicBrainz uses hundreds of fine-grained labels ("southern hip hop", "cloud
rap", "dance-pop"). For analysis we want one broad genre per song, plus the
full set of broad genres a song touches.

Genre source, in order:
  1. the song's own MusicBrainz genres
  2. otherwise the primary credited artist's genres
  3. otherwise none
genre_source records which was used.

Run:
    python scripts/05_genres.py --preview   # mapping table + coverage, writes nothing
    python scripts/05_genres.py             # writes data/clean/genres.csv
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from artist_utils import cell_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SONGS_PATH = PROJECT_ROOT / "data" / "clean" / "billboard_songs.csv"
MB_SONGS = PROJECT_ROOT / "data" / "raw" / "mb_songs.csv"
MB_ARTISTS = PROJECT_ROOT / "data" / "raw" / "mb_artists.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "clean" / "genres.csv"

OTHER = "other"
BROAD_GENRES = [
    "hip-hop/rap",
    "pop",
    "R&B",
    "country",
    "rock/alternative",
    "Latin",
    "dance/electronic",
    OTHER,
]

# Column name for each broad genre's one-hot flag. A song gets a 1 for every
# broad genre its labels touch, not just its primary one, so a hip-hop/pop
# crossover counts as both.
GENRE_COLUMNS = {
    "hip-hop/rap": "is_hiphop",
    "pop": "is_pop",
    "R&B": "is_rnb",
    "country": "is_country",
    "rock/alternative": "is_rock",
    "Latin": "is_latin",
    "dance/electronic": "is_dance",
    OTHER: "is_other",
}

# What an unresolved tie is called in the output. "tie" describes the rule that
# produced it; "mixed" describes the song, which reads better on a chart.
MIXED = "mixed"

# "Latin" here means Spanish- or Portuguese-language Latin American music. That
# is a language/region definition, not a sound: Brazilian funk (funk carioca,
# funk ostentação) is Latin, while reggae and dancehall are Jamaican and
# English-language, so they stay in "other" however Latin they may sound.
#
# Ordered rules: the FIRST rule that matches a label wins, so the order encodes
# precedence. Two ideas drive it:
#   * Latin and country are family markers that beat the rest, so "latin pop"
#     is Latin and "country pop" is country.
#   * Otherwise the last word of a compound label is its head noun, which is
#     what the label really is: "pop rap" is rap, "dance-pop" is pop,
#     "pop rock" is rock. The "ends with pop" rule captures that for the many
#     "<something>-pop" fusions.
GENRE_RULES = [
    # (broad genre, regex matched against the lowercased label)
    (OTHER, r"\bdancehall\b"),  # Jamaican, not dance music; must precede "dance"
    ("Latin", r"\blatin|\breggaeton\b|\bregueton\b|\bsalsa\b|\bbachata\b|\bcumbia\b"
              r"|\bcorrido|\bbanda\b|\bmariachi\b|\btejano\b|\bmerengue\b|\bdembow\b"
              r"|\bnorte(n|ñ)o\b|\branchera\b|\bmexican\b|\bbolero\b|\bflamenco\b"
              r"|\bsamba\b|\bbossa\b|\bbrazil|\bbrasil|\bcarioca\b|\bostenta"
              r"|\bbrega\b|\bsertanejo\b|\bespa(n|ñ)ol|\bportugu"),
    ("country", r"\bcountry\b|\bbluegrass\b|\bamericana\b|\bhonky\b|\bnashville\b"
                r"|\burban cowboy\b"),
    ("pop", r"pop$"),  # dance-pop, electropop, synth-pop, indie pop, k-pop, art pop
    # Soul fusions are R&B first: "trap soul" and "hip hop soul" are R&B styles
    # that borrow rap production, so this rule precedes the rap rule.
    ("R&B", r"\bsoul\b|\br&b\b|\brnb\b"),
    ("hip-hop/rap", r"\bhip[\s-]?hop\b|\brap\b|\btrap\b|\bdrill\b|\bgrime\b"
                    r"|\bcrunk\b|\bboom bap\b|\bbounce\b|\bhyphy\b|\bdirty south\b"
                    r"|\brage\b|\bplugg\b|\bhorrorcore\b|\bratchet\b"),
    ("R&B", r"\bfunk|\bmotown\b|\bnew jack swing\b|\bquiet storm\b|\bdoo[\s-]?wop\b"),
    ("rock/alternative", r"\brock\b|\bpunk\b|\bmetal\b|\bgrunge\b|\bindie\b"
                         r"|\balternative\b|\bemo\b|\bgaze\b|\bbritpop\b"
                         r"|\bnew wave\b|\bpsychedelia\b|\bblackgaze\b"),
    ("dance/electronic", r"\belectronic|\belectro\b|\belectro[\s-]|\bedm\b|\bhouse\b"
                         r"|\btechno\b|\btrance\b|\bdubstep\b|\bdance\b|\bdisco\b"
                         r"|\bclub\b|\brave\b|\bdrum and bass\b|\bdnb\b|\bfuture bass\b"
                         r"|\bsynthwave\b|\beurodance\b|\bhardstyle\b|\bbass\b"
                         r"|\bgarage\b|\btrip hop\b|\bcomplextro\b|\bdowntempo\b"
                         r"|\bleftfield\b|\bindietronica\b|\bjungle\b|\bambient\b"
                         r"|\btronica\b|\bbreakbeat\b|\bmoombahton\b|\bbrostep\b"
                         # \bhardcore\b is safe here: the rap and rock rules run
                         # first, so "hardcore hip hop" and "hardcore punk" are
                         # already claimed, leaving happy/uk hardcore.
                         r"|\bebm\b|\bhardcore\b"),
    ("pop", r"\bpop\b"),  # anything still mentioning pop
]

_COMPILED = [(genre, re.compile(pattern)) for genre, pattern in GENRE_RULES]


def map_label(label: str) -> str:
    """Map one MusicBrainz label to a broad genre."""
    text = cell_text(label).lower()
    for genre, pattern in _COMPILED:
        if pattern.search(text):
            return genre
    return OTHER


def split_labels(value) -> list[str]:
    """Pipe-separated labels, most-voted first (that ordering is the weight)."""
    text = cell_text(value)
    if not text:
        return []
    return [p.strip() for p in text.split("|") if p.strip()]


TIE = "tie"


def score_labels(labels: list[str], counts: list[int] | None) -> tuple[Counter, dict]:
    """Total weight per broad genre, plus where each first appeared.

    Weight is the MusicBrainz vote count when we have it. Rows fetched before
    counts were stored fall back to rank (first label scores n, next n-1...).
    Rank is a weaker proxy: MusicBrainz returns genres alphabetically and script
    03 re-sorts them by count, so equal counts stay in alphabetical order and
    rank then measures nothing but the alphabet.
    """
    if not counts or len(counts) != len(labels):
        n = len(labels)
        counts = [n - i for i in range(n)]

    scores, first_seen = Counter(), {}
    for i, (label, weight) in enumerate(zip(labels, counts)):
        genre = map_label(label)
        scores[genre] += weight
        first_seen.setdefault(genre, i)
    return scores, first_seen


def weigh(labels, counts=None, artist_labels=None, artist_counts=None):
    """Pick the primary broad genre, in three steps, and say which step decided.

    1. "counts"  -- highest total vote count wins outright.
    2. "artist"  -- still tied, so the primary artist's own genre counts break
                    it. Only used when the song had its own genres; when the
                    genres already came from the artist there is nothing
                    independent to break the tie with.
    3. "tie"     -- genuinely undecidable, so primary_genre is "tie" and the
                    tied genres stay in multi_genre rather than being resolved
                    by an arbitrary rule such as alphabetical order.

    Returns (primary_genre, multi_genre, step).
    """
    if not labels:
        return None, "", "none"

    scores, first_seen = score_labels(labels, counts)
    ordered = sorted(scores, key=lambda g: (-scores[g], first_seen[g]))
    multi = "|".join(ordered)

    top = max(scores.values())
    leaders = [g for g in ordered if scores[g] == top]
    if len(leaders) == 1:
        return leaders[0], multi, "counts"

    if artist_labels:
        artist_scores, _ = score_labels(artist_labels, artist_counts)
        best = max(artist_scores.get(g, 0) for g in leaders)
        contenders = [g for g in leaders if artist_scores.get(g, 0) == best]
        if best > 0 and len(contenders) == 1:
            return contenders[0], multi, "artist"

    return TIE, multi, "tie"


def load_inputs():
    songs = pd.read_csv(SONGS_PATH)
    mb_songs = pd.read_csv(MB_SONGS, on_bad_lines="skip")
    mb_artists = pd.read_csv(MB_ARTISTS, on_bad_lines="skip")
    return songs, mb_songs, mb_artists


def parse_counts(value) -> list[int]:
    """Vote counts aligned with the genre names, empty for pre-backfill rows."""
    out = []
    for part in split_labels(value):
        try:
            out.append(int(part))
        except ValueError:
            return []
    return out


def build_genres(songs, mb_songs, mb_artists) -> pd.DataFrame:
    artist_genres = dict(zip(mb_artists["artist_mbid"], mb_artists["genres"]))
    artist_counts = dict(zip(mb_artists["artist_mbid"], mb_artists.get("genre_counts", "")))

    mb = mb_songs.set_index(["title", "artist"])
    rows = []
    for song in songs.itertuples():
        key = (song.title, song.artist)
        labels, counts, source = [], [], "none"

        artist_labels, artist_count_list = [], []

        if key in mb.index:
            record = mb.loc[key]
            if isinstance(record, pd.DataFrame):  # duplicate credit, take the first
                record = record.iloc[0]

            # The primary credited artist is first in the list.
            mbids = split_labels(record.get("artist_mbids"))
            if mbids:
                artist_labels = split_labels(artist_genres.get(mbids[0]))
                artist_count_list = parse_counts(artist_counts.get(mbids[0]))

            labels = split_labels(record.get("genres"))
            counts = parse_counts(record.get("genre_counts"))
            if labels:
                source = "song"
            elif artist_labels:
                labels, counts, source = artist_labels, artist_count_list, "artist"

        # The artist only breaks ties for song-level genres; if the genres came
        # from the artist, breaking the tie with the same data proves nothing.
        primary, multi, step = weigh(
            labels,
            counts,
            artist_labels if source == "song" else None,
            artist_count_list if source == "song" else None,
        )
        rows.append(
            {
                "title": song.title,
                "artist": song.artist,
                "primary_genre": primary,
                "multi_genre": multi,
                "n_broad_genres": len(multi.split("|")) if multi else 0,
                "genre_source": source,
                "genre_rule_step": step,
                "mb_labels": "|".join(labels),
                # Whether MusicBrainz has been queried for this song at all.
                # Without it, a song not yet fetched looks the same as a song
                # MusicBrainz simply has no genre for.
                "in_musicbrainz": key in mb.index,
                "debut_year": song.debut_year,
                "top10": song.top10,
            }
        )

    df = pd.DataFrame(rows)

    # One-hot per broad genre, from every genre the song touches.
    touched = df["multi_genre"].fillna("").apply(lambda v: set(v.split("|")) - {""})
    for genre, column in GENRE_COLUMNS.items():
        df[column] = touched.apply(lambda s, g=genre: g in s)

    # An unresolved tie keeps its own category, just under a clearer name.
    df["primary_genre"] = df["primary_genre"].replace({TIE: MIXED})
    return df


def print_mapping_table(mb_songs, mb_artists):
    """Every distinct label seen so far, grouped by the broad genre it maps to."""
    counts = Counter()
    for series in (mb_songs["genres"], mb_artists["genres"]):
        for value in series.dropna():
            for label in split_labels(value):
                counts[label.lower()] += 1

    grouped = defaultdict(list)
    for label, count in counts.items():
        grouped[map_label(label)].append((label, count))

    print(f"\n{'=' * 78}\nPROPOSED MAPPING  ({len(counts)} distinct labels)\n{'=' * 78}")
    for genre in BROAD_GENRES:
        entries = sorted(grouped.get(genre, []), key=lambda kv: -kv[1])
        total = sum(c for _, c in entries)
        print(f"\n{genre}  --  {len(entries)} labels, {total} mentions")
        line = "   "
        for label, count in entries[:28]:
            piece = f"{label} ({count})  "
            if len(line) + len(piece) > 96:
                print(line)
                line = "   "
            line += piece
        print(line)
        if len(entries) > 28:
            print(f"    ... and {len(entries) - 28} rarer labels")


def print_report(genres: pd.DataFrame):
    total = len(genres)
    covered = genres["primary_genre"].notna()
    fetched = genres["in_musicbrainz"]
    print(f"\n{'=' * 78}\nCOVERAGE\n{'=' * 78}")
    print(f"Songs fetched from MusicBrainz so far: {fetched.sum()}/{total} "
          f"({fetched.mean() * 100:.1f}%)")
    print(f"Songs with a genre, of all songs:     {covered.mean() * 100:.1f}% "
          f"({covered.sum()}/{total})")
    if fetched.sum():
        print(f"Songs with a genre, of those fetched: "
              f"{covered[fetched].mean() * 100:.1f}% ({covered[fetched].sum()}/{fetched.sum()})"
              "   <- the real genre coverage")
    print("\ngenre_source:")
    print(genres["genre_source"].value_counts().to_string())

    # Which step of the primary-genre rule decided each song.
    print("\nprimary-genre rule, songs decided at each step:")
    labels = {
        "counts": "1. vote counts alone",
        "artist": "2. tied, broken by primary artist's counts",
        "tie": "3. still tied -> primary_genre = 'tie'",
        "none": "   (no genre at all)",
    }
    steps = genres["genre_rule_step"].value_counts()
    decided = int(steps.get("counts", 0) + steps.get("artist", 0) + steps.get("tie", 0))
    for key in ("counts", "artist", "tie", "none"):
        count = int(steps.get(key, 0))
        share = f"{count / decided * 100:5.1f}%" if decided and key != "none" else "     -"
        print(f"  {labels[key]:44} {count:5}  {share}")

    ties = genres[genres["genre_rule_step"] == "tie"]
    if len(ties):
        print("\n  most common tied genre sets:")
        for combo, count in ties["multi_genre"].value_counts().head(5).items():
            print(f"    {combo:52} {count}")

    print("\nprimary_genre:")
    counts = genres["primary_genre"].value_counts()
    for genre, count in counts.items():
        print(f"  {genre:20} {count:5}  {count / covered.sum() * 100:5.1f}%")

    multi = genres[genres["n_broad_genres"] > 1]
    print(f"\nSongs touching more than one broad genre: "
          f"{len(multi) / max(covered.sum(), 1) * 100:.1f}% of covered songs")

    print(f"\n{'=' * 78}\nBY DEBUT YEAR (coverage drop check)\n{'=' * 78}")
    by_year = genres.groupby("debut_year").agg(
        songs=("title", "size"),
        fetched=("in_musicbrainz", "sum"),
        covered=("primary_genre", lambda s: s.notna().sum()),
    )
    # Coverage is measured against fetched songs, so a year still being
    # downloaded does not look like a year without genres.
    by_year["coverage_of_fetched_%"] = [
        round(c / f * 100, 1) if f else None
        for c, f in zip(by_year["covered"], by_year["fetched"])
    ]
    print(by_year.to_string())

    print("\nPrimary genre share by debut year (% of covered songs):")
    share = pd.crosstab(genres["debut_year"], genres["primary_genre"], normalize="index") * 100
    print(share.round(1).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Show the mapping table and coverage without writing output.")
    args = parser.parse_args()

    songs, mb_songs, mb_artists = load_inputs()
    print(f"billboard songs: {len(songs)} | mb_songs rows: {len(mb_songs)} | "
          f"mb_artists rows: {len(mb_artists)}")

    print_mapping_table(mb_songs, mb_artists)

    genres = build_genres(songs, mb_songs, mb_artists)
    print_report(genres)

    if args.preview:
        print("\n[preview] nothing written. Re-run without --preview to save "
              f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    genres.drop(columns=["debut_year", "top10", "in_musicbrainz",
                         "cutoff_used", "length_cutoff"],
                errors="ignore").to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(genres)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
