# Analysis files

Two files, same 5,342 rows and same `(title, artist)` key:

- **`songs_analysis.csv`** — use this one. Everything below, plus cleaned
  lyrics and the derived features in the last section. 68 columns.
- **`songs_merged.csv`** — the join without the derived features. 55 columns.

---

# songs_merged.csv

One row per song that entered the Billboard Hot 100 between 2018-01-01 and
September 2026. **5,342 rows, 55 columns.** Built by `scripts/07_merge_all.py`,
which left-joins every other table onto `billboard_songs.csv`, so a missing
source shows as an empty cell and never drops a song.

The key is `(title, artist)`. `artist` is always the credit the song **debuted**
under — see "Why the debut credit" below.

## Identity

| column | meaning |
|---|---|
| `title` | Billboard song title, exactly as printed |
| `artist` | credit from the song's **debut week** — the join key |
| `final_artist` | credit at the song's last chart week, if Billboard changed it |
| `merged_from` | the source rows a merged song came from, `title :: artist` separated by `\|`; empty for the 99% of songs that were never merged |

## Chart performance — the target lives here

| column | meaning |
|---|---|
| `top10` | **the target.** True if the song ever reached position 1-10 |
| `peak_position` | best position reached, 1 = number one |
| `weeks_on_chart` | count of distinct weeks on the chart |
| `debut_position` | position in its first chart week |
| `first_chart_date` / `last_chart_date` | first and last week on the chart |
| `debut_year` | year of `first_chart_date` |
| `still_charting` | True if the song is on the most recent chart in the archive, so its peak may not be final |

## Artist history — what was true AT DEBUT

Every column here counts only songs that debuted **strictly before** this song,
so none of it leaks information from the song's own chart run. Verified by three
checks in `scripts/04_artist_history.py`.

| column | meaning |
|---|---|
| `primary_artist` | lead artist, first name in the credit |
| `all_credited_artists` | every artist in the credit, `\|` separated |
| `n_credited_artists` | how many artists are credited |
| `has_feature` | True if the credit says "Featuring" / "feat." |
| `prior_entries` | chart entries the primary artist had before this debut |
| `prior_top10s` | of those, how many reached the Top 10 |
| `prior_number1s` | of those, how many reached number one |
| `best_prior_top10s` | highest `prior_top10s` across **all** credited artists, so a newcomer featuring a star scores high |
| `weeks_since_last_entry` | weeks between this debut and the artist's previous debut; **empty for first-timers**, which is why it is ~10% missing |
| `is_first_entry` | True if the primary artist had never charted before |

## Genre

| column | meaning |
|---|---|
| `primary_genre` | one of hip-hop/rap, pop, R&B, country, rock/alternative, Latin, dance/electronic, other, or **`mixed`** when two genres tie and the artist cannot break it |
| `multi_genre` | every broad genre the song touches, strongest first, `\|` separated |
| `n_broad_genres` | how many broad genres it touches |
| `is_hiphop`, `is_pop`, `is_rnb`, `is_country`, `is_rock`, `is_latin`, `is_dance`, `is_other` | one-hot flags from `multi_genre`. A song can be True for several — use these for modelling, `primary_genre` for charts |
| `genre_source` | `song` (the recording's own MusicBrainz genres), `artist` (fell back to the primary artist's), or `none` |
| `genre_rule_step` | how `primary_genre` was decided: `counts`, `artist` tie-break, or `tie` |
| `mb_labels` | the raw MusicBrainz labels behind it, strongest first |

## Lyrics

| column | meaning |
|---|---|
| `plain_lyrics` | full lyrics text from LRCLIB |
| `lyrics_chars` | character count |
| `chars_per_sec` | `lyrics_chars` / `duration_used`; empty when the duration is unusable |
| `lyrics_suspect_short` | **True if the lyrics look truncated** for the song's length. Keep or exclude, but do not treat as clean text without checking |
| `instrumental` | LRCLIB says the track has no vocals |
| `lyrics_match_score` | 0-100 confidence the lyrics belong to this song; 72+ is trusted |
| `lyrics_match_method` | which search found it: `full`, `primary_artist`, `title_only`, or `none` |

## MusicBrainz

| column | meaning |
|---|---|
| `recording_mbid` | MusicBrainz recording id |
| `first_release_date` | earliest known release date of the recording |
| `release_type` | Single, Album, EP … |
| `recording_length_ms` | track length. **Only fetched where LRCLIB's duration was unusable**, hence ~57% empty by design |
| `mb_genres` / `mb_genre_counts` | raw genre labels and their vote counts, aligned and strongest first |
| `artist_mbids` | MusicBrainz ids for every credited artist |
| `mb_match_score` / `mb_match_method` | same meaning as the lyrics pair |

## Duration

`duration_used` is the duration everything else is computed from.
`duration_source` says where it came from: `lrclib`, `musicbrainz` (used when
LRCLIB's was an album upload or a 0-2 second placeholder), or `none`.
`duration_reliable` is True when it falls in a plausible 30-600 second range.

## Why the debut credit

Billboard sometimes changes a credit mid-run: "Karma" entered as Taylor Swift
and later became "Taylor Swift Featuring Ice Spice". Those rows were merged into
one song, and the **debut** credit was kept throughout — including for the
lyrics, which are taken from the debut credit's match wherever that match is
good. Ice Spice's verse did not exist when the song charted, so using it would
feed post-debut information into features meant to describe the song at entry.
Same rule as the artist-history columns.

---

# songs_analysis.csv — the derived columns

Everything above, plus the following. Built by `scripts/08_features.py`.

## Modelling

| column | meaning |
|---|---|
| `in_model` | **False for the 99 songs still on the chart**, whose peak is not final and so cannot be used as a training label. They stay in the file for descriptive charts — filter on this before fitting anything |

## Cleaned lyrics

`lyrics_clean` is `plain_lyrics` with section labels (`[Chorus]`,
`[Verse 2: Drake]`), LRC timestamps and LRC metadata tags removed, whitespace
normalized, and `(x2)`-style repeat markers expanded by repeating the line
rather than dropping it. The raw column is kept so any of this can be redone.

Repeat-marker expansion is a no-op on the current data — **no song in the set
actually uses one** — but it is there so the cleaning does not silently
under-count if a future fetch includes them.

| column | meaning |
|---|---|
| `word_count` | words in the cleaned lyrics |
| `unique_word_ratio` | distinct words / total words; **lower means more repetitive** |
| `repetition_score` | share of lines that appear more than once |
| `avg_line_length` | average words per line |
| `has_profanity` | cleaned lyrics contain strong profanity |

**These five are blank, never zero, for the 32 songs with no lyrics** (31
unmatched, 1 instrumental). A song with no lyrics did not score zero words — we
do not know its value, and a zero would be a real measurement that drags every
average down. Filter with `.notna()`, do not `fillna(0)`.

The profanity list covers strong terms and their asterisk-censored forms, and
deliberately excludes mild words like "damn" and "hell", which are common
enough to flag most songs and say little. The list is at the top of
`scripts/08_features.py` and is meant to be edited.

## Timing and season

| column | meaning |
|---|---|
| `days_release_to_chart` | days from `first_release_date` to `first_chart_date`. **Blank for the 420 songs whose release date is only a year or a year-month** — dating those would mean inventing a day. Can be negative when MusicBrainz's earliest release is a later reissue |
| `debut_month` | calendar month of the chart debut, 1-12 |
| `dec_jan_share` | share of the song's chart weeks falling in December or January |
| `dec_jan_seasons` | how many separate Dec/Jan seasons it charted in; a December is paired with the January after it |
| `is_holiday` | seasonal song — see below |
| `holiday_reason` | which half of the rule fired: `title`, `season`, or `title+season` |

### How `is_holiday` works

A song is seasonal if **its title names the season and it charted at least once
in Dec/Jan**, OR if **70%+ of its chart weeks are in Dec/Jan across two or more
separate seasons**. 70 songs qualify.

Both guards matter. Seasonal share alone flagged **475 songs**, because any
album track released in December has a one-week run that is trivially 100%
seasonal — it caught most of Meek Mill's *Championships* and half of
XXXTENTACION's catalogue. Requiring recurrence across two seasons is what
separates a perennial from an album track. And a title keyword alone caught
"La Santa" (Bad Bunny — *santa* is Spanish for "saint") and "7969 Santa"
(Drake), neither of which charted in the season at all.

Two songs pass every rule but are not seasonal, so they are excluded by name in
`HOLIDAY_EXCEPTIONS` at the top of `scripts/08_features.py`:

| song | artist | why it slipped through |
|---|---|---|
| Santa Fe | Zach Bryan | the city, not Santa Claus; charted 100% in Dec/Jan by coincidence |
| Holiday Road | Kesha | the *National Lampoon's Vacation* song, matched on "holiday" |

Both charted in December, so no timing-based rule can catch them — a name list
is the only option. Add to it rather than loosening the keywords.

## Album releases

When an album drops, every track can enter the chart at once. Those songs
compete with each other and mostly fall straight back out, so they behave
differently from a single released on its own.

| column | meaning |
|---|---|
| `same_artist_debuts_that_week` | songs by the same primary artist entering the chart in that same week, including this one. 1 means it debuted alone |
| `album_debut_bucket` | that count as a band: `1`, `2-4`, `5-9`, `10+` |

Both are leakage-safe: they count only what is visible in the debut week
itself, never a later week.

### The effect is U-shaped, which is why there is no boolean

| songs debuting together | songs | Top 10 rate |
|---|---|---|
| 1 (alone) | 2,574 | **11.0%** |
| 2-4 | 434 | 6.2% |
| 5-9 | 530 | **2.3%** |
| 10+ | 1,804 | **11.0%** |

Two opposite forces are at work. Up to about nine songs, an album drop is
**dilution** — filler tracks chart briefly and fall out, so the rate collapses
to 2.3%. Past ten, the count stops measuring the album and starts measuring the
**artist**: only a Drake or a Taylor Swift can place ten songs at once, and
their songs chart high whatever else is true.

A boolean at any single cutoff averages those two together and cancels them
out. At 5+ it gave 9.0% versus 10.3% — almost no difference, from a variable
that ranges 2.3% to 11.0%. **Use the bucket, or the raw count; do not
dichotomise it.**

## Train/test split

| column | meaning |
|---|---|
| `model_split` | `train` for debut years 2018-2024 (4,293 songs), `test` for 2025-2026 (950), blank where `in_model` is False (99) |

**The split is by time, not at random, and a random split would be wrong here.**
Songs from one album drop enter the chart together, share an artist, and share
a moment — up to 40 of them at once. A random split scatters tracks from the
same release across train and test, so the model can memorise "this Drake album
charted well" and score well on songs it has effectively already seen. Splitting
by year keeps every release wholly on one side.

The blank rows are songs still on the chart, whose peak is not final, so there
is no reliable label to train on or score against.

## Known gaps

- **166 songs (3.1%) have no genre.** Coverage is ~97% for 2018-2025 but 93.7%
  for 2026, because recent releases have had less time to be tagged.
- **54 songs fall below the MusicBrainz match threshold** and carry no
  MusicBrainz data. Mostly initialism titles the search cannot handle
  ("B.I.T.C.H.", "1.5", "M.T.B.T.T.F.").
- **31 songs have no lyrics match**; 1 is a genuine instrumental.
- **96 songs have `primary_genre = "mixed"`** — a real tie, not an error.
- **Duplicate credits are merged only from 2018 onward.** Pre-2018 duplicates
  were measured and deliberately left unmerged: only 4 groups exist, touching
  58 of 5,342 songs (1.1%), 43 of them Justin Bieber. Merging was rejected
  because at least one pair is two genuinely different recordings — "All I Want
  For Christmas Is You" by *Mariah Carey* and by *Justin Bieber Duet With
  Mariah Carey*. So an artist's pre-2018 `prior_entries` may count a
  re-credited song twice, by at most one entry.
