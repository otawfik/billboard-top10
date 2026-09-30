# Morning report

Billboard Top 10 pipeline. **Data collection, cleaning and feature building are
complete.** The analysis file is `data/clean/songs_analysis.csv`
(5,342 rows x 71 columns); `data/clean/README.md` documents every column.

## Headline findings so far

Descriptive only — nothing modelled yet, and all of it from the training years
unless stated.

**1. Genre separates the groups more than anything else measured.**

| primary genre | share of Top 10 | share of the rest |
|---|---|---|
| pop | **35.7%** | 18.4% |
| hip-hop/rap | 38.0% | 46.6% |
| country | 7.6% | 13.6% |

Pop is roughly twice as common among Top 10 songs as among the rest, while
hip-hop/rap and country are both *under*-represented at the top despite
hip-hop/rap being the largest genre overall.

**2. Lyrics look flat.** Almost nothing separates the two groups:

| feature | Top 10 | non-Top 10 |
|---|---|---|
| word count (median) | 427 | 407 |
| unique-word ratio | 0.347 | 0.361 |
| repetition score | 0.423 | 0.418 |
| average line length | 7.85 | 7.86 |
| profanity | 51.3% | 52.5% |

On these measures lyric *style* does not look like a promising predictor.
Content, which none of these capture, may still be.

**3. Album drops have a U-shaped effect, not a linear one.**

| songs debuting together | Top 10 rate |
|---|---|
| 1 (alone) | 11.0% |
| 2-4 | 6.2% |
| 5-9 | **2.3%** |
| 10+ | **11.0%** |

Up to nine songs it is dilution — filler charts briefly and falls out. Past ten
the count stops measuring the album and starts measuring the artist: only a
superstar can place ten songs at once. This is why the feature is a bucket and
not a boolean; a 5+ boolean gave 9.0% vs 10.3%, cancelling a range of 2.3-11.0%.

**4. The May and November "seasons" were album drops.** Excluding releases of
5+ songs, May's Top 10 share falls from 14.5% to 10.0% (against 8.0% for
non-Top 10) and November's from 10.6% to 8.1% (against 7.6%) — both effectively
gone. Three of the four biggest drop weeks were in May: Drake (40 songs,
2026-05-30), Taylor Swift (31, 2024-05-04) and Morgan Wallen (29, 2025-05-31).
**Treat any month effect as an artist-schedule effect until shown otherwise.**

**5. Prior success tracks with reaching the Top 10.** On the training years
only, artists with 5+ previous Top 10s reach it at **16.8%**, against 8.7% for
1-4 and 5.8% for none — a roughly threefold spread, and every input is
leakage-checked.

## Where the data stands

All pipeline steps have run to completion. Nothing is in progress.

| file | rows x cols | state |
|---|---|---|
| `data/clean/songs_merged.csv` | 5,342 x 55 | **the analysis file** |
| `data/clean/billboard_songs.csv` | 5,342 x 12 | duplicate credits merged |
| `data/clean/artist_history.csv` | 5,342 x 12 | leakage checks pass |
| `data/clean/genres.csv` | 5,342 x 16 | includes one-hot columns |
| `data/clean/lyrics_quality.csv` | 5,342 x 13 | 109 rows flagged |
| `data/raw/lyrics.csv` | 5,377 x 11 | complete, 99.4% match rate |
| `data/raw/mb_songs.csv` | 5,349 x 17 | complete, 99.0% match rate |
| `data/raw/mb_artists.csv` | 1,269 x 10 | complete |

The raw files hold slightly more rows than the clean ones: they are fetch
caches and keep rows for credits the duplicate merge collapsed. The clean
outputs are driven by the song table, so this is intentional.

Headline numbers: **5,342 songs, 519 Top 10 (9.7%).**
`data/clean/README.md` describes every column for teammates.

First thing the data says, before any modelling: **pop is 35.7% of Top 10 songs
but 18.4% of the rest**, while hip-hop/rap runs 38.0% against 46.6% and country
7.6% against 13.6%.

## Your decisions, applied

1. **Suspect-short lyrics kept** with `lyrics_suspect_short` for the
   sensitivity check. 109 rows.
2. **One-hot genre columns added** (`is_hiphop`, `is_pop`, `is_rnb`,
   `is_country`, `is_rock`, `is_latin`, `is_dance`, `is_other`), built from
   `multi_genre` so a crossover counts in every genre it touches. `tie` is
   relabelled **`mixed`** in `primary_genre` (96 songs).
3. **Missing genres left as-is**, recorded in `data/clean/README.md`.
4. **The 54 unmatched accepted.**
5. **`vs.` / `Introducing` / `Presents` now split.** See below.
6. **Pre-2018 measured only, nothing merged.** See below.
7. **Genre mapping extended**: `moombahton`, `brostep`, `ebm`, `happy hardcore`
   and `uk hardcore` to dance/electronic; `urban cowboy` to country.
   `hardcore hip hop` and `hardcore punk` still map correctly, because the rap
   and rock rules run before the dance rule.

### Decision 5 — new separators, 12 songs changed

18 archive credits are affected, all split correctly: "Snoop Dogg Presents Tha
Eastsidaz" is now two artists, "Elvis Presley vs JXL" two, "Diplo Presents
Thomas Wesley Featuring Morgan Wallen" three. No false positives — "The
Presidents" and "Mr. President" are untouched.

**12 songs changed** in step 4. Diplo's "Heartless" went from 0 to 3 prior
entries, because its whole credit had been treated as one long artist name;
Shoreline Mafia's "Heat Stick" went from 2 to 3 credited artists; six
Tiesto and Lana Del Rey songs each gained one prior entry.

Step 1's merge set was unchanged at 34 songs, and all three leakage checks
still pass.

### Decision 6 — pre-2018 duplicates, measured

**Only 4 pre-2018 duplicate-credit groups exist** (8 rows, 4 would be removed),
touching 7 artists. **58 of 5,342 songs (1.1%)** credit one of those artists and
could see `prior_entries` shift by at most one each. Justin Bieber accounts for
43 of them.

**I would not merge them as they stand.** One of the four pairs is "All I Want
For Christmas Is You" under *Mariah Carey* and under *Justin Bieber Duet With
Mariah Carey* — two different recordings, not one song re-credited, so merging
would be wrong. The Elvis Presley pair ("Elvis Presley" / "Elvis Presley With
The Jordanaires") does look like a genuine duplicate. Worth eyeballing before
any merge; the gain is small either way.

## What ran overnight

### Step 0 — lyrics now prefer the debut credit

A merged song was fetched once per credit, and the later credit often matches
"better" precisely because it is a later, fuller version — LRCLIB's "Taylor
Swift Featuring Ice Spice" entry carries a verse that did not exist when
"Karma" entered the chart. Taking it would feed post-debut text into features
describing the song at debut.

The debut credit now wins whenever its match is good: at or above the match
threshold, with real lyrics, not instrumental, and not flagged
`lyrics_suspect_short`. Only if the debut match fails one of those does a later
credit get used, decided on a second pass so the flag is known.

Result: **all 34 merged songs use their debut credit.** "Karma" carries the
1,961-character pre-Ice-Spice lyrics rather than the 2,383-character later
version, reversing the 8 switches reported earlier.

### Step 1 — MusicBrainz complete

Part A matched **5,288 of 5,342 songs (99.0%)**; Top 10 98.8%, non-Top 10 99.0%.
Part B fetched **1,269 artists**. Song-level genre coverage is 70.7%, rising to
**98.3% once artist-level genres are allowed**.

The job crashed once on a `ConnectionResetError` that the retry logic did not
catch; both fetchers now catch `OSError`, and the resumed run finished cleanly.

### Step 2 — backfill and retries

Backfill gave genre vote counts to **688 song rows and 176 artist rows**; 69
songs were left without because MusicBrainz no longer lists genres for them.
Every row that has genres now has counts. Recording lengths landed on 2,274
songs, cutting unusable durations from 346 to **66**.

Retries: lyrics re-fetched 63 rows (35 changed, 3 improved); MusicBrainz
re-fetched 150 rows (16 changed, 2 improved). Neither pushed a song across the
match threshold, so match rates are unchanged.

### Step 3 — genres and lyrics quality

**Genre coverage is 96.9%** (5,176 of 5,342): 3,739 song-level, 1,437 from the
artist fallback, 166 with nothing. Coverage by debut year is steady:

| year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| coverage % | 98.0 | 97.9 | 97.2 | 96.8 | 98.1 | 95.9 | 97.2 | 96.7 | **93.7** |

2026 being weakest is expected — recent releases have had less time to be
tagged. Worth re-running later in the year.

The primary-genre rule resolved **91.4% on vote counts alone**, 6.7% on the
artist tie-break, and **1.9% (96 songs) stayed a genuine tie**, now labelled
`mixed`.

Step 6 re-ran on the complete data: **109 rows flagged `lyrics_suspect_short`**,
and 285 songs take their duration from MusicBrainz.

### Step 4 — merged table

`data/clean/songs_merged.csv`: **5,342 rows x 55 columns**, one row per song,
every join a left join from the song table so nothing is dropped.

Coverage is even across both groups, so the target is not confounded by missing
data:

| group | lyrics | genre | musicbrainz |
|---|---|---|---|
| Top 10 | 99.6% | 98.8% | 100.0% |
| non-Top 10 | 99.4% | 96.7% | 99.9% |

Columns with real gaps: `recording_length_ms` 56.6% missing (only fetched where
needed), `mb_genres` 30.0%, `weeks_since_last_entry` 9.7% (first entries have no
previous entry by definition), `primary_genre` 3.1%. `merged_from` is 99.4%
empty by design.

## Known data quirks

- **66 songs still have an unusable duration**, down from 346: the MusicBrainz
  recording length covers 285 of them. The rest have no length in either
  source, so `chars_per_sec` is empty for them.
- **31 songs have no lyrics match**; 1 is a true instrumental (Andre 3000's
  *New Blue Sun*, correctly identified).
- **54 songs fall below the MusicBrainz match threshold**, mostly initialism
  titles ("B.I.T.C.H.", "1.5", "M.T.B.T.T.F."). Accepted per decision 4.
- **166 songs have no genre** (3.1%), 2026 weakest at 93.7%. Accepted per
  decision 3 and recorded in `data/clean/README.md` under Known gaps.
- `data/raw/lyrics.csv` and `mb_songs.csv` keep rows for credits that no longer
  exist as separate songs. Intentional: raw is the fetch cache, clean outputs
  are driven by the song table.
- MusicBrainz returns genres alphabetically, so vote counts are stored and used
  as the weight; rank alone was hiding real ties.

## Open items

Nothing is blocked. Two things you may want to look at:

1. The **Elvis Presley pre-2018 pair** from decision 6, if you want that merge.
2. **Re-running MusicBrainz for 2026 songs** later in the year, once more of
   them have been tagged.
