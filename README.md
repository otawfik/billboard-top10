# Billboard: Top 10 or Not?

What separates a Billboard Hot 100 song that reaches the Top 10 from one that
peaks lower? We built a dataset of **5,342 songs** that entered the chart between
January 2018 and September 2026, joined the weekly chart archive to lyrics and
release metadata, and tested ten hypotheses about what predicts a Top 10 hit.
**519 songs (9.7%)** reached it. The short answer: *who* released a song and
*how* it was released matter far more than anything measurable in its lyrics.

**Analysis:** [`B02-Billboard-Top-10-Or-Not.ipynb`](B02-Billboard-Top-10-Or-Not.ipynb)
· **Column reference:** [`docs/data_dictionary.md`](docs/data_dictionary.md)

## Key findings

Tested on the 2018-2024 training years (n = 4,293); 2025-2026 is held out.

| Finding | Top 10 rate | Test |
|---|---|---|
| **Artist history is the strongest predictor.** Past hits roughly triple the odds of the next | 5.8% (no prior Top 10s) → 18.5% (10+) | p ≈ 3×10⁻²³, V = 0.159 |
| **Album drops are U-shaped, not linear.** Filler tracks compete with each other; only superstars release 10+ at once | 11.6% alone → **2.1%** at 5-9 → 11.8% at 10+ | p ≈ 1×10⁻⁹ |
| **Genre shifts the odds, modestly.** Pop over-performs, country under-performs | pop 18.7%, country 5.3% | p ≈ 7×10⁻¹⁶, V = 0.146 |
| **May and November are not a hit season.** The peaks were superstar album drops (Drake, Taylor Swift, Morgan Wallen) | effect vanishes without 5+ song drops | p 0.009 → 0.906 |

**Null results worth reporting.** Lyric style barely separates the groups: only
vocabulary diversity differs (0.348 vs 0.361), while word count, repetition and
line length do not. Featured artists (9.6% vs 10.4%) and profanity (10.1% vs
10.5%) show no effect.

**A leak we caught.** Debut position looked perfectly predictive, but a song
debuting inside the Top 10 has already met the definition of the target. Question
10 in the notebook shows the corrected result.

## Data sources

| Source | Used for | Licence |
|---|---|---|
| [Billboard Hot 100 archive](https://github.com/mhollingshead/billboard-hot-100) (mhollingshead) | every weekly chart, 1958-2026 | **None declared** by the repository; the chart is Billboard's copyrighted compilation |
| [LRCLIB](https://lrclib.net/docs) | lyrics | **None stated** on the public docs page; lyrics belong to their rights holders |
| [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_API) | release dates, genres, artist metadata | Core data **[CC0](https://musicbrainz.org/doc/About/Data_License)** |

**Why `data/` is not in this repo.** Two of the three sources declare no licence,
and the dataset contains full song lyrics. We use them for non-commercial academic
analysis only and do not redistribute them. Rebuild the data locally (below), or
get it from the team Drive.

## Repository structure

Scripts run in this order; each reads the previous step's output.

| Script | What it does |
|---|---|
| [`01_billboard.py`](scripts/01_billboard.py) | Downloads the chart archive; builds one row per song debuting 2018+; merges songs Billboard re-credited mid-run |
| [`02_lyrics.py`](scripts/02_lyrics.py) | Fetches lyrics from LRCLIB with fuzzy matching and a confidence score |
| [`03_musicbrainz.py`](scripts/03_musicbrainz.py) | Fetches release dates, genres (with vote counts) and artist data, at 1 request/second |
| [`04_artist_history.py`](scripts/04_artist_history.py) | Counts each artist's prior chart record, using only songs that debuted earlier; includes leakage checks |
| [`05_genres.py`](scripts/05_genres.py) | Maps hundreds of MusicBrainz labels to 8 broad genres |
| [`06_lyrics_quality.py`](scripts/06_lyrics_quality.py) | Flags lyrics that look truncated; repairs unusable durations |
| [`07_merge_all.py`](scripts/07_merge_all.py) | Joins everything into one row per song |
| [`08_features.py`](scripts/08_features.py) | Cleans lyric text and builds the analysis features and train/test split |

Shared helpers: [`artist_utils.py`](scripts/artist_utils.py) (credit splitting,
text normalisation) and [`song_utils.py`](scripts/song_utils.py) (duplicate-credit
merging).

## How to reproduce

**Setup** (Python 3.14):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option A — rebuild from scratch (~5 hours).** Almost all of it is API time: the
MusicBrainz limit of 1 request/second dominates. Every fetch saves progress and
resumes if interrupted.

```bash
for s in 01_billboard 02_lyrics 03_musicbrainz 04_artist_history \
         05_genres 06_lyrics_quality 07_merge_all 08_features; do
  python scripts/$s.py
done
```

**Option B — use the team's copy (minutes).** Download `data/clean/` from the
team Drive into this folder, then open the notebook. It needs only
`data/clean/songs_analysis.csv` and makes no API calls.

```bash
jupyter notebook B02-Billboard-Top-10-Or-Not.ipynb
```

## Team

BA780 · Team B02 *(team ID to be confirmed)*

- *(member name)*
- *(member name)*
- *(member name)*

## AI assistance

The data pipeline, cleaning and notebook were built with AI assistance
(Claude Code). See the AI usage disclosure in the notebook for how it was used
and how the results were checked.
