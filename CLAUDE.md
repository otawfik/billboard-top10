# Billboard Top 10 project (2018–Sep 2026)

## Goal
Compare Billboard Hot 100 songs that reached the Top 10 against songs that peaked lower. Look at how hits changed over time and whether lyrics, genre, and the artist's past success predict reaching the Top 10.

## Data sources
- Billboard Hot 100 weekly charts: github.com/mhollingshead/billboard-hot-100 (full 1958–2026 JSON archive; filter to 2018 onward)
- LRCLIB lyrics API: lrclib.net/docs (one row per song)
- MusicBrainz API: musicbrainz.org/doc/MusicBrainz_API (release dates, genres, artist country and type)
  - Max 1 request per second
  - Must send a descriptive User-Agent, e.g. "BillboardHitsProject/1.0 (student-email@example.com)"

## Folder layout
- scripts/ : all Python scripts
- data/raw/ : downloaded and API data, never edited by hand
- data/clean/ : merged and cleaned outputs

## Rules
- Python and pandas
- API pulls save progress to CSV after every batch and resume where they left off if rerun
- Normalize titles and artists before matching (lowercase, strip punctuation, remove "feat."/"featuring" and parentheticals)
- Record a match-confidence column for every lyrics and metadata match
- Comment code clearly so every team member can explain it
- Show a plan before writing large scripts
