"""New debut-week features for the extension (on top of the team's 13).

Rule for every feature here: it must be computable from information that exists in
the song's debut week, without using the song's own debut position or anything that
happens after the debut chart. Artist history uses only chart weeks *strictly before*
the debut week, from the full 1958-2026 archive, so a song never sees itself or
anything later. Same-week counts (drop size, number of new entries) come from the
debut chart itself, the same convention the team uses for `album_debut_bucket`.

Each feature, and why it is legitimate, is listed in FEATURE_NOTES (also in the README).
"""
import numpy as np
import pandas as pd

FEATURE_NOTES = {
    # history of the lead artist, from chart weeks before the debut week
    'prior_chart_weeks': "Lead artist's total Hot 100 chart weeks (all songs) before the debut week.",
    'artist_weeks_last_52': "Lead artist's chart weeks in the 52 weeks before the debut week (current momentum).",
    'artist_songs_prev_week': "How many of the lead artist's songs were on the chart the week before the debut.",
    'prior_best_peak': "Best (lowest) position the lead artist had reached before the debut week; 101 if never charted.",
    'weeks_since_last_chart': "Weeks since the lead artist was last on the chart (capped at 520 = 10 years; 520 if never).",
    'weeks_since_last_top10': "Weeks since the lead artist last had a song in the Top 10 (capped at 520; 520 if never).",
    'prior_top10_rate': "Lead artist's earlier hits / earlier entries, shrunk toward the training base rate "
                        "(adds 1 pseudo-hit in 10 pseudo-entries) so one-song artists don't get 0% or 100%.",
    'feat_prior_top10s': "Most earlier Top 10s (reached before debut) held by any credited artist other than the "
                         "lead, e.g. a superstar feature. Same 'hit before debut' rule as the team's count.",
    # release facts
    'same_artist_debuts': "Raw count of songs the lead artist debuted that week (the team bins it into 4 groups).",
    'week_new_entries': "Total songs debuting on the Hot 100 that week, across all artists (competition for slots).",
    'release_format': "MusicBrainz release type of the song's first release (single / album / EP / other / unknown). "
                      "In the team's cleaned table but not used in their model.",
    'lead_country': "Lead artist's MusicBrainz country, grouped as US / non-US / unknown. Also in the team's table.",
}


def _artist_weeks(weekly, artist_songs):
    """One row per (artist_name, chart_date, position, title, artist) for every credited artist."""
    keys = artist_songs[['title', 'artist', 'artist_name']]
    return weekly.merge(keys, on=['title', 'artist'], how='inner')


def add_features(songs, extras):
    """Return `songs` with the extension's new columns added (same rows, same order)."""
    weekly = extras['weekly']
    artist_songs = extras['artist_songs']
    wide = extras['wide']
    out = songs.copy()
    key = ['title', 'artist']

    aw = _artist_weeks(weekly, artist_songs)
    aw = aw[['artist_name', 'chart_date', 'position', 'title', 'artist']].rename(
        columns={'title': 'w_title', 'artist': 'w_artist'})

    # ---- lead-artist history from chart weeks strictly before the debut week
    base = out[key + ['primary_artist', 'debut_date']]
    pairs = base.merge(aw, left_on='primary_artist', right_on='artist_name', how='left')
    before = pairs[pairs['chart_date'] < pairs['debut_date']].copy()
    before['weeks_back'] = (before['debut_date'] - before['chart_date']).dt.days / 7
    before['in_top10'] = before['position'] <= 10

    # One row per chart week per artist (two songs in one week count as two chart weeks,
    # which is what "chart weeks" means on Billboard).
    g = before.groupby(key)
    hist = pd.DataFrame({
        'prior_chart_weeks': g.size(),
        'prior_best_peak': g['position'].min(),
        'weeks_since_last_chart': g['weeks_back'].min(),
        'artist_weeks_last_52': g['weeks_back'].apply(lambda s: int((s <= 52).sum())),
        'artist_songs_prev_week': g['weeks_back'].apply(lambda s: int((s.round() == 1).sum())),
    })
    t10 = before[before['in_top10']].groupby(key)['weeks_back'].min().rename('weeks_since_last_top10')
    hist = hist.join(t10).reset_index()
    out = out.merge(hist, on=key, how='left', validate='one_to_one')
    out['prior_chart_weeks'] = out['prior_chart_weeks'].fillna(0)
    out['artist_weeks_last_52'] = out['artist_weeks_last_52'].fillna(0)
    out['artist_songs_prev_week'] = out['artist_songs_prev_week'].fillna(0)
    out['prior_best_peak'] = out['prior_best_peak'].fillna(101)
    out['weeks_since_last_chart'] = out['weeks_since_last_chart'].fillna(520).clip(upper=520)
    out['weeks_since_last_top10'] = out['weeks_since_last_top10'].fillna(520).clip(upper=520)

    # Shrunk hit rate. 0.1 is roughly the training base rate (10.3%); it is a fixed
    # constant, not estimated from the test years.
    out['prior_top10_rate'] = (out['prior_top10s'] + 1) / (out['prior_entries'] + 10)

    # ---- featured / co-credited artists: best "hit before debut" count among non-lead artists
    a = artist_songs[['title', 'artist', 'debut_date', 'first_top10_date', 'artist_name']]
    credited = out[key + ['primary_artist', 'debut_date']].merge(
        artist_songs[['title', 'artist', 'artist_name']], on=key, how='left')
    credited = credited[credited['artist_name'] != credited['primary_artist']]
    cp = credited.merge(a.rename(columns={'title': 'e_title', 'artist': 'e_artist', 'debut_date': 'e_debut'}),
                        on='artist_name', how='inner')
    cp = cp[(cp['e_debut'] < cp['debut_date']) & (cp['first_top10_date'] < cp['debut_date'])]
    per_artist = cp.groupby(key + ['artist_name']).size().rename('n').reset_index()
    feat = per_artist.groupby(key)['n'].max().rename('feat_prior_top10s').reset_index()
    out = out.merge(feat, on=key, how='left', validate='one_to_one')
    out['feat_prior_top10s'] = out['feat_prior_top10s'].fillna(0)

    # ---- same-week release facts
    w = wide[key + ['same_artist_debuts']]
    out = out.merge(w, on=key, how='left', validate='one_to_one')
    debuts_all = weekly.sort_values('chart_date').groupby(key)['chart_date'].min()
    per_week = debuts_all.value_counts()
    out['week_new_entries'] = out['debut_date'].map(per_week).astype(float)

    # ---- MusicBrainz facts already in the team's table
    fmt = out['release_format'].fillna('unknown').str.lower()
    out['release_format'] = np.where(fmt.isin(['single', 'album', 'ep', 'unknown']), fmt, 'other')
    out['lead_country'] = np.where(out['artist_country'].isna(), 'unknown',
                                   np.where(out['artist_country'] == 'US', 'US', 'non-US'))
    return out


if __name__ == '__main__':
    from build_dataset import build
    songs, extras = build()
    df = add_features(songs, extras)
    cols = list(FEATURE_NOTES)
    print(df[cols].describe(include='all').T.to_string())
    tr = df[df['model_split'] == 'train']
    print('\nTop 10 rate by release_format (train):')
    print(tr.groupby('release_format')['top10'].agg(['size', 'mean']).round(3))
