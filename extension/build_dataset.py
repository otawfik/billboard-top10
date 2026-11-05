"""Rebuild Team A05's cleaned song-level dataset, exactly as the notebook does.

This is the cleaning code from `A05-Billboard-Top-10-Or-Not.ipynb` (Section 3,
Steps 1-12), lifted out of the notebook cell by cell so the extension can import
it. Nothing about the cleaning is re-invented here: the same rename list, the same
artist splitting, the same "hit before debut" history rule, the same genre rules.
Only display-only lines (head(), value_counts() printouts, the missingno chart)
were dropped. Cell numbers refer to the notebook's code cells.

`build()` returns:
    songs     the team's 26 analysis columns (Step 12), one row per song from 2018 on
    extras    a dict of intermediate tables the extension's new features need
              (weekly rows after the credit merge, song-artist rows, the wider
              pre-Step-12 song table). These are the notebook's own objects.

Run directly to print the reproduction checks:
    python extension/build_dataset.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')

BILLBOARD_URL = 'https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json'
LAST_CHART = '2026-09-19'

# ---- Step 3 (cell 26): songs Billboard listed under two credits, checked by hand.
DUPLICATE_CREDITS = [
    ['360', 'Charli xcx', '360', 'Charli XCX'],
    ['Act II: Date @ 8', '4Batz Featuring Drake', 'Act II: Date @ 8', '4Batz'],
    ['Ahi', 'Karol G With Drake', 'Ahi', 'Karol G & Drake'],
    ['Apple', 'Charli xcx', 'Apple', 'Charli XCX'],
    ['BbY WOW', 'Karol G With Judeline & rusowsky', 'BbY WOW', 'Karol G, Judeline & rusowsky'],
    ['Blue Strips', 'jessie murph', 'Blue Strips', 'Jessie Murph'],
    ['Bottle Rockets', 'Scotty McCreery & Hootie & the Blowfish',
     'Bottle Rockets', 'Scotty McCreery & Hootie & The Blowfish'],
    ['Burn', '¥$: Ye & Ty Dolla $ign', 'Burn', '¥$: Kanye West & Ty Dolla $ign'],
    ['Burning Blue', 'Mariah the Scientist', 'Burning Blue', 'Mariah The Scientist'],
    ['Carnival', '¥$: Ye & Ty Dolla $ign Featuring Rich The Kid & Playboi Carti',
     'Carnival', '¥$: Kanye West & Ty Dolla $ign Featuring Rich The Kid & Playboi Carti'],
    ['Dial Drunk', 'Noah Kahan With Post Malone', 'Dial Drunk', 'Noah Kahan'],
    ['Dracula', 'Tame Impala & JENNIE', 'Dracula', 'Tame Impala'],
    ['Ever Since U Left Me (I Went Deaf)', 'French Montana & Max B',
     'Ever Since U Left Me', 'Max B & French Montana'],
    ['Fuk Sumn', '¥$: Ye & Ty Dolla $ign', 'Fuk Sumn', '¥$: Kanye West & Ty Dolla $ign'],
    ['Get In With Me', 'BossMan DLow', 'Get In With Me', 'BossMan Dlow'],
    ['Go Away', 'Weezer Featuring Best Coast', 'Go Away', 'Weezer'],
    ['Golden', 'HUNTR/X: EJAE, Audrey Nuna & REI AMI', 'Golden', 'HUNTR/X, EJAE, Audrey Nuna & Rei Ami'],
    ['Hard Fought Hallelujah', 'Brandon Lake X Jelly Roll', 'Hard Fought Hallelujah', 'Brandon Lake'],
    ['Hell At Night', 'BigXthaPlug Featuring Ella Langley', 'Hell At Night', 'BigXthaPlug & Ella Langley'],
    ["I Ain't Coming Back", 'Morgan Wallen Featuring Post Malone',
     "I Ain't Coming Back", 'Morgan Wallen & Post Malone'],
    ['Is It A Crime', 'Mariah The Scientist & Kali Uchis', 'Is It A Crime', 'Mariah the Scientist & Kali Uchis'],
    ['Jingle Bells', 'Frank Sinatra With Orchestra Conducted By Gordon Jenkins', 'Jingle Bells', 'Frank Sinatra'],
    ['Jingle Bells', 'Frank Sinatra With The Orchestra & Chorus Of Gordon Jenkins', 'Jingle Bells', 'Frank Sinatra'],
    ['Karma', 'Taylor Swift Featuring Ice Spice', 'Karma', 'Taylor Swift'],
    ["Let's Go", 'Key Glock & Young Dolph', "Let's Go", 'Key Glock'],
    ['One More Time', 'blink-182', 'One More Time', 'Blink-182'],
    ['One Of The Girls', 'The Weeknd, Jennie & Lily Rose Depp', 'One Of The Girls', 'The Weeknd, JENNIE & Lily Rose Depp'],
    ['Shake Dat Ass (Twerk Song)', 'BossMan Dlow', 'Shake Dat Ass (Twerk Song)', 'BossMan DLow'],
    ['Soak City', '310babii', 'Soak City (Do It)', '310babii'],
    ['Still', 'Karol G With Bruno Mars', 'Still', 'Karol G & Bruno Mars'],
    ['Sugar On My Tongue', 'Tyler, The Creator', 'Sugar On My Tongue', 'Tyler, the Creator'],
    ['Take Me Thru Dere', 'Metro Boomin, Quavo, Breskii & YKNIECE',
     'Take Me Thru Dere', 'Metro Boomin, Quavo, Breskii & YK NIECE'],
    ['Tweaker', 'GELO', 'Tweaker', 'Gelo'],
    ['Typa', 'Glorilla', 'Typa', 'GloRilla'],
    ['Your Idol', 'Saja Boys: Andrew Choi, Neckwav, Danny Chung, Kevin Woo & samUIL Lee',
     'Your Idol', 'Saja Boys, Andrew Choi, Neckwav, Danny Chung, Kevin Woo & samUIL Lee'],
]

# ---- Step 4 (cell 33)
PROTECTED_NAMES = {'lil nas x': 'lil nas-x', 'sam & dave': 'sam and dave', 'sonny & cher': 'sonny and cher'}
SEPARATORS = [' a duet with ', ' duet with ', ' featuring ', ' feauring ', ' featauring ',
              ' feat. ', ' feat ', ' ft. ', ' with ', ' w/ ', ' vs. ', ' vs ',
              ' introducing ', ' presents ', ' x ', ' & ', ', ', ': ', ' (']
FEATURE_WORDS = ['featuring', 'feauring', 'featauring', 'feat.', 'ft.']

# ---- Step 8 (cell 58)
LATIN_WORDS = ['latin', 'latino', 'latina', 'reggaeton', 'regueton', 'salsa', 'bachata', 'cumbia',
               'corrido', 'corridos', 'banda', 'mariachi', 'tejano', 'merengue', 'dembow', 'norteño',
               'norteno', 'ranchera', 'mexican', 'mexicano', 'bolero', 'flamenco', 'samba', 'bossa',
               'brazilian', 'brasileiro', 'brasileira', 'carioca', 'ostentação', 'brega', 'sertanejo',
               'español', 'espanol', 'portuguese']
COUNTRY_WORDS = ['country', 'bluegrass', 'americana', 'honky', 'nashville', 'urban cowboy']
HIPHOP_WORDS = ['hip hop', 'hiphop', 'rap', 'trap', 'drill', 'grime', 'crunk', 'boom bap', 'bounce',
                'hyphy', 'dirty south', 'rage', 'plugg', 'horrorcore', 'ratchet']
RNB_WORDS = ['funk', 'funky', 'funktronica', 'motown', 'new jack swing', 'quiet storm', 'doo wop']
ROCK_WORDS = ['rock', 'punk', 'metal', 'grunge', 'indie', 'alternative', 'emo', 'shoegaze',
              'blackgaze', 'britpop', 'new wave', 'psychedelia']
DANCE_WORDS = ['electronic', 'electronica', 'electro', 'edm', 'house', 'techno', 'trance', 'dubstep',
               'dance', 'disco', 'club', 'rave', 'drum and bass', 'dnb', 'bass', 'synthwave',
               'eurodance', 'hardstyle', 'garage', 'trip hop', 'complextro', 'downtempo', 'leftfield',
               'indietronica', 'jungle', 'ambient', 'tronica', 'breakbeat', 'moombahton', 'brostep',
               'ebm', 'hardcore']

# ---- Step 10 (cell 78)
PROFANE_STARTS = ['fuck', 'shit', 'bitch', 'nigga', 'nigger', 'cunt', 'pussy', 'motherfuck',
                  'mothafuck', 'asshole', 'bastard', 'whore', 'slut']
PROFANE_WORDS = ['dick', 'dicks', 'cock', 'hoe', 'hoes']

ANALYSIS_COLUMNS = ['title', 'artist', 'primary_artist', 'debut_date', 'top10', 'in_model', 'model_split',
                    'debut_year', 'debut_month', 'debut_position', 'primary_genre', 'prior_entries',
                    'prior_top10s', 'album_debut_bucket', 'has_feature', 'n_credited_artists', 'has_profanity',
                    'days_release_to_chart', 'duration_used', 'word_count', 'unique_word_ratio',
                    'repetition_score', 'avg_line_length', 'lyrics_suspect_short', 'release_format',
                    'artist_country']


# ---- genre helpers (cells 59, 61, 62, 64), unchanged
def has_keyword(label, keywords):
    label = label.replace('-', ' ')
    words = label.split()
    for keyword in keywords:
        if ' ' in keyword and keyword in label:
            return True
        if keyword in words:
            return True
    return False


def broad_genre(label):
    label = label.lower().strip()
    if 'dancehall' in label:
        return 'other'
    elif has_keyword(label, LATIN_WORDS):
        return 'Latin'
    elif has_keyword(label, COUNTRY_WORDS):
        return 'country'
    elif label.endswith('pop'):
        return 'pop'
    elif has_keyword(label, ['soul', 'r&b', 'rnb']):
        return 'R&B'
    elif has_keyword(label, HIPHOP_WORDS):
        return 'hip-hop/rap'
    elif has_keyword(label, RNB_WORDS):
        return 'R&B'
    elif has_keyword(label, ROCK_WORDS):
        return 'rock/alternative'
    elif has_keyword(label, DANCE_WORDS):
        return 'dance/electronic'
    elif has_keyword(label, ['pop']):
        return 'pop'
    else:
        return 'other'


def genre_votes(names, counts):
    votes = {}
    if pd.isna(names):
        return votes
    name_list = names.split('|')
    count_list = [1] * len(name_list)
    if not pd.isna(counts):
        count_list = [int(c) for c in str(counts).split('|')]
    for name, count in zip(name_list, count_list):
        genre = broad_genre(name)
        votes[genre] = votes.get(genre, 0) + count
    return votes


def song_genre(row):
    song_votes = genre_votes(row['song_genres'], row['song_genre_counts'])
    artist_votes = genre_votes(row['artist_genres'], row['artist_genre_counts'])
    if len(song_votes) == 0:
        song_votes = artist_votes
        artist_votes = {}
    if len(song_votes) == 0:
        return np.nan
    top = max(song_votes.values())
    leaders = [genre for genre in song_votes if song_votes[genre] == top]
    if len(leaders) == 1:
        return leaders[0]
    best = max([artist_votes.get(genre, 0) for genre in leaders])
    winners = [genre for genre in leaders if artist_votes.get(genre, 0) == best]
    if best > 0 and len(winners) == 1:
        return winners[0]
    return 'mixed'


# ---- lyric helpers (cells 74, 76, 77, 78), unchanged
def unique_word_ratio(text):
    if pd.isna(text) or len(text.split()) == 0:
        return np.nan
    words = text.split()
    return len(set(words)) / len(words)


def lyric_lines(text):
    lines = []
    for line in text.split('\n'):
        line = ' '.join(line.split())
        if line != '':
            lines.append(line)
    return lines


def repetition_score(text):
    if pd.isna(text) or len(lyric_lines(text)) == 0:
        return np.nan
    lines = lyric_lines(text)
    repeated = 0
    for line in lines:
        if lines.count(line) > 1:
            repeated = repeated + 1
    return repeated / len(lines)


def has_profanity(text):
    if pd.isna(text) or len(text.split()) == 0:
        return np.nan
    for word in text.split():
        if word in PROFANE_WORDS:
            return True
        for start in PROFANE_STARTS:
            if word.startswith(start):
                return True
    return False


def _song_table(weekly):
    """Cells 23 / 29: one row per (title, artist) credit."""
    songs = weekly.sort_values('chart_date').groupby(['title', 'artist']).agg(
        debut_date=('chart_date', 'min'),
        last_date=('chart_date', 'max'),
        peak_position=('position', 'min'),
        debut_position=('position', 'first'),
        weeks_on_chart=('chart_date', 'count'),
    ).reset_index()
    songs['top10'] = songs['peak_position'] <= 10
    return songs


def build(verbose=False):
    log = print if verbose else (lambda *a, **k: None)

    # Step 1 (cells 10, 12, 16)
    path = os.path.join(RAW, 'all.json')
    charts = pd.read_json(path) if os.path.exists(path) else pd.read_json(BILLBOARD_URL)
    rows = []
    for i in range(len(charts)):
        for entry in charts['data'][i]:
            rows.append([charts['date'][i], entry['song'], entry['artist'], entry['this_week']])
    weekly = pd.DataFrame(rows, columns=['chart_date', 'title', 'artist', 'position'])
    weekly['chart_date'] = pd.to_datetime(weekly['chart_date'])
    weekly = weekly[weekly['chart_date'] <= LAST_CHART]
    log('weekly charts kept:', weekly['chart_date'].nunique())

    lyrics = pd.read_csv(os.path.join(RAW, 'lyrics.csv'))
    lyrics = lyrics[['title', 'artist', 'plain_lyrics', 'instrumental', 'duration', 'match_score']]
    mb_songs = pd.read_csv(os.path.join(RAW, 'mb_songs.csv'))
    mb_songs = mb_songs[['title', 'artist', 'first_release_date', 'release_type', 'recording_length_ms',
                         'genres', 'genre_counts', 'artist_mbids', 'match_score']]
    mb_artists = pd.read_csv(os.path.join(RAW, 'mb_artists.csv'))
    mb_artists = mb_artists[['artist_mbid', 'country', 'genres', 'genre_counts']]

    # Step 2 (cells 23-24)
    credits = _song_table(weekly)
    credits_2018 = credits[credits['debut_date'] >= '2018-01-01']

    # Step 3 (cells 28-29)
    weekly = weekly.copy()
    for old_title, old_artist, new_title, new_artist in DUPLICATE_CREDITS:
        match = (weekly['title'] == old_title) & (weekly['artist'] == old_artist)
        weekly.loc[match, 'title'] = new_title
        weekly.loc[match, 'artist'] = new_artist
    songs = _song_table(weekly)
    songs_2018 = songs[songs['debut_date'] >= '2018-01-01']
    log('songs from 2018 on:', len(credits_2018), '->', len(songs_2018))
    log('Top 10 songs from 2018 on:', credits_2018['top10'].sum(), '->', songs_2018['top10'].sum())

    # Step 4 (cells 34, 36)
    names = songs['artist'].str.lower()
    names = names.str.replace('"', '', regex=False).str.replace(')', '', regex=False)
    for name in PROTECTED_NAMES:
        names = names.str.replace(name, PROTECTED_NAMES[name], regex=False)
    for separator in SEPARATORS:
        names = names.str.replace(separator, '|', regex=False)
    artist_columns = names.str.split('|', expand=True)
    songs['primary_artist'] = artist_columns[0].str.strip()
    songs['n_credited_artists'] = artist_columns.notna().sum(axis=1)
    songs['has_feature'] = False
    for word in FEATURE_WORDS:
        songs['has_feature'] = songs['has_feature'] | songs['artist'].str.lower().str.contains(word, regex=False)

    # Step 5 (cells 40, 42, 44, 45)
    top10_weeks = weekly[weekly['position'] <= 10]
    first_top10 = top10_weeks.groupby(['title', 'artist']).agg(first_top10_date=('chart_date', 'min')).reset_index()
    songs = pd.merge(songs, first_top10, on=['title', 'artist'], how='left', validate='one_to_one')

    pieces = []
    for column in artist_columns.columns:
        piece = songs[['title', 'artist', 'debut_date', 'top10', 'first_top10_date']].copy()
        piece['artist_name'] = artist_columns[column].str.strip()
        pieces.append(piece)
    artist_songs = pd.concat(pieces)
    artist_songs = artist_songs.dropna(subset=['artist_name'])
    artist_songs = artist_songs[artist_songs['artist_name'] != '']
    artist_songs = artist_songs.drop_duplicates(subset=['title', 'artist', 'artist_name'])

    earlier_songs = artist_songs.rename(columns={
        'title': 'earlier_title', 'artist': 'earlier_artist', 'debut_date': 'earlier_debut',
        'top10': 'earlier_top10', 'first_top10_date': 'earlier_first_top10', 'artist_name': 'primary_artist'})
    recent = songs[songs['debut_date'] >= '2018-01-01']
    pairs = pd.merge(recent[['title', 'artist', 'debut_date', 'primary_artist']],
                     earlier_songs, on='primary_artist', how='left')
    earlier = pairs[pairs['earlier_debut'] < pairs['debut_date']].copy()
    earlier['hit_before_debut'] = earlier['earlier_first_top10'] < earlier['debut_date']
    history = earlier.groupby(['title', 'artist']).agg(
        prior_entries=('earlier_title', 'count'),
        prior_top10s=('hit_before_debut', 'sum'),
        eventual_top10s=('earlier_top10', 'sum'),
    ).reset_index()
    self_counted = (earlier['earlier_title'] == earlier['title']) & (earlier['earlier_artist'] == earlier['artist'])
    assert self_counted.sum() == 0, 'a song was counted in its own history'

    # Step 6 (cells 49, 51)
    last_week = weekly['chart_date'].max()
    songs = songs[songs['debut_date'] >= '2018-01-01'].copy()
    songs = pd.merge(songs, history, on=['title', 'artist'], how='left', validate='one_to_one')
    songs['prior_entries'] = songs['prior_entries'].fillna(0)
    songs['prior_top10s'] = songs['prior_top10s'].fillna(0)
    songs['still_charting'] = songs['last_date'] == last_week
    songs['in_model'] = ~songs['still_charting']
    songs['debut_year'] = songs['debut_date'].dt.year
    songs['debut_month'] = songs['debut_date'].dt.month
    songs['model_split'] = 'train'
    songs.loc[songs['debut_year'] >= 2025, 'model_split'] = 'test'
    songs.loc[songs['still_charting'], 'model_split'] = 'excluded'

    # Step 7 (cells 53, 54, 56)
    lyrics = lyrics.rename(columns={'match_score': 'lyrics_match_score'})
    songs = pd.merge(songs, lyrics, on=['title', 'artist'], how='left', validate='one_to_one')
    mb_songs = mb_songs.rename(columns={'match_score': 'mb_match_score', 'genres': 'song_genres',
                                        'genre_counts': 'song_genre_counts', 'release_type': 'release_format'})
    songs = pd.merge(songs, mb_songs, on=['title', 'artist'], how='left', validate='one_to_one')
    songs['lead_mbid'] = songs['artist_mbids'].str.split('|', expand=True)[0]
    mb_artists = mb_artists.rename(columns={'artist_mbid': 'lead_mbid', 'genres': 'artist_genres',
                                            'genre_counts': 'artist_genre_counts', 'country': 'artist_country'})
    songs = pd.merge(songs, mb_artists, on='lead_mbid', how='left', validate='many_to_one')

    # Step 8 (cell 65)
    songs['primary_genre'] = songs.apply(song_genre, axis=1)

    # Step 9 (cells 67, 69)
    songs['duration_used'] = songs['duration']
    bad = (songs['duration'] < 30) | (songs['duration'] > 600) | songs['duration'].isna()
    songs.loc[bad, 'duration_used'] = songs.loc[bad, 'recording_length_ms'] / 1000
    still_bad = (songs['duration_used'] < 30) | (songs['duration_used'] > 600)
    songs.loc[still_bad, 'duration_used'] = np.nan
    songs['lyrics_chars'] = songs['plain_lyrics'].str.len()
    songs['chars_per_sec'] = songs['lyrics_chars'] / songs['duration_used']
    cutoffs = songs.groupby('primary_genre')['chars_per_sec'].quantile(0.02)
    songs['genre_cutoff'] = songs['primary_genre'].map(cutoffs)
    songs['genre_cutoff'] = songs['genre_cutoff'].fillna(songs['chars_per_sec'].quantile(0.02))
    songs['lyrics_suspect_short'] = songs['chars_per_sec'] < songs['genre_cutoff']

    # Step 10 (cells 71, 72, 74, 76, 77, 78)
    clean = []
    for text in songs['plain_lyrics']:
        if pd.notna(text):
            while '[' in text and ']' in text[text.index('['):]:
                start = text.index('[')
                end = text.index(']', start)
                text = text[:start] + text[end + 1:]
        clean.append(text)
    songs['lyrics_clean'] = clean
    songs.loc[songs['instrumental'] == True, 'lyrics_clean'] = np.nan
    keep_chars = "abcdefghijklmnopqrstuvwxyz0123456789'"
    lyrics_words = []
    for text in songs['lyrics_clean'].str.lower():
        if pd.isna(text):
            lyrics_words.append(text)
        else:
            lyrics_words.append(''.join(ch if (ch in keep_chars or ch.isspace()) else ' ' for ch in text))
    songs['lyrics_words'] = lyrics_words
    songs['word_count'] = songs['lyrics_words'].str.split().str.len()
    songs.loc[songs['word_count'] == 0, 'word_count'] = np.nan
    songs['unique_word_ratio'] = songs['lyrics_words'].apply(unique_word_ratio)
    line_counts = songs['lyrics_words'].dropna().apply(lyric_lines).str.len()
    songs['avg_line_length'] = songs['word_count'] / line_counts
    songs['repetition_score'] = songs['lyrics_words'].apply(repetition_score)
    songs['has_profanity'] = songs['lyrics_words'].apply(has_profanity)

    # Step 11 (cells 81, 83)
    songs['same_artist_debuts'] = songs.groupby(['primary_artist', 'debut_date'])['title'].transform('size')
    songs['album_debut_bucket'] = '10+'
    songs.loc[songs['same_artist_debuts'] <= 9, 'album_debut_bucket'] = '5-9'
    songs.loc[songs['same_artist_debuts'] <= 4, 'album_debut_bucket'] = '2-4'
    songs.loc[songs['same_artist_debuts'] == 1, 'album_debut_bucket'] = '1'
    release_date = pd.to_datetime(songs['first_release_date'], format='%Y-%m-%d', errors='coerce')
    songs['days_release_to_chart'] = (songs['debut_date'] - release_date).dt.days

    # Step 12 (cell 85). The wide table keeps the extra columns the extension needs,
    # but never the lyrics text itself.
    wide = songs.drop(columns=['plain_lyrics', 'lyrics_clean', 'lyrics_words'])
    extras = {'weekly': weekly, 'artist_songs': artist_songs, 'wide': wide}
    return songs[ANALYSIS_COLUMNS].copy(), extras


def team_model(songs):
    """Cells 128-139: the team's two logistic regressions, unchanged.

    Returns a dict with the fitted models, the prepared matrices and the scores.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    finished = songs[songs['in_model'] == True].copy()
    baseline_features = ['prior_top10s', 'prior_entries']
    full_features = baseline_features + ['primary_genre', 'album_debut_bucket', 'n_credited_artists',
                                         'has_feature', 'has_profanity', 'duration_used', 'days_release_to_chart',
                                         'word_count', 'unique_word_ratio', 'repetition_score', 'avg_line_length']
    X = finished[full_features].copy()
    X = pd.get_dummies(X, columns=['primary_genre', 'album_debut_bucket'], drop_first=True)
    X = X.astype(float)
    y = finished['top10']
    Xtrain = X[finished['model_split'] == 'train']
    Xtest = X[finished['model_split'] == 'test']
    ytrain = y[finished['model_split'] == 'train']
    ytest = y[finished['model_split'] == 'test']
    train_medians = Xtrain.median()
    Xtrain = Xtrain.fillna(train_medians)
    Xtest = Xtest.fillna(train_medians)
    train_means = Xtrain.mean()
    train_stds = Xtrain.std()
    Xtrain = (Xtrain - train_means) / train_stds
    Xtest = (Xtest - train_means) / train_stds

    baseline = LogisticRegression(solver='liblinear')
    baseline.fit(Xtrain[baseline_features], ytrain)
    y_baseline = baseline.predict(Xtest[baseline_features])
    model = LogisticRegression(solver='liblinear')
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)

    scores = {}
    for name, predicted in [['baseline', y_baseline], ['full', y_model]]:
        mat = confusion_matrix(ytest, predicted)
        tp, fp, fn = mat[1, 1], mat[0, 1], mat[1, 0]
        scores[name] = {'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'flagged': int(predicted.sum()),
                        'caught': int(tp)}
    return {'finished': finished, 'Xtrain': Xtrain, 'Xtest': Xtest, 'ytrain': ytrain, 'ytest': ytest,
            'baseline': baseline, 'model': model, 'baseline_features': baseline_features,
            'full_features': full_features, 'scores': scores}


if __name__ == '__main__':
    songs, extras = build(verbose=True)
    print('\n== Reproduction check ==')
    print('songs from 2018 on:', len(songs), '(team: 5,342)')
    print('Top 10:', int(songs['top10'].sum()), f"({songs['top10'].mean() * 100:.1f}%; team: 519, 9.7%)")
    print(songs.groupby('model_split').agg(songs=('title', 'size'), top10=('top10', 'sum')))
    out = team_model(songs)
    for name, s in out['scores'].items():
        print(f"{name:9s} precision {s['precision'] * 100:.1f}%  recall {s['recall'] * 100:.1f}%  "
              f"flagged {s['flagged']}  caught {s['caught']}")
    print('random-pick precision (test Top 10 rate):', f"{out['ytest'].mean() * 100:.1f}%")
    print('team: full 42.9% / 26.8% (35 flagged, 15 caught); baseline 19.6% / 16.1%; random 5.9%')
