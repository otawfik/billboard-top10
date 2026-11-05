"""Print worked examples from the held-out predictions (run model.py first).

    python extension/examples.py [model name]

Shows the best-ranked real hits, the highest-scored false alarms and the hits the
model ranked lowest, with the debut-week facts the model saw. Outcomes (peak,
debut position) are shown for reading only; they were never features.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
pred = pd.read_csv(os.path.join(HERE, 'output', 'heldout_predictions.csv'))
model = sys.argv[1] if len(sys.argv) > 1 else 'Regularised LR (extended)'
col = 'p_' + model
pred['rank'] = pred[col].rank(ascending=False, method='first').astype(int)
show = ['rank', 'title', 'artist', col, 'top10', 'peak_position', 'debut_position', 'prior_top10s',
        'prior_entries', 'weeks_since_last_top10', 'artist_weeks_last_52', 'same_artist_debuts',
        'feat_prior_top10s', 'primary_genre', 'release_format']
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 28)
print(f'Model: {model}\n')
print('Highest-ranked songs that became hits:')
print(pred[pred['top10']].sort_values(col, ascending=False)[show].head(8).round(3).to_string(index=False))
print('\nHighest-ranked songs that did NOT become hits (false alarms):')
print(pred[~pred['top10']].sort_values(col, ascending=False)[show].head(8).round(3).to_string(index=False))
print('\nHits the model ranked lowest (misses):')
print(pred[pred['top10']].sort_values(col)[show].head(8).round(3).to_string(index=False))
print('\nHits by rank band:')
bands = pd.cut(pred['rank'], [0, 50, 100, 200, 400, 1000])
print(pred.groupby(bands, observed=True)['top10'].agg(['size', 'sum']).to_string())
print('\nHits that debuted inside the Top 10 vs climbed there, by whether the model ranked them top 50:')
hits = pred[pred['top10']].copy()
hits['debuted_in_top10'] = hits['debut_position'] <= 10
hits['ranked_top50'] = hits['rank'] <= 50
print(pd.crosstab(hits['debuted_in_top10'], hits['ranked_top50']))
