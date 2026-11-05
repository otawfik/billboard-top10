"""How many chart weeks until a climber becomes predictable?

    python extension/weeks.py

The debut-week model finds hits that are already in the Top 10 in week one, and misses
the songs that climb there later. This script asks the next question: after k weeks on
the chart (k = 1, 2, 3, 4, 6), can the songs that have NOT reached the Top 10 yet be
ranked by whether they ever will?

For each k:
  population  every song (2018 on, final peak known) that was not in the Top 10 in any of
              its first k chart weeks (week 1 = debut week). Songs that already fell off the
              chart stay in: at week k you would still have to score them.
  target      reaches the Top 10 at any later point (= the team's `top10` for these songs).
  features    known by the end of week k only:
              - the debut-week features from model.py (team + extended), unchanged;
              - the song's position in each of weeks 1..k (101 = not on the chart that week),
                best position so far, last week-over-week move, weeks on chart so far;
              - debut position, which is allowed here: no song in the population debuted in
                the Top 10, so it no longer defines the target;
              - the lead artist's record as of week k: earlier songs that had reached the
                Top 10 by week k, whether the artist has another song in the Top 10 in week
                k, and how many songs from the same-week drop have reached it by week k.
  models      regularised LR and HistGradientBoosting, each tuned by rolling origin inside
              2018-2024 (validate 2019..2024 on earlier years); the one with the better mean
              validation PR-AUC is the pre-chosen model for that k. Two untuned references:
              "best position so far" (a rule, no training) and gradient boosting on the
              debut-week features only (the debut-week model's tuned settings).
  evaluation  one held-out 2025-2026 score per k, with 2,000 bootstrap resamples.

Writes aggregate results to extension/results/weeks_*.{csv,json} and a chart to
extension/charts/chart-weeks-to-predictable.jpg. Song-level rows stay in extension/output/.
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402
from build_dataset import build  # noqa: E402
from features import add_features  # noqa: E402

warnings.filterwarnings('ignore')

KS = [1, 2, 3, 4, 6]
TOP_N = 20
SEED = M.SEED
N_BOOT = M.N_BOOT
OFF = 101  # position code for "not on the chart that week"


def traj_cols(k):
    return [f'pos_w{w}' for w in range(1, k + 1)] + [
        'debut_position', 'best_pos_so_far', 'last_move', 'weeks_on_so_far', 'on_chart_now',
        'artist_hits_by_k', 'artist_other_top10_now', 'drop_siblings_hit_by_k']


def week_tables(songs, extras):
    """Per song: position in chart weeks 1..max(KS), and the week it first hit the Top 10."""
    weekly = extras['weekly']
    w = weekly.merge(songs[['title', 'artist', 'debut_date']], on=['title', 'artist'], how='inner')
    w['week'] = ((w['chart_date'] - w['debut_date']).dt.days // 7 + 1).astype(int)
    pos = w[w['week'] <= max(KS)].pivot_table(index=['title', 'artist'], columns='week', values='position',
                                               aggfunc='min')
    pos = pos.reindex(columns=range(1, max(KS) + 1)).fillna(OFF)
    pos.columns = [f'pos_w{c}' for c in pos.columns]
    first_hit = w[w['position'] <= 10].groupby(['title', 'artist'])['week'].min().rename('first_top10_week')
    return pos.reset_index(), first_hit.reset_index()


def artist_state(songs, extras, k):
    """Lead artist's record as of the end of week k (chart date = debut + 7(k-1) days)."""
    a = extras['artist_songs'][['title', 'artist', 'artist_name', 'debut_date', 'first_top10_date']]
    weekly = extras['weekly']
    base = songs[['title', 'artist', 'primary_artist', 'debut_date']].copy()
    base['cutoff'] = base['debut_date'] + pd.Timedelta(days=7 * (k - 1))
    pairs = base.merge(a.rename(columns={'title': 'o_title', 'artist': 'o_artist', 'debut_date': 'o_debut'}),
                       left_on='primary_artist', right_on='artist_name', how='inner')
    pairs = pairs[~((pairs['o_title'] == pairs['title']) & (pairs['o_artist'] == pairs['artist']))]
    key = ['title', 'artist']
    # earlier songs (debuted before this one) that had reached the Top 10 by week k
    hits = pairs[(pairs['o_debut'] < pairs['debut_date']) & (pairs['first_top10_date'] <= pairs['cutoff'])]
    out = base[key].merge(hits.groupby(key).size().rename('artist_hits_by_k').reset_index(), on=key, how='left')
    # songs from the same-week drop that had reached the Top 10 by week k
    sib = pairs[(pairs['o_debut'] == pairs['debut_date']) & (pairs['first_top10_date'] <= pairs['cutoff'])]
    out = out.merge(sib.groupby(key).size().rename('drop_siblings_hit_by_k').reset_index(), on=key, how='left')
    # another song by the lead artist in the Top 10 on the week-k chart
    top = weekly[weekly['position'] <= 10][['title', 'artist', 'chart_date']].rename(
        columns={'title': 'o_title', 'artist': 'o_artist'})
    now = pairs.merge(top, left_on=['o_title', 'o_artist', 'cutoff'], right_on=['o_title', 'o_artist', 'chart_date'])
    out = out.merge(now.groupby(key).size().rename('artist_other_top10_now').reset_index(), on=key, how='left')
    return out.fillna(0)


def population(df, pos, first_hit, songs, extras, k):
    d = df.merge(pos, on=['title', 'artist'], how='left').merge(first_hit, on=['title', 'artist'], how='left')
    d = d[~(d['first_top10_week'] <= k)].copy()  # not in the Top 10 in weeks 1..k
    cols = [f'pos_w{w}' for w in range(1, k + 1)]
    P = d[cols].values
    d['best_pos_so_far'] = P.min(axis=1)
    d['last_move'] = (P[:, -2] - P[:, -1]) if k > 1 else 0.0  # positive = climbed last week
    d['weeks_on_so_far'] = (P < OFF).sum(axis=1)
    d['on_chart_now'] = (P[:, -1] < OFF).astype(float)
    d = d.merge(artist_state(songs, extras, k), on=['title', 'artist'], how='left')
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ validation helpers
def rolling(df, factory, num, cat):
    preds = pd.Series(np.nan, index=df.index)
    for y in M.VAL_YEARS:
        tr, va = df['debut_year'] < y, df['debut_year'] == y
        m = factory().fit(M.prepare(df[tr], num, cat), df.loc[tr, 'top10'].astype(int))
        preds[va] = m.predict_proba(M.prepare(df[va], num, cat))[:, 1]
    return preds


def mean_val_ap(df, preds):
    s = []
    for y in M.VAL_YEARS:
        va = df['debut_year'] == y
        if df.loc[va, 'top10'].sum() > 0:
            s.append(average_precision_score(df.loc[va, 'top10'].astype(int), preds[va]))
    return float(np.mean(s))


def tune(df, cands, num, cat):
    best = None
    for params, factory in cands:
        sc = mean_val_ap(df, rolling(df, factory, num, cat))
        if best is None or sc > best[0]:
            best = (sc, params, factory)
    return best


def scores(y, p, n=TOP_N):
    order = np.argsort(-p, kind='stable')
    top = order[:n]
    return {'pr_auc': average_precision_score(y, p), 'roc_auc': roc_auc_score(y, p),
            f'p_at_{n}': y[top].mean(), f'recall_at_{n}': y[top].sum() / max(y.sum(), 1),
            'p_at_50': y[order[:50]].mean(), 'recall_at_50': y[order[:50]].sum() / max(y.sum(), 1)}


def boot(y, preds, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    acc = {m: [] for m in preds}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        for m, p in preds.items():
            acc[m].append(scores(y[idx], p[idx]))
    return {m: {k: (float(np.percentile(pd.DataFrame(v)[k], 2.5)), float(np.percentile(pd.DataFrame(v)[k], 97.5)))
                for k in v[0]} for m, v in acc.items()}


def main():
    print('Building the team dataset and debut-week features ...')
    songs, extras = build()
    df = add_features(songs, extras)
    df = df[df['in_model'] == True].reset_index(drop=True)  # noqa: E712
    pos, first_hit = week_tables(df, extras)
    summary = json.load(open(os.path.join(HERE, 'results', 'summary.json')))
    gb_debut_params = summary['tuned_params']['HistGB (extended)']

    M.LOG_COLS.extend(['artist_hits_by_k', 'artist_other_top10_now', 'drop_siblings_hit_by_k', 'weeks_on_so_far'])
    lr_grid = list(itertools.product(['l2', 'l1'], [0.01, 0.03, 0.1, 0.3, 1.0]))
    gb_grid = list(itertools.product([0.05, 0.1], [4, 8, 16], [20, 50], [100, 300]))

    rows, examples, curves = [], [], {}
    for k in KS:
        d = population(df, pos, first_hit, songs, extras, k)
        tr = d[d['model_split'] == 'train'].reset_index(drop=True)
        te = d[d['model_split'] == 'test'].reset_index(drop=True)
        y_te = te['top10'].astype(int).values
        num = M.EXT_NUM + traj_cols(k)
        cat = M.EXT_CAT
        print(f'\nk = {k}: train {len(tr)} songs ({int(tr.top10.sum())} later hits), '
              f'test {len(te)} songs ({y_te.sum()} later hits, base rate {y_te.mean():.1%})')

        lr_best = tune(tr, [({'penalty': pe, 'C': C}, lambda pe=pe, C=C: M.lr_pipeline(num, cat, C, pe))
                            for pe, C in lr_grid], num, cat)
        gb_best = tune(tr, [(dict(learning_rate=a, max_leaf_nodes=b, min_samples_leaf=c, max_iter=e),
                             lambda a=a, b=b, c=c, e=e: M.hgb_pipeline(num, cat, learning_rate=a, max_leaf_nodes=b,
                                                                       min_samples_leaf=c, max_iter=e))
                            for a, b, c, e in gb_grid], num, cat)
        chosen = 'LR' if lr_best[0] >= gb_best[0] else 'GB'
        print(f'  validation PR-AUC: LR {lr_best[0]:.3f} {lr_best[1]} | GB {gb_best[0]:.3f} {gb_best[1]} '
              f'-> pre-chosen: {chosen}')

        preds = {}
        for name, best in [('LR', lr_best), ('GB', gb_best)]:
            m = best[2]().fit(M.prepare(tr, num, cat), tr['top10'].astype(int))
            preds[name] = m.predict_proba(M.prepare(te, num, cat))[:, 1]
        preds['Chosen'] = preds[chosen]
        gb_debut = M.hgb_pipeline(M.EXT_NUM, M.EXT_CAT, **gb_debut_params).fit(
            M.prepare(tr, M.EXT_NUM, M.EXT_CAT), tr['top10'].astype(int))
        preds['Debut-week features only'] = gb_debut.predict_proba(M.prepare(te, M.EXT_NUM, M.EXT_CAT))[:, 1]
        # rule: best position so far (lower is better); ties broken by current position
        preds['Best position so far'] = -(te['best_pos_so_far'].values + te[f'pos_w{k}'].values / 1000)

        ci = boot(y_te, preds)
        for name, p in preds.items():
            s = scores(y_te, p)
            row = {'k': k, 'model': name, 'n_test': len(te), 'climbers_test': int(y_te.sum()),
                   'base_rate': float(y_te.mean()), 'on_chart_now_test': int(te['on_chart_now'].sum()),
                   'n_train': len(tr), 'climbers_train': int(tr['top10'].sum()),
                   'val_pr_auc': {'LR': lr_best[0], 'GB': gb_best[0]}.get(name if name != 'Chosen' else chosen),
                   'chosen': chosen if name == 'Chosen' else ''}
            for key, v in s.items():
                row[key], (row[key + '_lo'], row[key + '_hi']) = v, ci[name][key]
            rows.append(row)
        r = pd.DataFrame([x for x in rows if x['k'] == k]).set_index('model')
        print(r[['pr_auc', 'pr_auc_lo', 'pr_auc_hi', f'p_at_{TOP_N}', f'recall_at_{TOP_N}', 'recall_at_50',
                 'roc_auc']].round(3).to_string())

        # ranks of each held-out climber at this k (for worked examples; stays local)
        te = te.assign(score=preds['Chosen'])
        te['rank'] = te['score'].rank(ascending=False, method='first').astype(int)
        hit = te[te['top10']][['title', 'artist', 'rank'] + [f'pos_w{w}' for w in range(1, k + 1)]].copy()
        hit['k'] = k
        hit['of'] = len(te)
        examples.append(hit)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, 'results', 'weeks_results.csv'), index=False)
    ex = pd.concat(examples)
    os.makedirs(os.path.join(HERE, 'output'), exist_ok=True)
    ex.to_csv(os.path.join(HERE, 'output', 'weeks_climber_ranks.csv'), index=False)

    # climbers' paths for the README (positions only)
    traj = pos.merge(first_hit, on=['title', 'artist']).merge(
        df[df['model_split'] == 'test'][['title', 'artist', 'top10']], on=['title', 'artist'])
    traj = traj[traj['first_top10_week'] > 1]
    traj.to_csv(os.path.join(HERE, 'output', 'weeks_climber_paths.csv'), index=False)
    print('\nHeld-out climbers: week they first reached the Top 10')
    print(traj['first_top10_week'].value_counts().sort_index().to_string())
    wide = ex.pivot_table(index=['title', 'artist'], columns='k', values='rank')
    wide = wide.join(traj.set_index(['title', 'artist'])[['first_top10_week'] + [f'pos_w{w}' for w in range(1, 7)]])
    print('\nRank of each held-out climber at each k (chosen model; blank = already in the Top 10):')
    print(wide.sort_values('first_top10_week').to_string())


if __name__ == '__main__':
    main()
