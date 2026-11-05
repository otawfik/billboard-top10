"""Tune, compare and evaluate debut-week Top 10 models.

    python extension/model.py

1. Rebuild the team's dataset (build_dataset.py) and add the new features (features.py).
2. Tune every model with rolling-origin validation inside 2018-2024: for each year y in
   2019..2024, train on 2018..y-1 and validate on y. Imputation and scaling are fitted
   inside each fold's training years only (sklearn Pipelines).
3. Refit each tuned model on all of 2018-2024 and score the held-out 2025-2026 songs
   ONCE. Bootstrap the held-out songs for 95% CIs.
4. Explain the best model: permutation importance on the held-out songs, partial
   dependence, and worked examples.

Writes aggregate results to extension/results/ (committed) and song-level predictions
to extension/output/ (git-ignored: it holds chart data).
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_dataset import build, team_model  # noqa: E402
from features import add_features  # noqa: E402

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)  # liblinear 'penalty' deprecation notice
RESULTS = os.path.join(HERE, 'results')
OUTPUT = os.path.join(HERE, 'output')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

SEED = 42
VAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
TOP_K = 50
N_BOOT = 2000

# ---------------------------------------------------------------- feature sets
TEAM_NUM = ['prior_top10s', 'prior_entries', 'n_credited_artists', 'has_feature', 'has_profanity',
            'duration_used', 'days_release_to_chart', 'word_count', 'unique_word_ratio',
            'repetition_score', 'avg_line_length']
TEAM_CAT = ['primary_genre', 'album_debut_bucket']
NEW_NUM = ['prior_chart_weeks', 'artist_weeks_last_52', 'artist_songs_prev_week', 'prior_best_peak',
           'weeks_since_last_chart', 'weeks_since_last_top10', 'prior_top10_rate', 'feat_prior_top10s',
           'same_artist_debuts', 'week_new_entries']
NEW_CAT = ['release_format', 'lead_country']
EXT_NUM = TEAM_NUM + NEW_NUM
EXT_CAT = TEAM_CAT + NEW_CAT
# Heavy-tailed counts get log1p before scaling in the linear model (trees don't care).
LOG_COLS = ['prior_top10s', 'prior_entries', 'prior_chart_weeks', 'artist_weeks_last_52',
            'artist_songs_prev_week', 'prior_best_peak', 'weeks_since_last_chart', 'weeks_since_last_top10',
            'feat_prior_top10s', 'same_artist_debuts', 'days_release_to_chart']

LEAK_COLS = ['debut_position', 'peak_position', 'weeks_on_chart', 'last_date', 'first_top10_date']


def prepare(df, num, cat):
    X = df[num + cat].copy()
    for c in num:
        X[c] = X[c].astype(float)
    for c in cat:
        X[c] = X[c].fillna('unknown').astype(str)
    return X


def lr_pipeline(num, cat, C, penalty):
    logs = [c for c in num if c in LOG_COLS]
    plain = [c for c in num if c not in LOG_COLS]
    # days_release_to_chart can be negative (MusicBrainz date after debut): clip at 0 before log.
    log_tf = make_pipeline(SimpleImputer(strategy='median'),
                           FunctionTransformer(lambda a: np.log1p(np.clip(a, 0, None)), feature_names_out='one-to-one'),
                           StandardScaler())
    pre = ColumnTransformer([
        ('log', log_tf, logs),
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), plain),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop=None), cat),
    ])
    return Pipeline([('pre', pre), ('lr', LogisticRegression(C=C, penalty=penalty, solver='liblinear',
                                                            max_iter=2000))])


def hgb_pipeline(num, cat, **params):
    def to_cat(X):
        X = X.copy()
        for c in cat:
            X[c] = X[c].astype('category')
        return X
    return Pipeline([('cat', FunctionTransformer(to_cat)),
                     ('hgb', HistGradientBoostingClassifier(categorical_features='from_dtype',
                                                            random_state=SEED, **params))])


class Ensemble:
    """Average of the fitted LR and gradient-boosting pipelines, then the Platt map."""

    def __init__(self, lr, gb, platt):
        self.lr, self.gb, self.platt = lr, gb, platt

    def predict_proba(self, X):
        raw = (self.lr.predict_proba(X)[:, 1] + self.gb.predict_proba(X)[:, 1]) / 2
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        return self.platt.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))


class TeamLR:
    """The team's model exactly: get_dummies(drop_first), train medians, train mean/std, C=1."""

    def __init__(self, num, cat):
        self.num, self.cat = num, cat

    def _dummies(self, X):
        D = pd.get_dummies(X[self.num + self.cat].replace({'unknown': np.nan}),
                           columns=self.cat, drop_first=True).astype(float)
        return D

    def fit(self, X, y):
        D = self._dummies(X)
        self.cols = D.columns
        self.med = D.median()
        D = D.fillna(self.med)
        self.mu, self.sd = D.mean(), D.std().replace(0, 1)
        self.lr = LogisticRegression(solver='liblinear').fit((D - self.mu) / self.sd, y)
        return self

    def predict_proba(self, X):
        D = self._dummies(X).reindex(columns=self.cols, fill_value=0.0).fillna(self.med)
        return self.lr.predict_proba((D - self.mu) / self.sd)


# ---------------------------------------------------------------- metrics
def precision_at_k(y, p, k=TOP_K):
    order = np.argsort(-p, kind='stable')[:k]
    return y[order].mean()


def metrics(y, p, thr):
    flag = p >= thr
    tp = (flag & (y == 1)).sum()
    return {
        'precision': tp / flag.sum() if flag.sum() else np.nan,
        'recall': tp / (y == 1).sum(),
        'flagged': int(flag.sum()),
        'pr_auc': average_precision_score(y, p),
        'roc_auc': roc_auc_score(y, p),
        'brier': brier_score_loss(y, p),
        f'p_at_{TOP_K}': precision_at_k(y, p),
    }


def best_f1_threshold(y, p):
    best, best_t = -1, 0.5
    for t in np.unique(np.round(p, 4)):
        f = p >= t
        tp = (f & (y == 1)).sum()
        if tp == 0:
            continue
        prec, rec = tp / f.sum(), tp / (y == 1).sum()
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best, best_t = f1, t
    return float(best_t)


# ---------------------------------------------------------------- rolling origin
def rolling_predictions(df, make_model, num, cat):
    """Out-of-time predictions for 2019..2024, each from a model trained on earlier years only."""
    preds = pd.Series(np.nan, index=df.index)
    for y in VAL_YEARS:
        tr = df['debut_year'] < y
        va = df['debut_year'] == y
        m = make_model().fit(prepare(df[tr], num, cat), df.loc[tr, 'top10'].astype(int))
        preds[va] = m.predict_proba(prepare(df[va], num, cat))[:, 1]
    return preds


def fold_score(df, preds):
    """Mean PR-AUC across the six validation years (each year scored separately)."""
    s = []
    for y in VAL_YEARS:
        va = df['debut_year'] == y
        s.append(average_precision_score(df.loc[va, 'top10'].astype(int), preds[va]))
    return float(np.mean(s)), s


def tune(df, name, candidates, num, cat):
    rows = []
    best = None
    for params, factory in candidates:
        p = rolling_predictions(df, factory, num, cat)
        mean_ap, per_year = fold_score(df, p)
        v = df['debut_year'].isin(VAL_YEARS)
        ll = log_loss(df.loc[v, 'top10'].astype(int), p[v])
        rows.append({'model': name, **params, 'mean_pr_auc': mean_ap, 'pooled_log_loss': ll,
                     **{f'pr_auc_{y}': a for y, a in zip(VAL_YEARS, per_year)}})
        if best is None or mean_ap > best[0]:
            best = (mean_ap, params, factory, p)
    print(f'  {name}: best mean validation PR-AUC {best[0]:.3f} with {best[1]}')
    return best, pd.DataFrame(rows)


# ---------------------------------------------------------------- bootstrap
def bootstrap(y, preds, thresholds, ref='Team LR', n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    draws = {m: [] for m in preds}
    diffs = {m: [] for m in preds if m != ref}
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        cur = {}
        for m, p in preds.items():
            cur[m] = metrics(yb, p[idx], thresholds[m])
            draws[m].append(cur[m])
        for m in diffs:
            diffs[m].append({k: cur[m][k] - cur[ref][k] for k in ('pr_auc', 'roc_auc', 'brier', f'p_at_{TOP_K}')})
    out = {}
    for m, lst in draws.items():
        d = pd.DataFrame(lst)
        out[m] = {k: (float(np.nanpercentile(d[k], 2.5)), float(np.nanpercentile(d[k], 97.5))) for k in d}
    dout = {}
    for m, lst in diffs.items():
        d = pd.DataFrame(lst)
        dout[m] = {k: {'lo': float(np.percentile(d[k], 2.5)), 'hi': float(np.percentile(d[k], 97.5)),
                       'share_better': float((d[k] < 0).mean() if k == 'brier' else (d[k] > 0).mean())}
                   for k in d}
    return out, dout


# ---------------------------------------------------------------- explanation
GROUPS = {
    'primary_genre': ['primary_genre'], 'album_debut_bucket': ['album_debut_bucket'],
    'lyric measures (5)': ['has_profanity', 'word_count', 'unique_word_ratio', 'repetition_score', 'avg_line_length'],
}


def permutation_importance(model, X, y, cols, n_repeats=50, seed=SEED):
    """Drop in held-out PR-AUC (and ROC-AUC) when one feature (or group) is shuffled."""
    rng = np.random.default_rng(seed)
    base_p = model.predict_proba(X)[:, 1]
    base_ap, base_auc = average_precision_score(y, base_p), roc_auc_score(y, base_p)
    grouped = set(itertools.chain.from_iterable(GROUPS.values()))
    units = {k: v for k, v in GROUPS.items() if all(c in cols for c in v)}
    units.update({c: [c] for c in cols if c not in grouped})
    rows = []
    for name, members in units.items():
        ap_drops, auc_drops = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for c in members:  # shuffle a group's columns together, keeping them consistent
                Xp[c] = X[c].values[perm]
            p = model.predict_proba(Xp)[:, 1]
            ap_drops.append(base_ap - average_precision_score(y, p))
            auc_drops.append(base_auc - roc_auc_score(y, p))
        rows.append({'feature': name, 'pr_auc_drop': np.mean(ap_drops),
                     'pr_auc_drop_lo': np.percentile(ap_drops, 2.5), 'pr_auc_drop_hi': np.percentile(ap_drops, 97.5),
                     'roc_auc_drop': np.mean(auc_drops)})
    return pd.DataFrame(rows).sort_values('pr_auc_drop', ascending=False), base_ap


def partial_dependence(model, X, col, grid):
    vals = []
    for g in grid:
        Xg = X.copy()
        Xg[col] = g
        vals.append(model.predict_proba(Xg)[:, 1].mean())
    return np.array(vals)


# ---------------------------------------------------------------- main
def main():
    print('Building the team dataset ...')
    songs, extras = build()
    team = team_model(songs)
    s = team['scores']
    assert len(songs) == 5342 and int(songs['top10'].sum()) == 519, 'dataset does not reproduce'
    assert round(s['full']['precision'] * 100, 1) == 42.9 and round(s['full']['recall'] * 100, 1) == 26.8
    assert round(s['baseline']['precision'] * 100, 1) == 19.6 and round(s['baseline']['recall'] * 100, 1) == 16.1
    print('  reproduced: 5,342 songs, 519 Top 10, team LR 42.9% / 26.8%, baseline 19.6% / 16.1%')

    df = add_features(songs, extras)
    df = df[df['in_model'] == True].reset_index(drop=True)
    assert not set(LEAK_COLS) & set(EXT_NUM + EXT_CAT)
    train = df[df['model_split'] == 'train'].reset_index(drop=True)
    test = df[df['model_split'] == 'test'].reset_index(drop=True)
    y_tr, y_te = train['top10'].astype(int).values, test['top10'].astype(int).values
    print(f'  train {len(train)} songs ({y_tr.sum()} hits), test {len(test)} songs ({y_te.sum()} hits)')

    # ---------------- candidates
    def lr_cands(num, cat):
        out = []
        for pen, C in itertools.product(['l2', 'l1'], [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]):
            out.append(({'penalty': pen, 'C': C}, lambda n=num, c=cat, C=C, pen=pen: lr_pipeline(n, c, C, pen)))
        return out

    def hgb_cands(num, cat):
        out = []
        for lr_, leaves, msl, l2, iters in itertools.product([0.03, 0.1], [4, 8, 16], [20, 50, 100],
                                                             [0.0, 1.0], [100, 300]):
            params = dict(learning_rate=lr_, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=l2,
                          max_iter=iters)
            out.append((params, lambda n=num, c=cat, p=params: hgb_pipeline(n, c, **p)))
        return out

    print('Rolling-origin tuning (validate each of 2019-2024 on a model trained on earlier years) ...')
    specs = {
        'Team LR': ([({'C': 1.0}, lambda: TeamLR(TEAM_NUM, TEAM_CAT))], TEAM_NUM, TEAM_CAT),
        'Artist-history LR': ([({'C': 1.0}, lambda: TeamLR(['prior_top10s', 'prior_entries'], []))],
                              ['prior_top10s', 'prior_entries'], []),
        'Regularised LR (team features)': (lr_cands(TEAM_NUM, TEAM_CAT), TEAM_NUM, TEAM_CAT),
        'Regularised LR (extended)': (lr_cands(EXT_NUM, EXT_CAT), EXT_NUM, EXT_CAT),
        'HistGB (team features)': (hgb_cands(TEAM_NUM, TEAM_CAT), TEAM_NUM, TEAM_CAT),
        'HistGB (extended)': (hgb_cands(EXT_NUM, EXT_CAT), EXT_NUM, EXT_CAT),
    }
    tuned, logs = {}, []
    for name, (cands, num, cat) in specs.items():
        best, log = tune(train, name, cands, num, cat)
        tuned[name] = {'score': best[0], 'params': best[1], 'factory': best[2], 'val_pred': best[3],
                       'num': num, 'cat': cat}
        logs.append(log)
    pd.concat(logs).to_csv(os.path.join(RESULTS, 'tuning_log.csv'), index=False)

    # Ensemble: average of the two tuned extended models, then a Platt calibration fitted on
    # the pooled out-of-time validation predictions (2019-2024). The test years are not used.
    v = train['debut_year'].isin(VAL_YEARS).values
    ens_val = (tuned['Regularised LR (extended)']['val_pred'] + tuned['HistGB (extended)']['val_pred']) / 2
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    platt = LogisticRegression(C=1e6).fit(logit(ens_val[v].values).reshape(-1, 1), y_tr[v])
    ens_val_cal = pd.Series(np.nan, index=train.index)
    ens_val_cal[v] = platt.predict_proba(logit(ens_val[v].values).reshape(-1, 1))[:, 1]
    ens_mean, ens_years = fold_score(train, ens_val_cal)
    print(f'  Calibrated ensemble: mean validation PR-AUC {ens_mean:.3f}')

    # ---------------- thresholds: the team's 0.5 for its own model; for every other model
    # the F1-best threshold on the pooled 2019-2024 out-of-time predictions.
    thresholds = {'Team LR': 0.5, 'Artist-history LR': 0.5}
    for name in tuned:
        if name not in thresholds:
            thresholds[name] = best_f1_threshold(y_tr[v], tuned[name]['val_pred'][v].values)
    thresholds['Calibrated ensemble'] = best_f1_threshold(y_tr[v], ens_val_cal[v].values)

    # ---------------- final fit on 2018-2024, one evaluation on 2025-2026
    print('Final fit on 2018-2024; scoring 2025-2026 once ...')
    fitted, test_pred = {}, {}
    for name, t in tuned.items():
        m = t['factory']().fit(prepare(train, t['num'], t['cat']), y_tr)
        fitted[name] = m
        test_pred[name] = m.predict_proba(prepare(test, t['num'], t['cat']))[:, 1]
    fitted['Calibrated ensemble'] = Ensemble(fitted['Regularised LR (extended)'], fitted['HistGB (extended)'], platt)
    tuned['Calibrated ensemble'] = {'num': EXT_NUM, 'cat': EXT_CAT, 'params': {'components': 'tuned LR + tuned GB',
                                                                              'calibration': 'Platt on 2019-2024 OOT'}}
    test_pred['Calibrated ensemble'] = fitted['Calibrated ensemble'].predict_proba(prepare(test, EXT_NUM, EXT_CAT))[:, 1]

    # sanity: our TeamLR wrapper must give the team's published predictions
    tl = test_pred['Team LR'] >= 0.5
    assert tl.sum() == 35 and (tl & (y_te == 1)).sum() == 15, 'TeamLR wrapper differs from notebook'

    point = {m: metrics(y_te, p, thresholds[m]) for m, p in test_pred.items()}
    ci, diff = bootstrap(y_te, test_pred, thresholds)
    val_scores = {m: tuned[m]['score'] for m in tuned if m != 'Calibrated ensemble'}
    val_scores['Calibrated ensemble'] = ens_mean

    table = []
    for m in test_pred:
        row = {'model': m, 'validation_mean_pr_auc': val_scores[m], 'threshold': thresholds[m]}
        for k, val in point[m].items():
            row[k] = val
            if k in ci[m]:
                row[k + '_lo'], row[k + '_hi'] = ci[m][k]
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(os.path.join(RESULTS, 'heldout_results.csv'), index=False)
    print(table[['model', 'validation_mean_pr_auc', 'threshold', 'flagged', 'precision', 'recall', 'pr_auc',
                 'roc_auc', 'brier', f'p_at_{TOP_K}']].round(3).to_string(index=False))

    # ---------------- choose the model to explain: best mean validation PR-AUC (not test!)
    best_name = max(val_scores, key=val_scores.get)
    explain_name = best_name
    print(f'Best on validation: {best_name}. Explaining: {explain_name}')

    # ---------------- explanation on the held-out songs
    t = tuned[explain_name]
    X_te = prepare(test, t['num'], t['cat'])
    imp, base_ap = permutation_importance(fitted[explain_name], X_te, y_te, t['num'] + t['cat'])
    imp.to_csv(os.path.join(RESULTS, 'permutation_importance.csv'), index=False)
    print(imp.head(10).round(4).to_string(index=False))

    # partial dependence over the pooled train+test feature distribution, for the top numeric features
    X_all = prepare(pd.concat([train, test]), t['num'], t['cat'])
    top_num = [f for f in imp['feature'] if f in t['num']][:4]
    pdp = {}
    for col in top_num:
        vals = X_all[col].dropna()
        grid = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 25)))
        pdp[col] = {'grid': grid.tolist(), 'pd': partial_dependence(fitted[explain_name], X_all, col, grid).tolist()}
    # and each extended component's own PD for the same features, as a cross-check
    lr_name = 'Regularised LR (extended)'
    X_all_ext = prepare(pd.concat([train, test]), EXT_NUM, EXT_CAT)
    for col in top_num:
        if col in EXT_NUM:
            pdp[col]['pd_lr'] = partial_dependence(fitted[lr_name], X_all_ext, col, pdp[col]['grid']).tolist()
            pdp[col]['pd_gb'] = partial_dependence(fitted['HistGB (extended)'], X_all_ext, col,
                                                   pdp[col]['grid']).tolist()

    # calibration curves (held-out), 8 quantile bins
    calib = {}
    for m in ['Team LR', 'Regularised LR (extended)', 'HistGB (extended)', 'Calibrated ensemble']:
        p = test_pred[m]
        bins = pd.qcut(p, 8, duplicates='drop', labels=False)
        g = pd.DataFrame({'p': p, 'y': y_te, 'b': bins}).groupby('b')
        calib[m] = {'mean_pred': g['p'].mean().tolist(), 'obs_rate': g['y'].mean().tolist(),
                    'n': g.size().tolist()}

    # PR curves (held-out)
    from sklearn.metrics import precision_recall_curve
    pr = {}
    for m in ['Artist-history LR', 'Team LR', 'Regularised LR (extended)', 'HistGB (extended)', 'Calibrated ensemble']:
        pp, rr, _ = precision_recall_curve(y_te, test_pred[m])
        pr[m] = {'precision': pp.tolist(), 'recall': rr.tolist()}

    # tuned LR coefficients (standardised), for the README
    lr_m = fitted[lr_name]
    coef = pd.Series(lr_m.named_steps['lr'].coef_[0], index=lr_m.named_steps['pre'].get_feature_names_out())
    coef = coef[coef != 0].sort_values()
    coef.to_csv(os.path.join(RESULTS, 'lr_extended_coefficients.csv'), header=['coefficient'])

    summary = {
        'n_train': int(len(train)), 'hits_train': int(y_tr.sum()), 'n_test': int(len(test)), 'hits_test': int(y_te.sum()),
        'test_base_rate': float(y_te.mean()), 'top_k': TOP_K, 'n_boot': N_BOOT,
        'tuned_params': {m: tuned[m]['params'] for m in tuned},
        'thresholds': thresholds, 'validation_mean_pr_auc': val_scores,
        'validation_pr_auc_by_year': {**{m: fold_score(train, tuned[m]['val_pred'])[1] for m in tuned
                                         if m != 'Calibrated ensemble'},
                                      'Calibrated ensemble': ens_years},
        'paired_vs_team_lr': diff, 'best_on_validation': best_name, 'explained_model': explain_name,
        'permutation_base_pr_auc': base_ap, 'pdp': pdp, 'calibration': calib, 'pr_curves': pr,
    }
    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)

    # ---------------- song-level predictions (local only) and worked examples
    # Outcome columns are attached only here, for describing examples, never as features.
    outcome = extras['wide'][['title', 'artist', 'peak_position', 'weeks_on_chart']]
    pred = test[['title', 'artist', 'primary_artist', 'debut_date', 'debut_position', 'top10'] + EXT_NUM + EXT_CAT]
    pred = pred.merge(outcome, on=['title', 'artist'], how='left', validate='one_to_one')
    for m, p in test_pred.items():
        pred['p_' + m] = p
    pred.to_csv(os.path.join(OUTPUT, 'heldout_predictions.csv'), index=False)
    print('\nPaired differences vs Team LR (95% CI):')
    for m, d in diff.items():
        print(f"  {m:32s} PR-AUC {d['pr_auc']['lo']:+.3f}..{d['pr_auc']['hi']:+.3f} "
              f"(better in {d['pr_auc']['share_better'] * 100:.0f}% of resamples); "
              f"P@{TOP_K} {d[f'p_at_{TOP_K}']['lo']:+.3f}..{d[f'p_at_{TOP_K}']['hi']:+.3f}")


if __name__ == '__main__':
    main()
