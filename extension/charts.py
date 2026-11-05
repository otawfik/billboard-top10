"""Draw the extension's four charts from extension/results/ (aggregate numbers only).

    python extension/charts.py

No lyrics and no song-level rows go into any chart.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
CHARTS = os.path.join(HERE, 'charts')
os.makedirs(CHARTS, exist_ok=True)

# One palette for every chart: highlight = the best model, accent = the team's model,
# everything else recedes to gray. Text stays in ink, never in series colour.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'    # best model (slot 1)
ORANGE = '#eb6834'  # team model (slot 2)
GRAY = '#b5b3ab'    # other models
AQUA = '#1baf7a'    # second component line (slot 3)

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10.5, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
})

LABELS = {
    'Artist-history LR': 'Artist-history baseline (team)',
    'Team LR': 'Team logistic regression',
    'Regularised LR (team features)': 'Regularised LR, team features',
    'Regularised LR (extended)': 'Regularised LR, extended features',
    'HistGB (team features)': 'Gradient boosting, team features',
    'HistGB (extended)': 'Gradient boosting, extended features',
    'Calibrated ensemble': 'Calibrated ensemble (LR + GB)',
}
FEATURE_LABELS = {
    'prior_top10s': "Lead artist's earlier Top 10s",
    'prior_entries': "Lead artist's earlier entries",
    'prior_chart_weeks': "Lead artist's earlier chart weeks",
    'artist_weeks_last_52': 'Artist chart weeks, past year',
    'artist_songs_prev_week': "Artist's songs on last week's chart",
    'prior_best_peak': "Artist's best earlier peak",
    'weeks_since_last_chart': 'Weeks since artist last charted',
    'weeks_since_last_top10': 'Weeks since artist last in Top 10',
    'prior_top10_rate': "Artist's earlier hit rate (shrunk)",
    'feat_prior_top10s': "Featured artist's earlier Top 10s",
    'same_artist_debuts': 'Songs the artist debuted that week',
    'week_new_entries': 'All new entries that week',
    'release_format': 'Release format (single / album / EP)',
    'lead_country': 'Lead artist US / non-US',
    'primary_genre': 'Genre',
    'album_debut_bucket': 'Drop-size group (team)',
    'n_credited_artists': 'Number of credited artists',
    'has_feature': 'Has a featured artist',
    'duration_used': 'Song length',
    'days_release_to_chart': 'Days from release to debut',
    'lyric measures (5)': 'Lyric measures (5, shuffled together)',
}
PD_UNITS = {
    'weeks_since_last_top10': 'weeks (520 = none in 10 years)',
    'weeks_since_last_chart': 'weeks (520 = none in 10 years)',
    'same_artist_debuts': 'songs debuting that week',
    'prior_best_peak': 'chart position (101 = never charted)',
    'prior_top10_rate': 'shrunk hit rate',
    'artist_weeks_last_52': 'chart weeks in the past year',
    'prior_chart_weeks': 'chart weeks',
    'week_new_entries': 'new entries that week',
    'prior_top10s': 'earlier Top 10s',
    'prior_entries': 'earlier chart entries',
    'feat_prior_top10s': 'earlier Top 10s',
    'artist_songs_prev_week': 'songs',
    'days_release_to_chart': 'days',
    'duration_used': 'seconds',
}


def colour(model, best):
    if model == best:
        return BLUE
    if model == 'Team LR':
        return ORANGE
    if model == 'HistGB (extended)':
        return AQUA
    return GRAY


def save(fig, name):
    path = os.path.join(CHARTS, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', pil_kwargs={'quality': 92})
    plt.close(fig)
    print('wrote', os.path.relpath(path, os.path.dirname(HERE)))


def main():
    s = json.load(open(os.path.join(RESULTS, 'summary.json')))
    res = pd.read_csv(os.path.join(RESULTS, 'heldout_results.csv'))
    imp = pd.read_csv(os.path.join(RESULTS, 'permutation_importance.csv'))
    best = s['best_on_validation']
    explained = s['explained_model']
    k = s['top_k']
    base = s['test_base_rate']

    # ---------------- 1. held-out comparison with bootstrap CIs
    order = ['Artist-history LR', 'Team LR', 'Regularised LR (team features)', 'HistGB (team features)',
             'Regularised LR (extended)', 'HistGB (extended)', 'Calibrated ensemble']
    r = res.set_index('model').loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ypos = np.arange(len(order))[::-1]
    for ax, metric, title in [(axes[0], 'pr_auc', 'PR-AUC (average precision)'),
                              (axes[1], f'p_at_{k}', f'Precision in the top {k} flagged songs')]:
        for yi, m in zip(ypos, order):
            c = colour(m, best)
            ax.plot([r.loc[m, metric + '_lo'], r.loc[m, metric + '_hi']], [yi, yi], color=c, lw=2,
                    solid_capstyle='round')
            ax.scatter(r.loc[m, metric], yi, s=64, color=c, zorder=3, edgecolor=SURFACE, linewidth=2)
            ax.text(r.loc[m, metric + '_hi'] + 0.012, yi, f'{r.loc[m, metric]:.0%}' if metric != 'pr_auc'
                    else f'{r.loc[m, metric]:.2f}', va='center', fontsize=9.5, color=INK_2)
        ax.axvline(base, color=MUTED, lw=1)
        ax.text(base + 0.008, ypos[-1] - 0.45, f'random pick {base:.1%}', fontsize=8.5, color=MUTED, va='center')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, max(0.85, r[metric + '_hi'].max() + 0.1))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)
                                     if metric != 'pr_auc' else matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax.grid(axis='y', visible=False)
        ax.set_ylim(-0.85, len(order) - 0.4)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([LABELS[m] for m in order])
    for lab, m in zip(axes[0].get_yticklabels(), order):
        lab.set_color(INK if m in (best, 'Team LR') else INK_2)
        lab.set_fontweight('bold' if m in (best, 'Team LR') else 'normal')
    fig.suptitle(f'Held-out 2025-2026 songs ({s["n_test"]} songs, {s["hits_test"]} Top 10 hits): '
                 'dot = score, line = bootstrap 95% CI', x=0.0, ha='left', fontsize=12.5, fontweight='bold')
    fig.text(0.0, -0.03, 'Every model was tuned only on 2018-2024 (rolling origin) and scored on 2025-2026 once. '
             'Blue = best on validation (the pre-chosen model); orange = the team model; green = best single model on held-out.', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'chart-heldout-comparison.jpg')

    # ---------------- 2. PR curves + calibration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    curve_models = ['Artist-history LR', 'Team LR', 'HistGB (extended)', best]
    for m in curve_models:
        c = colour(m, best) if m != 'Artist-history LR' else MUTED
        pr = s['pr_curves'][m]
        a1.plot(pr['recall'], pr['precision'], color=c, lw=2, drawstyle='steps-post', label=LABELS[m])
    a1.axhline(base, color=MUTED, lw=1)
    a1.text(0.62, base - 0.035, f'random pick {base:.1%}', ha='left', fontsize=8.5, color=MUTED)
    a1.set_xlabel('Recall: share of real hits caught')
    a1.set_ylabel('Precision: share of flags that became hits')
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.02)
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.legend(loc='upper right', fontsize=9)
    a1.set_title('Precision-recall on the held-out songs')

    calib_models = ['Team LR', best] if best in s['calibration'] else ['Team LR', explained]
    lim = 0
    for m in calib_models:
        cdat = s['calibration'][m]
        c = colour(m, best)
        a2.plot(cdat['mean_pred'], cdat['obs_rate'], color=c, lw=2, marker='o', ms=7, mec=SURFACE, mew=2,
                label=LABELS[m])
        lim = max(lim, max(cdat['mean_pred']), max(cdat['obs_rate']))
    lim = min(1, lim * 1.08)
    a2.plot([0, lim], [0, lim], color=AXIS, lw=1)
    a2.text(lim * 0.97, lim * 0.9, 'perfect calibration', ha='right', fontsize=8.5, color=MUTED, rotation=0)
    a2.set_xlim(0, lim)
    a2.set_ylim(0, lim)
    a2.set_xlabel('Predicted chance of a Top 10 (bin average)')
    a2.set_ylabel('Share that reached the Top 10')
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.legend(loc='upper left', fontsize=9, handlelength=3)
    a2.set_title('Calibration: 8 equal-size bins of held-out songs')
    fig.tight_layout()
    save(fig, 'chart-pr-calibration.jpg')

    # ---------------- 3. permutation importance (held-out)
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    yy = np.arange(len(top))
    ax.barh(yy, top['pr_auc_drop'], color=BLUE, height=0.62)
    ax.errorbar(top['pr_auc_drop'], yy,
                xerr=[top['pr_auc_drop'] - top['pr_auc_drop_lo'], top['pr_auc_drop_hi'] - top['pr_auc_drop']],
                fmt='none', ecolor=INK_2, elinewidth=1, capsize=0)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in top['feature']])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'Drop in held-out PR-AUC when the feature is shuffled (base {s["permutation_base_pr_auc"]:.2f})')
    ax.set_title(f'What the model leans on: {LABELS[explained]}')
    fig.text(0.0, -0.02, 'Bars: mean of 50 shuffles of the 2025-2026 songs; lines: middle 95% of shuffles. '
             'Correlated features share credit, so each bar is a lower bound on that feature\'s role.',
             fontsize=8.3, color=MUTED, ha='left')
    fig.tight_layout()
    save(fig, 'chart-permutation-importance.jpg')

    # ---------------- 4. partial dependence (small multiples)
    feats = list(s['pdp'])[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for ax, f in zip(axes.ravel(), feats):
        d = s['pdp'][f]
        g = np.array(d['grid'])
        if 'pd_lr' in d:
            ax.plot(g, d['pd_lr'], color=GRAY, lw=1.5, label='its regularised-LR part')
        if 'pd_gb' in d:
            ax.plot(g, d['pd_gb'], color=AQUA, lw=1.5, label='its gradient-boosting part')
        ax.plot(g, d['pd'], color=BLUE, lw=2.2, label=LABELS[explained])
        ax.set_title(FEATURE_LABELS.get(f, f), fontsize=11)
        ax.set_xlabel(PD_UNITS.get(f, ''), fontsize=9)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, max(max(d['pd']), max(d.get('pd_lr', [0])), max(d.get('pd_gb', [0]))) * 1.15)
    axes[0, 0].set_ylabel('Average predicted chance')
    axes[1, 0].set_ylabel('Average predicted chance')
    axes[0, 0].legend(loc='best', fontsize=9)
    fig.suptitle('Partial dependence: average predicted Top 10 chance as one feature changes, all else as observed',
                 x=0.0, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'chart-partial-dependence.jpg')


if __name__ == '__main__':
    main()
