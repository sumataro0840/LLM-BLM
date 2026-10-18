"""
Figures of LLM-BLM_補足実験_共分散とTE (図/). Reads the results of experiments/X02_covariance_window,
X03_te_calibration and X04_vix_scaling (summary.json and the by-date CSV files).

    python 資料集/02_実験/補足実験_共分散とTE/作成スクリプト/figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
OUT = Path(__file__).resolve().parents[1] / '図'
EXP = REPO / 'experiments'
BLUE, ORANGE, AQUA, GRAY, INK, INK2, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e', '#6250d6'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})


def load(name):
    return json.loads((EXP / name / 'results' / 'summary.json').read_text(encoding='utf-8'))


def fig_x02_te(s):
    names = ['lw63', 'lw126', 'lw252', 'lw504', 'ewma63', 'ewma126']
    vals = [100 * s['active_te'][n]['mean'] for n in names]
    colors = [BLUE if n == 'lw252' else (AQUA if n == 'ewma126' else GRAY) for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.bar(range(6), vals, color=colors, width=0.6)
    for b, v, n in zip(bars, vals, names):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f'{v:.2f}%\n(×{s["active_te_ratio"][n]:.2f})',
                ha='center', fontsize=8, color=INK)
    ax.axhline(3, color=ORANGE, lw=1.2, ls='--')
    ax.text(-0.45, 3.1, '目標 3%', color=ORANGE, fontsize=8.5, ha='left', va='bottom')
    ax.set_xticks(range(6), ['63 日', '126 日', '252 日\n（今）', '504 日', '指数の重み\n半減期 63', '指数の重み\n半減期 126'])
    ax.set_ylabel('実現の TE（年率、%）')
    ax.set_ylim(0, 6.6)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'x02_te.pdf')


def fig_x03_series():
    d = pd.read_csv(EXP / 'X03_te_calibration' / 'results' / 'by_date.csv')
    d = d[d.estimator == 'lw252']
    x = pd.to_datetime(d.date)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(x, 100 * d.te_uncal, color=GRAY, lw=1.3, label='校正なし（上限 3%）')
    ax.plot(x, 100 * d.te, color=BLUE, lw=1.5, label='校正あり（上限 3% ÷ 過去 36 か月の比）')
    ax.axhline(3, color=ORANGE, lw=1.2, ls='--')
    ax.set_ylabel('実現の TE（年率、%）')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / 'x03_series.pdf')


def fig_x04_rmse(s):
    runs = [('x03_uncal', '校正なし'), ('x03_cal', '校正（X03）'), ('vix', 'VIX だけ'), ('vix_cal', 'VIX ＋ 校正')]
    colors = [GRAY, BLUE, ORANGE, VIOLET]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    w = 0.2
    for i, ((key, label), c) in enumerate(zip(runs, colors)):
        vals = [100 * s[e]['rmse_miss'][key] for e in ('lw252', 'ewma126')]
        bars = ax.bar(np.arange(2) + (i - 1.5) * w, vals, w, color=c, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.2f}', ha='center', fontsize=7.5, color=INK)
    ax.set_xticks(range(2), ['lw252（今）', 'ewma126'])
    ax.set_ylabel('3% からのずれ（RMSE、ポイント）')
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    ax.set_ylim(0, 2.6)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'x04_rmse.pdf')


def fig_x04_2020():
    c = pd.read_csv(EXP / 'X04_vix_scaling' / 'results' / 'by_date_vix_cal.csv')
    a = pd.read_csv(EXP / 'X03_te_calibration' / 'results' / 'by_date.csv')
    e = 'ewma126'
    c, a = c[c.estimator == e].set_index('date'), a[a.estimator == e].set_index('date')
    x = pd.to_datetime(c.index)
    fig, ax1 = plt.subplots(figsize=(7.0, 2.8))
    ax1.plot(x, 100 * a.loc[c.index].te, color=BLUE, lw=1.4, label='校正（X03）')
    ax1.plot(x, 100 * c.te, color=VIOLET, lw=1.4, label='VIX ＋ 校正（X04）')
    ax1.axhline(3, color=ORANGE, lw=1.2, ls='--')
    ax1.set_ylabel('実現の TE（年率、%）')
    ax2 = ax1.twinx()
    ax2.fill_between(x, c.vix, color=GRAY, alpha=0.25, step='mid', label='VIX（右軸）')
    ax2.set_ylabel('VIX', color=INK2)
    ax2.grid(False)
    ax2.spines['right'].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc='upper right')
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'x04_series.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_x02_te(load('X02_covariance_window'))
    fig_x03_series()
    fig_x04_rmse(load('X04_vix_scaling'))
    fig_x04_2020()
    print('saved to', OUT)
