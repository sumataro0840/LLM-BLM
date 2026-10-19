"""
Figures of LLM-BLM_補足実験_αの大きさ (図/). Reads experiments/X05_alpha_scale/results/summary.csv (k = 0.1 .. 2) and
results/small/summary.csv (k = 0.001 .. 0.03).

    python 資料集/02_実験/補足実験_αの大きさ/作成スクリプト/figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
OUT = Path(__file__).resolve().parents[1] / '図'
RES = REPO / 'experiments' / 'X05_alpha_scale' / 'results'
BLUE, ORANGE, AQUA, GRAY, INK, INK2, VIOLET, YELLOW = ('#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e',
                                                       '#6250d6', '#eda100')
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
NAMES = {'gpt4o_mini': 'gpt-4o-mini', 'chrono_ridge': 'ChronoGPT リッジ', 'finbert_ft': 'FinBERT 追加学習',
         'tfidf': 'TF-IDF リッジ', 'finbert': 'FinBERT', 'dictionary': '辞書'}
COLORS = {'gpt4o_mini': BLUE, 'chrono_ridge': VIOLET, 'finbert_ft': AQUA, 'tfidf': ORANGE, 'finbert': YELLOW,
          'dictionary': GRAY}


def table():
    t = pd.concat([pd.read_csv(RES / 'summary.csv'), pd.read_csv(RES / 'small' / 'summary.csv')])
    return t.sort_values(['window', 'signal', 'k'])


def fig_scale(t):
    """Monthly excess return after costs and the share of months with a binding TE limit, against the multiplier."""
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True, height_ratios=[1.5, 1])
    for col, (window, title) in enumerate((('A', '期間 A（2024-01〜2026-07）'), ('L', '期間 L（2016-03〜2026-07）'))):
        ax, axb = axes[0, col], axes[1, col]
        for signal, g in t[t.window == window].groupby('signal'):
            ax.plot(g.k, 100 * g.active_net_monthly, marker='o', ms=3, lw=1.4, color=COLORS[signal], label=NAMES[signal])
            axb.plot(g.k, 100 * g.share_binding, marker='o', ms=3, lw=1.2, color=COLORS[signal])
        for a in (ax, axb):
            a.set_xscale('log')
            a.axvline(1, color=INK2, lw=0.8, ls=':')
            a.axvline(1 / np.sqrt(21), color=INK2, lw=0.8, ls='--')
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=9.5, color=INK2)
        axb.set_xlabel('α の倍率 k（点線 1：実験 7・9、破線：1/√21）')
        axb.set_ylim(-5, 105)
    axes[0, 0].set_ylabel('超過リターン（費用の後、%／月）')
    axes[1, 0].set_ylabel('TE の上限が効いた月（%）')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / 'scale.pdf')


def fig_signals(t):
    """At k = 1 (equal to every k at which the limit binds): monthly excess return after costs and its t value."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), width_ratios=[1.5, 1])
    for ax, window, title in ((axes[0], 'A', '期間 A（31 か月）'), (axes[1], 'L', '期間 L（125 か月）')):
        g = t[(t.window == window) & np.isclose(t.k, 1.0)].set_index('signal')
        order = [s for s in NAMES if s in g.index]
        vals = [100 * g.loc[s, 'active_net_monthly'] for s in order]
        se = [abs(v / g.loc[s, 'nw_t']) if g.loc[s, 'nw_t'] != 0 else 0 for v, s in zip(vals, order)]
        ax.bar(range(len(order)), vals, color=[COLORS[s] for s in order], width=0.6)
        ax.errorbar(range(len(order)), vals, yerr=[1.96 * e for e in se], fmt='none', ecolor=INK, capsize=3, lw=1)
        for i, s in enumerate(order):
            ax.text(i, -0.62, f't {g.loc[s, "nw_t"]:+.1f}', ha='center', fontsize=7.5, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        short = {'gpt4o_mini': 'gpt-4o\nmini', 'chrono_ridge': 'Chrono\nリッジ', 'finbert_ft': 'FinBERT\n追加学習',
                 'tfidf': 'TF-IDF\nリッジ', 'finbert': 'FinBERT', 'dictionary': '辞書'}
        ax.set_xticks(range(len(order)), [short[s] for s in order], fontsize=8)
        ax.set_ylim(-0.7, 0.55)
        ax.set_title(title, fontsize=9.5, color=INK2)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('超過リターン（費用の後、%／月、95% 区間）')
    fig.tight_layout()
    fig.savefig(OUT / 'signals.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    t = table()
    fig_scale(t)
    fig_signals(t)
    print('saved to', OUT)
