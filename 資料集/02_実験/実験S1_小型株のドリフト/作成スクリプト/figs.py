"""
Figures of LLM-BLM_実験S1 (図/). Reads experiments/S1_smallcap_drift/results (summary.json, comparison.csv, by_month*.csv).

    python 資料集/02_実験/実験S1_小型株のドリフト/作成スクリプト/figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S1_smallcap_drift' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
BLUE, ORANGE, AQUA, GRAY, INK, INK2, VIOLET, YELLOW = ('#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e',
                                                       '#6250d6', '#eda100')
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
NAMES = {'dictionary': '辞書', 'finbert': 'FinBERT（学習なし）', 'finbert_ft': 'FinBERT（追加学習）',
         'chrono_ridge': 'ChronoGPT＋リッジ', 'chrono_mid_ridge': 'ChronoGPT 26 層目＋リッジ',
         'gpt': 'gpt-4o-mini（学習なし）', 'gpt_ridge': 'gpt-4o-mini（重みを学習）'}
COLORS = {'dictionary': GRAY, 'finbert': YELLOW, 'finbert_ft': AQUA, 'chrono_ridge': VIOLET, 'chrono_mid_ridge': '#9b8fe6',
          'gpt': BLUE, 'gpt_ridge': '#7fb0ea'}
TARGETS = {'abn_r_0_1': '2 日の\n反応', 'abn_drift_2_60': '2〜60 日の\nドリフト', 'abn_drift_2_21': '2〜21 日の\nドリフト'}


def fig_compare(c):
    """t of each model, for each target, in the two periods."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=False, width_ratios=[5, 7])
    for ax, (period, title) in zip(axes, [('2020-', '2020 年以降（gpt-4o-mini 以外）'),
                                          ('2024- (gpt)', '2024 年以降（gpt-4o-mini の期間）')]):
        d = c[(c.period == period) & c.t.notna()]
        models = [m for m in NAMES if m in set(d.model)]
        x = np.arange(len(TARGETS))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [d[(d.model == m) & (d.target == t)].t.mean() for t in TARGETS]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width * 0.95, color=COLORS[m], label=NAMES[m])
        ax.axhline(0, color=INK2, lw=0.8)
        for h in (1.96, -1.96):
            ax.axhline(h, color=INK2, lw=0.6, ls=':')
        ax.set_xticks(x, list(TARGETS.values()), fontsize=8.5)
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_ylabel('点数の係数の t')
        ax.grid(axis='x', visible=False)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / 'compare.pdf')


def fig_portfolio(name, out):
    df = pd.read_csv(RES / name)
    cost = 0.003
    top = (1 + df.ret_top - 2 * cost * df.turnover_top).cumprod()
    allw = (1 + df.ret_all - 2 * cost * df.turnover_all).cumprod()
    x = pd.to_datetime(df.month_end)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(x, top, color=BLUE, lw=1.8, label='点数の上位 20%（浮動株時価総額で加重）')
    ax.plot(x, allw, color=GRAY, lw=1.8, label='範囲の全社（浮動株時価総額で加重）')
    ax.set_ylabel('1 から始めた価値（費用 片道 0.3% の後）')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / out)


BLM_NAMES = {'band': '範囲の全社（浮動株時価総額で加重）', **{k: f'BLM：{v}' for k, v in NAMES.items()}}
BLM_COLORS = {'band': INK2, **COLORS}
FREQ_NAMES = {'quarter': '四半期ごと', 'half': '半年ごと', 'year': '1 年ごと'}


def fig_blm_paths(start, out):
    """Value from 1 at the end of the start year, one panel per rebalancing frequency (addendum 1)."""
    df = pd.read_csv(RES / 'blm_paths.csv')
    df = df[df.start == start]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    for ax, (freq, title) in zip(axes, FREQ_NAMES.items()):
        d = df[df.freq == freq]
        for m in [m for m in BLM_NAMES if m in set(d.model)]:
            s = d[d.model == m].sort_values('month')
            x = pd.to_datetime(s.month)
            ax.plot(x, s.value, color=BLM_COLORS[m], lw=1.9 if m == 'band' else 1.3,
                    ls='--' if m == 'band' else '-', label=BLM_NAMES[m])
        ax.axhline(1, color=GRAY, lw=0.7)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', labelsize=7.5, rotation=30)
    axes[0].set_ylabel(f'{start} 年末を 1 とした価値')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=7.8)
    fig.tight_layout(rect=(0, 0.14 if len(labels) > 3 else 0.1, 1, 1))
    fig.savefig(OUT / out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(RES / 'comparison.csv')
    fig_compare(c)
    fig_portfolio('by_month.csv', 'portfolio.pdf')
    if (RES / 'by_month_gpt.csv').exists():
        fig_portfolio('by_month_gpt.csv', 'portfolio_gpt.pdf')
    if (RES / 'blm_paths.csv').exists():
        fig_blm_paths(2015, 'blm_2015.pdf')
        fig_blm_paths(2023, 'blm_2023.pdf')
    print('saved to', OUT)


if __name__ == '__main__':
    main()
