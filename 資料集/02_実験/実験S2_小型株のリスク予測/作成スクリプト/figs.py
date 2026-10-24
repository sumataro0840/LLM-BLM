"""
Figures of LLM-BLM_実験S2 (図/). Reads experiments/S2_smallcap_risk/results (comparison.csv, forecasts.csv).

    python 資料集/02_実験/実験S2_小型株のリスク予測/作成スクリプト/figs.py [--dry OUTDIR]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S2_smallcap_risk' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
BLUE, ORANGE, AQUA, GRAY, INK2, VIOLET, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#52514e', '#6250d6', '#eda100'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
NAMES = {'chrono_risk': 'ChronoGPT＋リッジ（主）', 'chrono_mid_risk': 'ChronoGPT 26 層目＋リッジ',
         'dict_uncertainty': '辞書：不確実さ', 'dict_negative': '辞書：否定的', 'finbert_negative': 'FinBERT：否定的',
         'gpt_flags': 'gpt-4o-mini：リスクの印', 'gpt_risk': 'gpt-4o-mini：14 項目＋リッジ'}
COLORS = {'chrono_risk': VIOLET, 'chrono_mid_risk': '#9b8fe6', 'dict_uncertainty': GRAY, 'dict_negative': '#8c8b86',
          'finbert_negative': YELLOW, 'gpt_flags': BLUE, 'gpt_risk': '#7fb0ea'}
TARGETS = {'log_rv_2_60': '変動率\n2〜60 日', 'log_rv_2_21': '変動率\n2〜21 日', 'log_ivol_2_60': '固有の変動率\n2〜60 日',
           'abs_r_0_1': '2 日の反応\nの大きさ'}


def fig_compare(c, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), width_ratios=[5, 7])
    for ax, (period, title) in zip(axes, [('2020-', '2020 年以降（gpt-4o-mini 以外）'),
                                          ('2024- (gpt)', '2024 年以降（gpt-4o-mini の期間）')]):
        d = c[(c.period == period) & c.t.notna()]
        models = [m for m in NAMES if m in set(d.model)]
        x = np.arange(len(TARGETS))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals = [d[(d.model == m) & (d.target == t)].t.mean() for t in TARGETS]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width=width * 0.95, color=COLORS[m], label=NAMES[m])
        ax.axhline(0, color=INK2, lw=0.8)
        for h in (1.96, -1.96):
            ax.axhline(h, color=INK2, lw=0.6, ls=':')
        ax.set_xticks(x, list(TARGETS.values()), fontsize=8)
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
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / 'compare.pdf')


def fig_forecast(fc, out):
    """Realized log volatility against the out-of-sample prediction, in deciles of each prediction (descriptive)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for col, name, color in (('pred_base', '基準のモデル', GRAY), ('pred_with_score', '基準＋ChronoGPT＋リッジ', VIOLET)):
        q = pd.qcut(fc[col].rank(method='first'), 10, labels=False)
        g = fc.groupby(q)
        ax.plot(np.exp(g[col].mean()) * 100, np.exp(g.log_rv_2_60.mean()) * 100, marker='o', ms=4, lw=1.4, color=color,
                label=name)
    lim = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lim, lim, color=INK2, lw=0.7, ls=':')
    ax.set_xlabel('予測した変動率（年率 %、10 分位の平均）')
    ax.set_ylabel('実現の変動率（年率 %）')
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / 'forecast.pdf')


def main():
    dry = '--dry' in sys.argv
    res = RES / 'dry' if dry else RES
    out = Path(sys.argv[sys.argv.index('--dry') + 1]) if dry else OUT
    out.mkdir(parents=True, exist_ok=True)
    fig_compare(pd.read_csv(res / 'comparison.csv'), out)
    fig_forecast(pd.read_csv(res / 'forecasts.csv'), out)
    print('saved to', out)


if __name__ == '__main__':
    main()
