"""
Figures of LLM-BLM_実験G (図/): G1's shuffled views against the real BLM, G5's costs, the active return by year, G6.
Reads experiments/G_proof/results/ (summary.json, contrib_summary.json and the daily series of the shuffles, which are
outside Git and rebuilt by experiments/G_proof/run.py).

    python 資料集/02_実験/実験G_BLMが効くことの確かめ/作成スクリプト/figs.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'G_proof' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
GRAY, INK2, VIOLET, ORANGE, AQUA = '#b9b8b2', '#52514e', '#6250d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
SAMPLES = (('orig', '元の 1,000 社'), ('rep', '新しい 1,000 社'))
COST = 0.003


def fig_g1():
    summary = json.loads((RES / 'summary.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
    for ax, (s, title) in zip(axes, SAMPLES):
        sh = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / s / 'shuffle_*.csv')))
        sh = sh[sh.date >= '2017-01-01']
        fake = (sh.active - COST * (sh.traded - sh.traded_band)).groupby(sh.seed).mean() * 252 * 100
        real = summary[s]['G1_shuffled_views'][f'cost_{COST}']['real_annual'] * 100
        ax.hist(fake, bins=20, color=GRAY, edgecolor='white', linewidth=1.0, label='偽のビュー（100 回）')
        ax.axvline(real, color=VIOLET, lw=2, label='本物の点数')
        ax.annotate(f'本物 +{real:.1f}%', (real, ax.get_ylim()[1] * 0.85), xytext=(-6, 0), textcoords='offset points',
                    ha='right', color=INK2, fontsize=8.5)
        ax.axvline(0, color=INK2, lw=0.7, ls=':')
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xlabel('比べる相手との差（年 %、費用 0.3% の後）')
    axes[0].set_ylabel('回数')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'g1.pdf')


def fig_g5():
    summary = json.loads((RES / 'summary.json').read_text(encoding='utf-8'))
    costs = [0.0, 0.003, 0.005, 0.01]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for k, ((s, name), color, marker) in enumerate(zip(SAMPLES, (GRAY, VIOLET), ('s', 'o'))):
        t = [summary[s]['G5_costs'][f'cost_{c}'] for c in costs]
        x = [c * 100 + (k - 0.5) * 0.03 for c in costs]
        y = [v['annual_excess'] * 100 for v in t]
        lo = [y[i] - v['ci95_annual'][0] * 100 for i, v in enumerate(t)]
        hi = [v['ci95_annual'][1] * 100 - y[i] for i, v in enumerate(t)]
        ax.errorbar(x, y, yerr=[lo, hi], color=color, marker=marker, markersize=6, lw=2, capsize=3, label=name)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([c * 100 for c in costs], ['0%（これまでの値）', '0.3%', '0.5%', '1.0%'])
    ax.set_xlabel('片道の費用')
    ax.set_ylabel('比べる相手との差（年 %）')
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / 'g5.pdf')


def fig_years():
    c = json.loads((RES / 'contrib_summary.json').read_text(encoding='utf-8'))
    years = list(c['orig']['by_year'])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for k, ((s, name), color) in enumerate(zip(SAMPLES, (GRAY, VIOLET))):
        ax.bar([i + (k - 0.5) * 0.38 for i in range(len(years))], [c[s]['by_year'][y] for y in years], width=0.36,
               color=color, label=name)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(years)), years)
    ax.set_ylabel('比べる相手との差（年 %、費用の前）')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'years.pdf')


def fig_g6():
    c = json.loads((RES / 'contrib_summary.json').read_text(encoding='utf-8'))
    items = (('blm', 'BLM', VIOLET, 'o'), ('top_equal', '上位 20%・等金額', ORANGE, '^'),
             ('top_float', '上位 20%・時価総額加重', AQUA, 's'))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True, sharex=True)
    for ax, (s, title) in zip(axes, SAMPLES):
        for key, name, color, marker in items:
            g = c[s]['G6'][f'{key}_cost{COST}']
            ax.scatter(g['te_realized'], g['annual_excess'], color=color, marker=marker, s=70, zorder=3,
                       edgecolor='white', linewidth=1.5, label=name)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(3, color=INK2, lw=0.7, ls=':')
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xlabel('実現の TE（年 %）。点線は約束の 3%')
        ax.set_xlim(0, 8)
    axes[0].set_ylabel('比べる相手との差（年 %、費用 0.3% の後）')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'g6.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_g1()
    fig_g5()
    fig_years()
    fig_g6()
    print('figures written to', OUT)
