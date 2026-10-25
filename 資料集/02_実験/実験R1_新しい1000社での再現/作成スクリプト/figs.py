"""
Figures of LLM-BLM_実験R1 (図/): the calibrated BLM against the band in the original and the new sample, and the
primary coefficients of S1 and S2 in both. Reads experiments/S4_te_calibration/results{,_rep}/paths.csv and the
summary.json of S1/S2.

    python 資料集/02_実験/実験R1_新しい1000社での再現/作成スクリプト/figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
EXP = REPO / 'experiments'
OUT = Path(__file__).resolve().parents[1] / '図'
GRAY, INK2, VIOLET, ORANGE, BLUE, AQUA = '#b9b8b2', '#52514e', '#6250d6', '#eb6834', '#2a78d6', '#1baf7a'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})


def fig_paths():
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    for ax, (sub, title) in zip(axes, (('results', '元の標本（S4）'), ('results_rep', '新しい標本（R1）'))):
        df = pd.read_csv(EXP / 'S4_te_calibration' / sub / 'paths.csv', keep_default_na=False)
        d = df[(df.start == 2015) & (df.freq == 'quarter')]
        band = d[d.model == 'band'].sort_values('month')
        blm = d[(d.model == 'chrono_ridge') & (d.version == 'B_cal')].sort_values('month')
        ax.plot(pd.to_datetime(band.month), band.value, color=INK2, lw=1.6, ls='--', label='範囲の全社（比べる相手）')
        ax.plot(pd.to_datetime(blm.month), blm.value, color=VIOLET, lw=1.6, label='BLM：ChronoGPT＋B＋校正')
        ax.axhline(1, color=GRAY, lw=0.7)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', labelsize=7.5, rotation=30)
    axes[0].set_ylabel('2015 年末を 1 とした価値')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / 'paths.pdf')


def fig_coef():
    """t of the primary score of S1 (drift) and S2 (volatility), and gpt-4o-mini's drift, in both samples."""
    items = [('S1：ドリフト\n（ChronoGPT）', 'S1_smallcap_drift', lambda s: s['primary']['t']),
             ('S1：ドリフト\n（gpt-4o-mini）', 'S1_smallcap_drift', lambda s: s['secondary_drift_gpt_period']['gpt']['t']),
             ('S2：変動率\n（ChronoGPT）', 'S2_smallcap_risk', lambda s: s['primary']['t'])]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for k, (sub, name, color) in enumerate((('results', '元の標本', GRAY), ('results_rep', '新しい標本', VIOLET))):
        vals = [get(json.loads((EXP / d / sub / 'summary.json').read_text(encoding='utf-8'))) for _, d, get in items]
        ax.bar([i + (k - 0.5) * 0.38 for i in range(len(items))], vals, width=0.36, color=color, label=name)
    ax.axhline(1.96, color=INK2, lw=0.7, ls=':')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=8.5)
    ax.set_ylabel('主な点数の係数の t')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'coef.pdf')


def fig_active():
    """Daily active return of the primary portfolio (S4's u.csv x 3%/sqrt(252)) in both samples, from 2017."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8), sharey=True)
    for ax, (sub, title) in zip(axes, (('results', '元の標本'), ('results_rep', '新しい標本'))):
        u = pd.read_csv(EXP / 'S4_te_calibration' / sub / 'u.csv')
        s = u[(u.key == '2015/quarter/chrono_ridge/B_cal') & (u.date >= '2017-01-01')].set_index('date').u
        a = s * 0.03 / 252 ** 0.5 * 100
        lim = 3.0
        ax.plot(pd.to_datetime(a.index), a.clip(-lim, lim).values, color=VIOLET, lw=0.5)
        for d, x in a[a.abs() > lim].items():          # days off the scale: marked at the edge with their value
            y = lim if x > 0 else -lim
            ax.plot(pd.Timestamp(d), y, marker='v' if x > 0 else '^', color=ORANGE, ms=5)
            ax.annotate(f'{d}  {x:+.0f}%', (pd.Timestamp(d), y), xytext=(6, -2 if x > 0 else 2),
                        textcoords='offset points', fontsize=7.5, color=ORANGE, va='top' if x > 0 else 'bottom')
        ax.set_ylim(-lim * 1.08, lim * 1.08)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', labelsize=7.5, rotation=30)
    axes[0].set_ylabel('日々の差（BLM − 相手、%）')
    fig.tight_layout()
    fig.savefig(OUT / 'active.pdf')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_paths()
    fig_coef()
    fig_active()
    print('saved to', OUT)


if __name__ == '__main__':
    main()
