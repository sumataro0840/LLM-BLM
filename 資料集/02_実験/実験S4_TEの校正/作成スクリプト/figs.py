"""
Figures of LLM-BLM_実験S4 (図/). Reads experiments/S4_te_calibration/results (paths.csv, limits.csv, u.csv).

    python 資料集/02_実験/実験S4_TEの校正/作成スクリプト/figs.py [--dry OUTDIR]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S4_te_calibration' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
GRAY, INK2, VIOLET, AQUA, ORANGE, BLUE = '#b9b8b2', '#52514e', '#6250d6', '#1baf7a', '#eb6834', '#2a78d6'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
VERSIONS = {'A': ('A 基準', '#9b8fe6', '--'), 'A_cal': ('A＋校正', VIOLET, '-'), 'B': ('B 株価で補正', '#7fd3b4', '--'),
            'B_cal': ('B＋校正', AQUA, '-')}
FREQ = {'quarter': '四半期ごと', 'half': '半年ごと', 'year': '1 年ごと'}


def fig_paths(df, start, model, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    for ax, (freq, title) in zip(axes, FREQ.items()):
        d = df[(df.start == int(start)) & (df.freq == freq)]
        band = d[d.model == 'band'].sort_values('month')
        ax.plot(pd.to_datetime(band.month), band.value, color=INK2, lw=1.8, ls=':', label='範囲の全社（比べる相手）')
        for ver, (name, color, ls) in VERSIONS.items():
            s = d[(d.model == model) & (d.version == ver)].sort_values('month')
            if len(s):
                ax.plot(pd.to_datetime(s.month), s.value, color=color, lw=1.3, ls=ls, label=f'BLM：{name}')
        ax.axhline(1, color=GRAY, lw=0.7)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', labelsize=7.5, rotation=30)
    axes[0].set_ylabel(f'{start} 年末を 1 とした価値')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / f'paths_{start}_{model}.pdf')


def fig_te(u, limits, out):
    """Rolling one-year realized TE of the active return (u x 3%) and the calibrated limit (2015 end, quarterly)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for ver in ('B', 'B_cal'):
        key = f'2015/quarter/chrono_ridge/{ver}'
        s = u[u.key == key].set_index('date').u.sort_index()
        s.index = pd.to_datetime(s.index)
        roll = (s * 0.03).rolling(252).std()
        name, color, ls = VERSIONS[ver]
        ax.plot(roll.index, roll * 100, color=color, lw=1.4, ls=ls, label=f'実現の TE（直近 1 年）：{name}')
    lim = limits[limits.key == '2015/quarter/chrono_ridge/B_cal']
    ax.step(pd.to_datetime(lim.date), lim['limit'] * 100, where='post', color=ORANGE, lw=1.2, label='校正した事前の上限')
    ax.axhline(3, color=INK2, lw=0.8, ls=':')
    ax.set_ylabel('年率 %')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'te.pdf')


def main():
    dry = '--dry' in sys.argv
    res = RES / 'dry' if dry else RES
    out = Path(sys.argv[sys.argv.index('--dry') + 1]) if dry else OUT
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / 'paths.csv', keep_default_na=False)
    df['version'] = df.version.astype(str)
    fig_paths(df, '2015', 'chrono_ridge', out)
    fig_paths(df, '2023', 'chrono_ridge', out)
    fig_paths(df, '2023', 'gpt_ridge', out)
    fig_te(pd.read_csv(res / 'u.csv'), pd.read_csv(res / 'limits.csv'), out)
    print('saved to', out)


if __name__ == '__main__':
    main()
