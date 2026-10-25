"""
Figures of LLM-BLM_実験S3 (図/). Reads experiments/S3_smallcap_risk_blm/results (paths.csv, summary.json).

    python 資料集/02_実験/実験S3_リスクで補正したBLM/作成スクリプト/figs.py [--dry OUTDIR]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S3_smallcap_risk_blm' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
GRAY, INK2, VIOLET, AQUA, ORANGE, BLUE = '#b9b8b2', '#52514e', '#6250d6', '#1baf7a', '#eb6834', '#2a78d6'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
VERSIONS = {'A': ('A 基準', VIOLET, '-'), 'B': ('B 株価で補正', AQUA, '-'), 'C': ('C 株価＋テキスト', ORANGE, '-'),
            'C_gpt': ('C 株価＋テキスト', ORANGE, '-')}
FREQ = {'quarter': '四半期ごと', 'half': '半年ごと', 'year': '1 年ごと'}


def fig_paths(df, start, model, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    for ax, (freq, title) in zip(axes, FREQ.items()):
        d = df[(df.start == int(start)) & (df.freq == freq)]
        band = d[d.model == 'band'].sort_values('month')
        ax.plot(pd.to_datetime(band.month), band.value, color=INK2, lw=1.8, ls='--', label='範囲の全社（比べる相手）')
        for ver, (name, color, ls) in VERSIONS.items():
            s = d[(d.model == model) & (d.version == ver)].sort_values('month')
            if len(s):
                ax.plot(pd.to_datetime(s.month), s.value, color=color, lw=1.3, ls=ls, label=f'BLM：{name}')
        ax.axhline(1, color=GRAY, lw=0.7)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', labelsize=7.5, rotation=30)
    axes[0].set_ylabel(f'{start} 年末を 1 とした価値')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=7.8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / f'paths_{start}_{model}.pdf')


def fig_bias(runs, out):
    """Bias statistic b of every quarterly run (1 = the ex-ante TE is right on average)."""
    keys = [k for k in runs if '/quarter/' in k and not k.endswith('/band')]
    labels = [k.replace('/quarter/', ' ').replace('chrono_ridge', 'ChronoGPT').replace('gpt_ridge', 'gpt')
              .replace('/', ' ') for k in keys]
    vals = [runs[k]['bias_b'] for k in keys]
    colors = [VERSIONS[k.split('/')[-1]][1] for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(np.arange(len(keys)), vals, color=colors)
    ax.axhline(1, color=INK2, lw=0.8, ls=':')
    ax.set_xticks(np.arange(len(keys)), labels, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('偏り b（実現 ÷ 事前の TE）')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(out / 'bias.pdf')


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
    fig_bias(json.loads((res / 'summary.json').read_text(encoding='utf-8'))['runs'], out)
    print('saved to', out)


if __name__ == '__main__':
    main()
