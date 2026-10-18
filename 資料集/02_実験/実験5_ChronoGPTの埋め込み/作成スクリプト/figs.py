"""
Figures of LLM-BLM_実験5 (図/). Reads experiments/E5_chrono_embeddings/results/summary.json and, for the comparison
in sample A, E3's and E4's summary.json.

    python 資料集/02_実験/実験5_ChronoGPTの埋め込み/作成スクリプト/figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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


def bars(ax, rows, ylabel):
    """rows: (label, value, (lo, hi) or None, color)."""
    for i, (name, v, ci, color) in enumerate(rows):
        ax.bar(i, v, color=color, width=0.55)
        if ci is not None:
            ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color=INK, capsize=4, lw=1)
        top = ci[1] if ci is not None else v
        ax.text(i, max(top, 0) + 0.03 * ax.get_ylim()[1], f'{v:+.1f}' if ci is not None else f'+{v:.2f}',
                ha='center', fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)


def fig_block_b(s):
    b = s['block_B']
    p1, s3 = b['P1'], b['S3_with_zero_shot_chrono']
    tf = p1['others']['tfidf_pred']
    zs = s3['others']['chrono_composite']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), width_ratios=[1.4, 1])
    ax1.set_ylim(-12, 52)
    rows = [('ChronoGPT の\n埋め込み＋リッジ', p1['coef_bp_per_sd'], p1['ci95_bp'], BLUE),
            ('TF-IDF＋リッジ\n（同じ回帰）', tf['coef_bp'], (tf['coef_bp'] - 1.96 * tf['coef_bp'] / tf['t'],
                                                         tf['coef_bp'] + 1.96 * tf['coef_bp'] / tf['t']), GRAY),
            ('学習なしの\nChronoGPT（S3）', zs['coef_bp'], (zs['coef_bp'] - 1.96 * zs['coef_bp'] / zs['t'],
                                                       zs['coef_bp'] + 1.96 * zs['coef_bp'] / zs['t']), ORANGE)]
    bars(ax1, rows, '1 標準偏差あたり（bp、95% 区間）')
    ax1.set_title('係数（標本 B、2016〜2023）', fontsize=9.5, color=INK2)
    r = b['r2_gain_vs_tfidf']
    ax2.set_ylim(0, 0.36)
    bars(ax2, [('埋め込み', 100 * r['chrono_emb_only'], None, BLUE), ('TF-IDF', 100 * r['tfidf_pred_only'], None, GRAY),
               ('両方', 100 * r['both'], None, VIOLET)], '決定係数の増分（ポイント）')
    ax2.set_title('説明力', fontsize=9.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / 'block_b.pdf')


def fig_sample_a(s, e3, e4):
    a = s['block_A']
    m1 = e3['M1_main']
    ft = e4['secondary']['S1_ft_alone']
    emb = a['S1_emb_alone']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), width_ratios=[1.4, 1])
    ax1.set_ylim(0, 160)
    bars(ax1, [('gpt-4o-mini\n（学習なし）', m1['coef_bp_per_sd'], m1['ci95_bp'], BLUE),
               ('FinBERT\n（追加学習）', ft['coef_bp_per_sd'], ft['ci95_bp'], AQUA),
               ('ChronoGPT\n（埋め込み＋リッジ）', emb['coef_bp_per_sd'], emb['ci95_bp'], VIOLET)],
         '1 標準偏差あたり（bp、95% 区間）')
    ax1.set_title('単独で入れたときの係数（標本 A）', fontsize=9.5, color=INK2)
    ax2.set_ylim(0, 2.5)
    bars(ax2, [('gpt-4o-mini', 100 * a['r2_gain_over_controls']['llm_composite_only'], None, BLUE),
               ('FinBERT', 100 * e4['r2_gain_over_controls']['ft_only'], None, AQUA),
               ('ChronoGPT', 100 * a['r2_gain_over_controls']['chrono_emb_only'], None, VIOLET)],
         '決定係数の増分（ポイント）')
    ax2.set_title('説明力', fontsize=9.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / 'sample_a.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    s = load('E5_chrono_embeddings')
    fig_block_b(s)
    fig_sample_a(s, load('E3_analyst_surprise'), load('E4_finbert_finetune'))
    print('saved to', OUT)
