"""
Figures of LLM-BLM_実験4 (図/). Reads experiments/E4_finbert_finetune/results/summary.json, the stored fine-tuned
predictions (score run in summary.json) and E3's sample.

    python 資料集/02_実験/実験4_FinBERT追加学習/作成スクリプト/figs.py
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())  # リポジトリ（資料集の中で動かしても変わらない）
OUT = Path(__file__).resolve().parents[1] / '図'
RES = REPO / 'experiments' / 'E4_finbert_finetune' / 'results'
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})


def fig_coefficients(s):
    m2 = s['M2_main']
    s1 = s['secondary']['S1_ft_alone']
    ft_joint = m2['others']['ft_pred']
    rows = [('追加学習した FinBERT だけ', s1['coef_bp_per_sd'], s1['se_bp'], AQUA),
            ('LLM（FinBERT と同時、主な検定）', m2['coef_bp_per_sd'], m2['se_bp'], BLUE),
            ('追加学習した FinBERT（LLM と同時）', ft_joint['coef_bp'], ft_joint['coef_bp'] / ft_joint['t'], AQUA)]
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    y = np.arange(len(rows))[::-1]
    for yy, (name, c, se, color) in zip(y, rows):
        ax.errorbar(c, yy, xerr=1.96 * se, fmt='o', color=color, ecolor=color, capsize=3, ms=6, lw=1.4)
        ax.text(c, yy + 0.25, f'{c:+.0f}bp', ha='center', fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel('1 標準偏差あたりの 2 取引日の超過リターン（bp、95% 区間）')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'coefficients.pdf')


def fig_r2(s):
    r = s['r2_gain_over_controls']
    names = ['LLM だけ', '追加学習した\nFinBERT だけ', '両方']
    vals = [100 * r['llm_only'], 100 * r['ft_only'], 100 * r['both']]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    bars = ax.bar(range(3), vals, color=[BLUE, AQUA, GRAY], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f'+{v:.2f}', ha='center', fontsize=9, color=INK)
    ax.set_xticks(range(3), names)
    ax.set_ylabel('決定係数の増分（ポイント）')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'r2_gain.pdf')


def fig_double_sort(s):
    """Mean reaction by terciles of the LLM score and of the fine-tuned prediction (exploratory)."""
    spec = importlib.util.spec_from_file_location('e3_run', REPO / 'experiments' / 'E3_analyst_surprise' / 'run.py')
    e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e3)
    from events import config
    from marketdb.db import events as db
    from marketdb.db.connection import connect
    conn = connect(config.DB_PATH)
    A = e3.sample_a(conn)
    ft = db.load_scores(conn, s['score_run'], ['predicted_reaction'])['predicted_reaction']
    A = A.join(ft.rename('ft_pred'), on='accession')
    y = A.abn_r_0_1.clip(A.abn_r_0_1.quantile(0.01), A.abn_r_0_1.quantile(0.99)) * 100
    tl = pd.qcut(A.llm_composite.rank(method='first'), 3, labels=['低', '中', '高'])
    tf = pd.qcut(A.ft_pred.rank(method='first'), 3, labels=['低', '中', '高'])
    table = y.groupby([tf, tl], observed=True).mean().unstack()
    counts = y.groupby([tf, tl], observed=True).size().unstack()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    lim = np.abs(table.values).max()
    im = ax.imshow(table.values, cmap='RdBu', vmin=-lim, vmax=lim)
    for i in range(3):
        for j in range(3):
            v = table.values[i, j]
            ax.text(j, i, f'{v:+.2f}%\n({counts.values[i, j]:,})', ha='center', va='center', fontsize=8.5,
                    color='white' if abs(v) > lim * 0.6 else INK)
    ax.set_xticks(range(3), ['低', '中', '高'])
    ax.set_yticks(range(3), ['低', '中', '高'])
    ax.set_xlabel('LLM の合成点数（3 分位）')
    ax.set_ylabel('追加学習した FinBERT（3 分位）')
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label='2 取引日の超過リターン（%）')
    fig.tight_layout()
    fig.savefig(OUT / 'double_sort.pdf')
    return table, counts


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((RES / 'summary.json').read_text(encoding='utf-8'))
    fig_coefficients(s)
    fig_r2(s)
    table, counts = fig_double_sort(s)
    print(table.round(2).to_string()); print(counts.to_string())
    print('saved to', OUT)
