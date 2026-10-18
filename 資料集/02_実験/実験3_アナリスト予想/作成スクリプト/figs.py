"""
Figures of LLM-BLM_実験3 (図/). Reads experiments/E3_analyst_surprise/results/summary.json, E2's panel and the
analyst estimates in the database.

    python 資料集/02_実験/実験3_アナリスト予想/作成スクリプト/figs.py
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
RES = REPO / 'experiments' / 'E3_analyst_surprise' / 'results'
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})


def load_e3():
    spec = importlib.util.spec_from_file_location('e3_run', REPO / 'experiments' / 'E3_analyst_surprise' / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fig_coefficients(s):
    """LLM score without and with the analyst control, next to the numeric surprises in the main model."""
    m0, m1 = s['M0_e2_model'], s['M1_main']
    rows = [('LLM の点数（E2 のモデル）', m0['coef_bp_per_sd'], m0['se_bp'], GRAY),
            ('LLM の点数（予想との差を統制）', m1['coef_bp_per_sd'], m1['se_bp'], BLUE),
            ('アナリスト予想との差', m1['others']['analyst_surprise']['coef_bp'],
             m1['others']['analyst_surprise']['coef_bp'] / m1['others']['analyst_surprise']['t'], AQUA),
            ('SUE（前年同期との差）', m1['others']['sue']['coef_bp'],
             m1['others']['sue']['coef_bp'] / m1['others']['sue']['t'], AQUA)]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
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


def fig_within_groups(A):
    """Mean reaction by LLM quintile, separately for releases that beat and that missed the consensus (exploratory)."""
    d = A.copy()
    d['y'] = d.abn_r_0_1.clip(d.abn_r_0_1.quantile(0.01), d.abn_r_0_1.quantile(0.99)) * 100
    d['q'] = pd.qcut(d.llm_composite.rank(method='first'), 5, labels=False) + 1
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for mask, name, color in [(d.surprise > 0, '予想を上回った発表', BLUE), (d.surprise < 0, '予想を下回った発表', ORANGE)]:
        g = d[mask].groupby('q').y
        m, se, n = g.mean(), g.std() / np.sqrt(g.count()), int(mask.sum())
        ax.errorbar(m.index, m, yerr=1.96 * se, fmt='-o', color=color, capsize=3, ms=5, lw=1.8, label=f'{name}（{n:,} 本）')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(1, 6), ['1（低い）', '2', '3', '4', '5（高い）'])
    ax.set_xlabel('LLM の合成点数の 5 分位（標本 A 全体で区切る）')
    ax.set_ylabel('2 取引日の超過リターン（%）')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'within_groups.pdf')


def fig_robustness(s):
    rows = [('主な検定（M1）', s['M1_main'], BLUE)]
    names = {'R1_beat_miss': 'R1 上回り・下回りの印', 'R3_matched_and_timing': 'R3 照合と時刻のずれを除く',
             'R4_no_winsorizing_surprise': 'R4 驚きを丸めない'}
    rows += [(names[k], s['robustness'][k], GRAY) for k in names]
    rows.append(('S2 その後のドリフト（2〜21 日）', s['secondary']['S2_drift'], ORANGE))
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    y = np.arange(len(rows))[::-1]
    for yy, (name, v, color) in zip(y, rows):
        ax.errorbar(v['coef_bp_per_sd'], yy, xerr=1.96 * v['se_bp'], fmt='o', color=color, ecolor=color, capsize=2.5, ms=5)
        ax.text(172, yy, f"{v['coef_bp_per_sd']:+.0f}（t {v['t']:.1f}）", va='center', fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-25, 205)
    ax.set_xlabel('LLM の合成点数の係数（1 標準偏差あたりの bp、95% 区間）')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'robustness.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((RES / 'summary.json').read_text(encoding='utf-8'))
    e3 = load_e3()
    from events import config
    from marketdb.db.connection import connect
    A = e3.sample_a(connect(config.DB_PATH))
    fig_coefficients(s)
    fig_within_groups(A)
    fig_robustness(s)
    print('saved to', OUT)
