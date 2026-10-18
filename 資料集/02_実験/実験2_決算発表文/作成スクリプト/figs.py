"""
Figures of LLM-BLM_実験2 (図/). Reads the database and experiments/E2_event_reaction/results/
(panel.csv is rebuilt by experiments/E2_event_reaction/run.py when missing).

    python 資料集/02_実験/実験2_決算発表文/作成スクリプト/figs.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())  # リポジトリ（資料集の中で動かしても変わらない）
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'E2_event_reaction'))
import run as e2
from events import config
from events.evaluate import regress
from events.import_h06 import trading_days
from events.returns import load_closes, sector_benchmarks
from marketdb.db.connection import connect

OUT = Path(__file__).resolve().parents[1] / '図'
RES = REPO / 'experiments' / 'E2_event_reaction' / 'results'
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#b9b8b2', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'pdf.fonttype': 42})


def sample_a(panel):
    return panel[panel.abn_r_0_1.notna() & panel.filing_date.between(*e2.SAMPLE_A) & panel.llm_composite.notna()].copy()


def fig_timing(panel):
    """When releases are accepted by EDGAR (New York time), by session."""
    t = pd.to_datetime(panel.accepted_at.str[:19]).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    hour = t.dt.hour + t.dt.minute / 60
    colors = {'pre': BLUE, 'open': YELLOW, 'post': AQUA, 'late': ORANGE}
    names = {'pre': '寄り前（〜9:30）', 'open': '場中（9:30〜16:00）', 'post': '引け後（16:00〜17:30）', 'late': '17:30 以後'}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bins = np.arange(0, 24.25, 0.25)
    bottom = np.zeros(len(bins) - 1)
    for s in ['pre', 'open', 'post', 'late']:
        h, _ = np.histogram(hour[panel.session == s], bins=bins)
        ax.bar(bins[:-1], h, width=0.25, bottom=bottom, align='edge', color=colors[s],
               label=f'{names[s]}  {(panel.session == s).mean():.0%}', linewidth=0)
        bottom += h
    ax.set_xlim(4, 24)
    ax.set_xticks(range(4, 25, 2))
    ax.set_xlabel('EDGAR の受理時刻（ニューヨーク時間）')
    ax.set_ylabel('発表の数（15 分ごと）')
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / 'timing.pdf')


def fig_quintile(A):
    d = e2.prepare(A.assign(y=A.abn_r_0_1), 'llm_composite')
    q = pd.qcut(d.llm_composite.rank(method='first'), 5, labels=False)
    g = d.groupby(q).y
    mean, se = g.mean() * 100, g.std() / np.sqrt(g.count()) * 100
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    colors = [ORANGE if m < 0 else BLUE for m in mean]
    ax.bar(np.arange(1, 6), mean, color=colors, width=0.62)
    ax.errorbar(np.arange(1, 6), mean, yerr=1.96 * se, fmt='none', ecolor=INK2, capsize=3, lw=1)
    for x, m in zip(range(1, 6), mean):
        ax.text(x + 0.33, m, f'{m:+.2f}%', ha='left', va='center', fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(1, 6), ['1（低い）', '2', '3', '4', '5（高い）'])
    ax.set_xlabel('LLM の合成点数の 5 分位')
    ax.set_ylabel('2 取引日の超過リターン（%）')
    ax.set_ylim(-2.6, 2.0)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'quintile.pdf')


def fig_car(A, conn):
    """Cumulative abnormal return from day -5 to +21 for the top and bottom fifths of the LLM score."""
    days = trading_days(conn)
    index = {d: n for n, d in enumerate(days)}
    bench = sector_benchmarks(conn)
    symbols = sorted(set(A.symbol) | {b for b in bench.values() if b} | {'^GSPC'})
    closes = load_closes(conn, symbols, '2023-09-01', '2026-12-31')
    rets = closes.pct_change()
    q = pd.qcut(A.llm_composite.rank(method='first'), 5, labels=False)
    span = range(-5, 22)
    paths = {0: [], 2: [], 4: []}
    for (sym, e), qq in zip(zip(A.symbol, A.event_date), q):
        if qq not in paths or e not in index:
            continue
        k = index[e]
        if k + 21 >= len(days):
            continue
        b = bench.get(sym) or '^GSPC'
        ds = [days[k + j] for j in span]
        ab = (rets[sym].reindex(ds) - rets[b].reindex(ds)).fillna(0).to_numpy()
        paths[qq].append(np.cumsum(ab))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for qq, color, name in [(4, BLUE, '上位 20%'), (2, GRAY, '中位 20%'), (0, ORANGE, '下位 20%')]:
        p = np.array(paths[qq]) * 100
        m = p.mean(axis=0)
        ax.plot(list(span), m, color=color, lw=2, label=f'{name}（{len(p):,} 件）')
        ax.text(21.3, m[-1], f'{m[-1]:+.2f}%', va='center', fontsize=8.5, color=INK)
    ax.axvspan(-0.5, 1.5, color='#e4eef9', zorder=0)
    ax.text(0.5, -1.1, '主な窓\n（0〜1 日目）', ha='center', va='center', fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(-5, 23.5)
    ax.set_xlabel('発表からの取引日（0 日目＝発表を初めて織り込む終値）')
    ax.set_ylabel('累積の超過リターン（%）')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.78), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / 'car.pdf')


def fig_compare(A):
    """Each text score alone (with the numeric controls) and together with the LLM score."""
    d = e2.prepare(A.assign(y=A.abn_r_0_1), 'llm_composite')
    text = ['dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred']
    rest = [c for c in e2.CONTROLS if c not in text]
    names = {'llm_composite': 'LLM（gpt-4o-mini）', 'tfidf_pred': 'TF-IDF＋リッジ', 'finbert_net_tone': 'FinBERT',
             'dictionary_net_tone': '辞書（LM）', 'sue': 'SUE（数値の驚き）'}
    alone, together = {}, {}
    full = regress(d, 'y', ['llm_composite', *e2.CONTROLS], e2.FIXED)
    for f in names:
        r = regress(d, 'y', [f, *[c for c in rest if c != f]], e2.FIXED).loc[f]
        alone[f] = (r.coef * 1e4, r.se * 1e4)
        together[f] = (full.loc[f].coef * 1e4, full.loc[f].se * 1e4)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(names))[::-1]
    for off, data, color, label in [(0.17, alone, GRAY, '1 つずつ（ほかの文章の点数なし）'),
                                    (-0.17, together, BLUE, 'すべて同時（主な検定のモデル）')]:
        c = [data[f][0] for f in names]
        e = [1.96 * data[f][1] for f in names]
        ax.errorbar(c, y + off, xerr=e, fmt='o', color=color, ecolor=color, capsize=2.5, ms=5, lw=1.2, label=label)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, list(names.values()))
    ax.set_xlabel('1 標準偏差あたりの効果（bp、95% 区間）')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'compare.pdf')


def fig_forest():
    s = json.loads((RES / 'summary.json').read_text(encoding='utf-8'))
    rows = [('主な検定', s['main'], BLUE)]
    labels = {'R1_window_0_0': 'R1 窓 [0, 0]', 'R1_window_m1_1': 'R1 窓 [−1, +1]', 'R2_without_open': 'R2 場中の受理を除く',
              'R3_vs_sp500': 'R3 S&P500 との差', 'R4_no_winsorizing': 'R4 丸めなし', 'R5_winsorize_0.5': 'R5 丸め 0.5%',
              'R5_winsorize_2.5': 'R5 丸め 2.5%'}
    rows += [(labels[k], s['robustness'][k], GRAY) for k in labels]
    sec = {'S4_tone': 'S4 語調だけ', 'S4_outlook': 'S4 見通しだけ', 'S4_guidance': 'S4 ガイダンスだけ',
           'S1_drift': 'S1 その後のドリフト（2〜21 日）', 'S3_sample_A_chrono': 'S3 ChronoGPT（標本 A）',
           'S2_sample_B_chrono': 'S2 ChronoGPT（標本 B）'}
    rows += [(sec[k], s['secondary'][k], AQUA if 'S4' in k else ORANGE) for k in sec]
    rows.append(('R7 標本 B、コロナの月を除く', s['robustness']['R7_B_without_covid'], GRAY))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))[::-1]
    for yy, (name, v, color) in zip(y, rows):
        ax.errorbar(v['coef_bp_per_sd'], yy, xerr=1.96 * v['se_bp'], fmt='o', color=color, ecolor=color, capsize=2.5,
                    ms=5, lw=1.2)
        ax.text(175, yy, f"{v['coef_bp_per_sd']:+.0f}（t {v['t']:.1f}）", va='center', fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-40, 215)
    ax.set_xlabel('1 標準偏差あたりの効果（bp、95% 区間）')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'forest.pdf')


def copy_x01():
    src = REPO / 'experiments' / 'X01_covariance_denoise' / 'results' / 'figures'
    for name in ('2_real_eigenvalues.png', '4_outcome.png'):
        shutil.copy(src / name, OUT / f'x01_{name}')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(RES / 'panel.csv')
    A = sample_a(panel)
    conn = connect(config.DB_PATH)
    fig_timing(panel)
    fig_quintile(A)
    fig_car(A, conn)
    fig_compare(A)
    fig_forest()
    copy_x01()
    print('saved to', OUT)
