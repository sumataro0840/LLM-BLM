"""
Figures of LLM-BLM_実験7・8 (図/). Reads experiments/E7_monthly_blm and E8_daily_blm results (summary.json, the
by-month / by-day CSV files) and E3's main coefficient for the 2-day window.

    python 資料集/02_実験/実験7・8_月次と日次のBLM/作成スクリプト/figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def fig_e7_cumulative():
    d = pd.read_csv(EXP / 'E7_monthly_blm' / 'results' / 'by_month.csv', parse_dates=['date'])
    # each row is the month after its decision date: plot at the end of the holding month
    month = d.date.dt.to_period('M')
    ends = [month.iloc[0]] + list(month + 1)
    x = pd.DatetimeIndex([m.to_timestamp(how='end').normalize() for m in ends])
    blm = np.r_[1, (1 + d.net).cumprod()]
    bench = np.r_[1, (1 + d.bench).cumprod()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.4), height_ratios=[1.6, 1], sharex=True)
    ax1.plot(x, 100 * (blm - 1), color=BLUE, lw=1.6, label='LLM の BLM（費用の後）')
    ax1.plot(x, 100 * (bench - 1), color=GRAY, lw=1.6, label='時価総額')
    ax1.set_ylabel('累積リターン（%）')
    ax1.legend(frameon=False, fontsize=8.5, loc='upper left')
    colors = [AQUA if v > 0 else ORANGE for v in d.active_net]
    ax2.bar(x[1:], 100 * d.active_net.to_numpy(), width=20, color=colors)
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_ylabel('月の超過（%）')
    fig.tight_layout()
    fig.savefig(OUT / 'e7_cumulative.pdf')


def fig_e8_windows(e3, e8):
    m1 = e3['M1_main']
    p, s1 = e8['part1_regression']['P'], e8['part1_regression']['S1']
    rows = [('発表日と翌日\n（day 0〜1、実験 3）', m1['coef_bp_per_sd'], m1['ci95_bp'], BLUE),
            ('翌日だけ\n（day 1、主な検定）', p['coef_bp_per_sd'], p['ci95_bp'], VIOLET),
            ('翌日から 5 日\n（day 1〜5）', s1['coef_bp_per_sd'], s1['ci95_bp'], GRAY)]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for i, (name, c, ci, color) in enumerate(rows):
        ax.bar(i, c, color=color, width=0.55)
        ax.errorbar(i, c, yerr=[[c - ci[0]], [ci[1] - c]], color=INK, capsize=4, lw=1)
        ax.text(i + 0.32, c, f'{c:+.1f}bp', va='center', fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3), [r[0] for r in rows])
    ax.set_ylabel('1 標準偏差あたり（bp、95% 区間）')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'e8_windows.pdf')


def fig_e8_daily():
    d = pd.read_csv(EXP / 'E8_daily_blm' / 'results' / 'by_day.csv', parse_dates=['day'])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(d.day, 100 * d.active_gross.cumsum(), color=BLUE, lw=1.5, label='費用の前')
    ax.plot(d.day, 100 * d.active_net.cumsum(), color=ORANGE, lw=1.5, label='費用の後（片道 0.1%）')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel('超過リターンの累計（%）')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    fig.savefig(OUT / 'e8_daily.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_e7_cumulative()
    fig_e8_windows(load('E3_analyst_surprise'), load('E8_daily_blm'))
    fig_e8_daily()
    print('saved to', OUT)
