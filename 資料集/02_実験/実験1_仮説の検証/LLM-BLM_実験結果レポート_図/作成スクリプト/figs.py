"""Figures of the results report (vector PDF), drawn from the saved results of experiments/H0x/results.

    python 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験結果レポート_図/作成スクリプト/figs.py      (from the repository root)

Same look as the figures of the overall report (palette slots in fixed order).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())  # リポジトリ（資料集の中で動かしても変わらない）
E = ROOT / 'experiments'
F = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b', 'axes.grid': True, 'grid.color': '#e6e5e1',
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.titlesize': 10,
                     'axes.titleweight': 'normal', 'axes.unicode_minus': False})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK2, NEUTRAL = '#52514e', '#b9b8b2'


def save(fig, name):
    fig.savefig(F / f'{name}.pdf')
    plt.close(fig)
    print(name)


def js(path):
    return json.loads((E / path).read_text(encoding='utf-8'))


def months(ax, dates, step=12):
    ax.set_xticks(range(0, len(dates), step), [d[:7] for d in list(dates)[::step]], fontsize=7)


def h01():
    m = pd.read_csv(E / 'H01_verification/results/monthly.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(len(m))
    axs[0].bar(x, 100 * m.slsqp_progress, color=C[0], width=0.7)
    axs[0].set_ylim(85, 101); axs[0].set_ylabel('改善できる幅のうち（%）'); axs[0].set_title('元の最適化（SLSQP）の到達度'); months(axs[0], m.date); axs[0].grid(axis='x', visible=False)
    axs[1].plot(x, m.original_effective_n, 'o-', ms=3, color=C[1], lw=1, label='SLSQP（元の計算）')
    axs[1].plot(x, m.exact_effective_n, 's-', ms=3, color=C[0], lw=1, label='最後まで解いた値')
    axs[1].set_yscale('log'); axs[1].set_ylabel('実質銘柄数（対数）'); axs[1].set_title('同じ式の比率の集中'); months(axs[1], m.date); axs[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, 'h01_slsqp')


def h02():
    s = js('H02_statistical_baseline/results/summary.json')['ic']
    names = [('ic_llm', 'LLM\n（実行 1）'), ('ic_replica', 'LLM の\n写し'), ('ic_input_mean', '入力の\n月の平均'),
             ('ic_llm_residual', 'LLM に\n固有の部分'), ('ic_mom', '12-1\nモメンタム'), ('ic_rev', '短期\n反転'), ('ic_stat', '統計\nモデル')]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for i, (k, lab) in enumerate(names):
        v = s[k]
        col = C[1] if k == 'ic_llm_residual' else (C[0] if k.startswith('ic_llm') or k in ('ic_replica', 'ic_input_mean') else C[2])
        ax.errorbar(i, v['mean'], yerr=[[v['mean'] - v['ci_low']], [v['ci_high'] - v['mean']]], fmt='o', color=col, capsize=3, ms=5)
        ax.text(i + 0.12, v['mean'], f"{v['mean']:+.3f}", fontsize=7.5, va='center')
    ax.axhline(0, color=INK2, lw=0.8); ax.axhline(0.03, color=NEUTRAL, lw=0.8, ls='--'); ax.text(len(names) - 0.6, 0.033, '意味のある最小の効果 0.03', fontsize=7, ha='right', color=INK2)
    ax.set_xticks(range(len(names)), [n for _, n in names], fontsize=7.5); ax.set_ylabel('IC（30 か月の平均と 95% 区間）'); ax.grid(axis='x', visible=False)
    save(fig, 'h02_ic')


def h03():
    m = pd.read_csv(E / 'H03_view_shape/results/monthly.csv')
    d = pd.read_csv(E / 'H03_view_shape/results/diagnostic_beta_tilt.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for name, col, lab in [('absolute|calibrated_diagonal|te3', C[0], '絶対ビュー'), ('relative|calibrated_diagonal|te3', C[1], '相対ビュー')]:
        g = m[m.name == name]
        axs[0].scatter(100 * g.forecast_level, g.active_beta, s=12, color=col, label=lab)
    axs[0].axhline(0, color=INK2, lw=0.6); axs[0].set_xlabel('予想の水準（時価総額加重、%/日）'); axs[0].set_ylabel('能動ベータ'); axs[0].set_title('対角の Ω、TE 3%'); axs[0].legend(fontsize=7)
    axs[1].scatter(100 * d.mkt_input, d.xs_corr_Q_beta, s=12, color=C[2])
    axs[1].axhline(0, color=INK2, lw=0.6); axs[1].set_xlabel('判断月の市場のリターン（%/日）'); axs[1].set_ylabel('予想とベータの順位相関')
    r = np.corrcoef(d.mkt_input, d.xs_corr_Q_beta)[0, 1]; axs[1].set_title(f'銘柄の間のなぞり（相関 {r:+.2f}）')
    fig.tight_layout(); save(fig, 'h03_beta')


def h04():
    p = pd.read_csv(E / 'H04_omega_calibration/results/placebo.csv')
    s = js('H04_omega_calibration/results/summary.json')['primary_placebo']
    m = pd.read_csv(E / 'H04_omega_calibration/results/monthly.csv')
    g = m[m.name == 'calibrated_proportional_shrunk']
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    axs[0].hist(100 * p.active_return, bins=25, color=NEUTRAL, edgecolor='white', linewidth=0.3)
    axs[0].axvline(100 * s['actual_active_return'], color=C[1], lw=1.6)
    axs[0].text(100 * s['actual_active_return'] + 0.1, axs[0].get_ylim()[1] * 0.85, f"LLM のビュー\n{100 * s['actual_active_return']:+.2f}%（p={s['p_one_sided']:.2f}）", fontsize=7.5, color=C[1])
    axs[0].set_xlabel('年率超過リターン（%、コスト控除後）'); axs[0].set_ylabel('置換の回数'); axs[0].set_title('並べ替えたビュー 200 回との比較')
    x = np.arange(len(g))
    axs[1].step(x, g.slope_raw.clip(lower=0), where='post', color=NEUTRAL, label='傾き（0 以上に）')
    axs[1].step(x, g.b, where='post', color=C[0], label='縮めた傾き $b_t$')
    axs[1].set_ylabel('採用率'); axs[1].set_title('判断日までのデータで測った傾き'); months(axs[1], g.date); axs[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, 'h04_placebo')


def h05():
    sc = pd.read_csv(E / 'H05_text_vs_numbers/results/scores.csv')
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.4), sharey=True)
    for ax, (c, lab) in zip(axs, [('G', 'G：文章＋数字'), ('T', 'T：文章だけ'), ('N', 'N：数字だけ')]):
        g = sc[(sc.condition == c) & (sc.name == 'market')]
        ax.scatter(100 * g.sp_1m, g.score, s=12, color=C[0] if c == 'T' else C[1])
        rho = g[['score', 'sp_1m']].corr('spearman').iloc[0, 1]
        ax.set_title(f'{lab}\n順位相関 {rho:+.2f}', fontsize=8.5); ax.set_xlabel('S&P500 の直近 1 か月（%）', fontsize=8)
    axs[0].set_ylabel('市場の点数（1〜5）'); fig.tight_layout(); save(fig, 'h05_trace')


def h06():
    m = pd.read_csv(E / 'H06_earnings_text/results/monthly.csv')
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={'width_ratios': [1.4, 1]})
    x = np.arange(len(m))
    axs[0].bar(x, m.ic, color=[C[0] if v >= 0 else C[1] for v in m.ic], width=0.72)
    axs[0].axhline(m.ic.mean(), color=INK2, lw=0.8, ls='--'); axs[0].axhline(0, color=INK2, lw=0.6)
    axs[0].set_ylabel('順位相関'); axs[0].set_title(f'決算発表文の点数の IC（平均 {m.ic.mean():+.3f}）'); months(axs[0], m.date); axs[0].grid(axis='x', visible=False)
    labs = ['直近 1 か月\nのリターン', '判断月の\n平均', '売上の\n前年比', '利益の\n前年比']
    vals = [m.corr_return_1m.mean(), m.corr_input_mean.mean(), m.corr_revenue_growth.mean(), m.corr_income_growth.mean()]
    b = axs[1].bar(range(4), vals, color=[C[1], C[1], C[2], C[2]], width=0.6)
    for bb, v in zip(b, vals):
        axs[1].text(bb.get_x() + bb.get_width() / 2, v + 0.01, f'{v:+.2f}', ha='center', fontsize=7.5)
    axs[1].set_xticks(range(4), labs, fontsize=7); axs[1].set_ylim(0, 0.5); axs[1].set_title('点数と各量の順位相関'); axs[1].grid(axis='x', visible=False)
    fig.tight_layout(); save(fig, 'h06_ic')


def h07():
    path = E / 'H07_feedback_llm/results/summary.json'
    if not path.exists():
        return
    s = json.loads(path.read_text(encoding='utf-8'))['by_condition']
    conds = ['F0', 'F0p', 'F1', 'F2', 'F3']
    labs = ['F0\n実行 1', "F0'\n聞き直し", 'F1\n外れのみ', 'F2\n較正の要約', 'F3\n役の分離']
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5))
    panels = [('IC', [s[c]['ic']['mean'] for c in conds], '{:+.3f}'),
              ('水準の偏り（%/日）', [100 * s[c]['bias'] for c in conds], '{:+.2f}'),
              ('入力との相関', [s[c]['input_corr'] for c in conds], '{:.2f}')]
    for ax, (t, v, fmt) in zip(axs, panels):
        b = ax.bar(range(5), v, color=[NEUTRAL, NEUTRAL, C[0], C[1], C[2]], width=0.65)
        for bb, val in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, val, fmt.format(val), ha='center', va='bottom' if val >= 0 else 'top', fontsize=6.8)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title(t, fontsize=9); ax.set_xticks(range(5), labs, fontsize=6.5); ax.grid(axis='x', visible=False)
    fig.tight_layout(); save(fig, 'h07_feedback')


def cumulative():
    rows = {}
    m4 = pd.read_csv(E / 'H04_omega_calibration/results/monthly.csv')
    m6 = pd.read_csv(E / 'H06_earnings_text/results/portfolio_monthly.csv')
    m3 = pd.read_csv(E / 'H03_view_shape/results/monthly.csv')
    bench = js('H04_omega_calibration/results/summary.json')
    series = {'回答の分散の Ω（元の方法）+ TE 3%': m4[m4.name == 'answer_variance'],
              '較正して縮めた Ω + TE 3%（LLM の数値予想）': m4[m4.name == 'calibrated_proportional_shrunk'],
              '決算発表文の点数 + 同じ作り方': m6[m6.name == 'text_scores'],
              '較正した Ω、TE の制約なし': m3[m3.name == 'absolute|proportional|no_te']}
    # the benchmark's net return is recomputed from the same months
    from experiments.common import data as D
    from experiments.common.backtest import market_cap_strategy, run
    b = run(market_cap_strategy, D.load_all(D.connect())).net
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = pd.to_datetime(b.index)
    for (lab, g), col in zip(series.items(), C):
        r = g.set_index('date').net.reindex(b.index)
        rel = np.cumprod(1 + r.values) / np.cumprod(1 + b.values)
        ax.plot(x, rel, color=col, lw=1.4, label=lab)
    ax.axhline(1, color=INK2, lw=0.6); ax.set_ylabel('時価総額加重に対する比'); ax.legend(fontsize=6.8, loc='upper left')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    save(fig, 'cumulative')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    for f in (h01, h02, h03, h04, h05, h06, h07, cumulative):
        f()
