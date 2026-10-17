"""
Figures that explain why the covariance needs denoising (results/figures/).

    python experiments/X01_covariance_denoise/figs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blm.covariance import denoised_covariance, marcenko_pastur_pdf, shrunk_covariance
from experiments.common import data as D
import run

OUT = Path(__file__).resolve().parent / 'results' / 'figures'
DATE, NEXT = '2024-06-28', '2024-07-31'
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 10, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def eig_desc(m):
    return np.sort(np.linalg.eigvalsh(m))[::-1]


def fig1_noise_only():
    """With independent stocks every true eigenvalue is 1, yet the sample spreads them out, more so with fewer days."""
    rng = np.random.default_rng(0)
    n = 457
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, t in zip(axes, [2520, 252]):
        x = rng.normal(size=(t, n))
        values = np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))
        positive = values[values > 1e-8]
        ratio = n / t
        lo, hi = (1 - np.sqrt(ratio)) ** 2, (1 + np.sqrt(ratio)) ** 2
        bins = np.linspace(0, 6, 61)
        ax.hist(positive, bins=bins, weights=np.full(len(positive), 1 / (n * 0.1)), color=BLUE, alpha=0.55,
                edgecolor='white', linewidth=0.6, label='標本の固有値')
        grid = np.linspace(0.001, 6, 600)
        ax.plot(grid, marcenko_pastur_pdf(grid, 1.0, ratio), color=ORANGE, lw=2, label='Marcenko–Pastur の理論')
        ax.axvline(1, color=INK, lw=1.2, ls='--')
        ax.text(1.05, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1, '本当の値（全部 1）', color=INK, fontsize=9)
        zero = (values <= 1e-8).sum()
        note = f'銘柄 {n}・日数 {t}（q = {ratio:.2f}）\n範囲 {lo:.2f}〜{hi:.2f}'
        if zero:
            note += f'\nほかに 0 の固有値が {zero} 個'
        ax.text(0.97, 0.95, note, transform=ax.transAxes, ha='right', va='top', fontsize=9, color=INK2)
        ax.set_title(f'{"10 年分" if t == 2520 else "1 年分（BLM の窓）"}のデータ')
        ax.set_xlabel('固有値')
        ax.set_xlim(0, 6)
    axes[0].set_ylabel('密度')
    axes[0].legend(frameon=False, loc='center right')
    fig.suptitle('① 銘柄どうしが無関係（本当は全部ノイズ）でも、推定した固有値は大きくばらつく', x=0.01, ha='left',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / '1_noise_only.png', dpi=160)


def real_returns():
    conn = D.connect()
    returns, mom, future, w_mkt = run.inputs(conn, DATE, NEXT)
    return returns


def fig2_real(returns):
    """Real S&P500 correlations: a few large eigenvalues (market, sectors) stand out of a bulk that looks like noise."""
    n, t = returns.shape[1], returns.shape[0]
    values = eig_desc(returns.corr().values)
    den = denoised_covariance(returns)
    variance, edge, k = den.attrs['noise_variance'], den.attrs['edge'], den.attrs['n_signal']
    positive = values[values > 1e-8]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    bins = np.linspace(0, 6, 61)
    ax.hist(positive[positive <= 6], bins=bins, weights=np.full((positive <= 6).sum(), 1 / (n * 0.1)), color=BLUE,
            alpha=0.55, edgecolor='white', linewidth=0.6, label='実際の固有値')
    grid = np.linspace(0.001, 6, 600)
    ax.plot(grid, marcenko_pastur_pdf(grid, variance, n / t), color=ORANGE, lw=2, label='ノイズだけのときの分布（当てはめ）')
    ax.axvline(edge, color=INK, lw=1.2, ls='--')
    ax.text(edge + 0.08, ax.get_ylim()[1] * 0.85, f'ノイズの上端 {edge:.2f}', fontsize=9, color=INK)
    ax.set_xlim(0, 6)
    ax.set_xlabel('固有値（6 まで）')
    ax.set_ylabel('密度')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 0.75))
    ax.set_title(f'実データ（{DATE}、{n} 銘柄・{t} 日）')
    ax = axes[1]
    top = values[:15]
    colors = [ORANGE if v > edge else GRAY for v in top]
    ax.bar(range(1, 16), top, color=colors, width=0.7)
    ax.set_yscale('log')
    ax.axhline(edge, color=INK, lw=1.2, ls='--')
    ax.text(15.4, edge * 1.15, 'ノイズの上端', ha='right', fontsize=9, color=INK)
    ax.text(1.5, top[0], f'市場全体 {top[0]:.0f}', fontsize=9, color=INK, va='center')
    ax.set_xlabel('固有値の大きさの順位')
    ax.set_title('大きい順に 15 個（対数）')
    ax.set_xticks([1, 5, 10, 15])
    ax.grid(axis='x', visible=False)
    share = values[0] / values.sum()
    fig.suptitle(f'② 実際の株価：市場全体（相関の {share:.0%}）など {k} 個が上端を超え、残りの {n - k} 個はノイズと区別できない',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / '2_real_eigenvalues.png', dpi=160)


def fig3_spectrum(returns):
    """What each estimator does to the eigenvalues; the inverse (what the optimizer uses) is dominated by the smallest."""
    sd = returns.std().values
    est = {
        '標本（そのまま）': returns.corr().values,
        'Ledoit–Wolf（縮小）': (lambda c: c / np.outer(sd, sd))(shrunk_covariance(returns).values),
        'Marcenko–Pastur（ノイズ除去）': (lambda c: c / np.outer(sd, sd))(denoised_covariance(returns).values),
    }
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for (name, m), color in zip(est.items(), [GRAY, BLUE, ORANGE]):
        v = np.clip(eig_desc(m), 1e-6, None)
        ax.plot(np.arange(1, len(v) + 1), v, color=color, lw=2, label=name)
    n_days = returns.shape[0]
    ax.axvline(n_days, color=INK2, lw=1, ls=':')
    ax.text(n_days + 5, 3e-6, f'{n_days} 番目より後は標本では 0\n（日数 < 銘柄数。図では 10⁻⁶ に描いた）', fontsize=9, color=INK2)
    ax.set_yscale('log')
    ax.set_xlabel('固有値の大きさの順位')
    ax.set_ylabel('固有値（対数）')
    ax.legend(frameon=False, loc='upper right')
    fig.suptitle('③ 最適化は Σ の逆行列を使う。逆行列では小さい固有値ほど大きく効く（1/固有値）→ 小さい側のノイズが比率を振り回す',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / '3_spectrum.png', dpi=160)


def fig4_outcome():
    """X01: unconstrained minimum variance, predicted vs realized volatility and how much the weights move."""
    s = json.loads((Path(__file__).resolve().parent / 'results' / 'summary.json').read_text(encoding='utf-8'))
    names = ['sample', 'lw', 'mp']
    labels = ['標本', 'Ledoit–Wolf', 'Marcenko–Pastur']
    pred = [100 * s['gmv_pred'][n]['mean'] for n in names]
    real = [100 * s['gmv_vol'][n]['mean'] for n in names]
    turn = [s['gmv_turnover'][n]['mean'] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    x = np.arange(3)
    b1 = ax.bar(x - 0.18, pred, 0.34, color=GRAY, label='推定したリスク（予測）')
    b2 = ax.bar(x + 0.18, real, 0.34, color=BLUE, label='翌月の実際のリスク')
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.2, f'{r.get_height():.1f}%', ha='center',
                    fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel('年率のボラティリティ（%）')
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('最小分散ポートフォリオ（制約なし）のリスク')
    ax.grid(axis='x', visible=False)
    ax = axes[1]
    bars = ax.bar(x, turn, 0.5, color=AQUA)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.1, f'{r.get_height():.1f}', ha='center', fontsize=9,
                color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel('1 か月の比率の入れ替え（Σ|Δw|）')
    ax.set_title('毎月どれだけ比率が入れ替わるか')
    ax.grid(axis='x', visible=False)
    fig.suptitle(f'④ ノイズのまま最適化すると、リスクを約 3 分の 1 に見誤り、比率も毎月大きく入れ替わる\n'
                 f'（2016〜2026 年の {s["n_dates"]} か月の平均、約 460 銘柄）',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / '4_outcome.png', dpi=160)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_noise_only()
    r = real_returns()
    fig2_real(r)
    fig3_spectrum(r)
    fig4_outcome()
    print('saved to', OUT)
