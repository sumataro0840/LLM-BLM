r"""
BLM解説.tex の数値・表・図を作る（本文の数字はすべてここで計算したもの）。
リポジトリの直下で実行する:

    .venv/bin/python 資料集/勉強用/BLM解説_計算.py

出力は 資料集/勉強用/BLM解説_生成/ に置く。
    values.tex  本文で使う数値（本文では \V{名前} で参照する）
    tab_*.tex   表（tabular 環境ごと）
    fig_*.pdf   図
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
OUT = HERE / 'BLM解説_生成'
OUT.mkdir(exist_ok=True)

import cvxpy as cp
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from blm import config
from blm.black_litterman import posterior_returns, posterior_returns_original
from blm.covariance import sample_covariance, shrunk_covariance
from blm.equilibrium import capm_returns_original, implied_returns
from blm.inputs import load_inputs
from blm.optimizer import optimize, optimize_original
from marketdb.db.connection import connect
from marketdb.db.llm import load_as_of_dates

# ============================================================
# 書き出しの道具
# ============================================================
VALUES = {}


def fmt(x, d):
    """小数 d 桁。負の数のマイナスは数式の記号にする（-0.0 は 0.0 にする）。"""
    x = float(x)
    if round(x, d) == 0:
        x = 0.0
    s = f'{abs(x):.{d}f}'
    return ('\\ensuremath{-}' if x < 0 else '') + s


def pct(x, d=1):
    return fmt(100 * x, d)


def sci(x, d=2):
    """a × 10^b の形。"""
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f'\\ensuremath{{{fmt(m, d)}\\times 10^{{{e}}}}}'


def put(key, text):
    VALUES[key] = text


def write_table(name, text):
    (OUT / f'tab_{name}.tex').write_text(text, encoding='utf-8')


def tabular(colspec, header, rows, midrules=()):
    """header: セルのリスト。rows: セルのリストのリスト。midrules: その行の前に \\midrule を入れる行番号。"""
    lines = [f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for i, row in enumerate(rows):
        if i in midrules:
            lines.append('\\midrule')
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


# ============================================================
# 図の体裁（報告書と同じ配色。系列は 3 色まで）
# ============================================================
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({
    'font.family': 'Hiragino Sans', 'font.size': 8.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'lines.linewidth': 1.6, 'legend.frameon': False, 'pdf.fonttype': 3,
})


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'fig_{name}.pdf')
    plt.close(fig)


def bars(ax, labels, series, colors, names, width=0.38):
    """グループの棒グラフ。棒どうしに小さなすき間を空ける。"""
    x = np.arange(len(labels))
    k = len(series)
    for j, (vals, color, name) in enumerate(zip(series, colors, names)):
        ax.bar(x + (j - (k - 1) / 2) * width, vals, width * 0.9, color=color, label=name, zorder=2)
    ax.set_xticks(x, labels)
    ax.axhline(0, color=INK2, linewidth=0.6, zorder=3)
    ax.grid(axis='x', visible=False)


# ============================================================
# 共通の計算
# ============================================================
def bl_update(Pi, S, P, Q, Om, tau):
    """更新形：事後の平均と、平均の不確かさ M。"""
    tS = tau * S
    G = P @ tS @ P.T + Om
    mu = Pi + tS @ P.T @ np.linalg.solve(G, Q - P @ Pi)
    M = tS - tS @ P.T @ np.linalg.solve(G, P @ tS)
    return mu, M


def bl_information(Pi, S, P, Q, Om, tau):
    """情報形（精度行列を足す形）。"""
    tSi = np.linalg.inv(tau * S)
    Omi = np.linalg.inv(Om)
    H = tSi + P.T @ Omi @ P
    mu = np.linalg.solve(H, tSi @ Pi + P.T @ Omi @ Q)
    return mu, np.linalg.inv(H)


def hl_lambda(P, Q, Om, S, w_eq, tau, delta):
    """He と Litterman の Λ（w* = (w_eq + P'Λ)/(1+τ)）。"""
    A = Om / tau + P @ S @ P.T / (1 + tau)
    a = tau * np.linalg.solve(Om, Q) / delta
    return a - np.linalg.solve(A, P @ S @ w_eq) / (1 + tau) - np.linalg.solve(A, P @ S @ P.T @ a) / (1 + tau)


def qp(mu, S, delta, long_only=True):
    """max w'mu - delta/2 w'Sw, sum w = 1（long_only なら w >= 0 も）。"""
    w = cp.Variable(len(mu))
    cons = [cp.sum(w) == 1] + ([w >= 0] if long_only else [])
    cp.Problem(cp.Maximize(1e4 * (mu @ w - delta / 2 * cp.quad_form(w, cp.psd_wrap(S)))), cons).solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum() if long_only else w.value


def budget_mv(mu, S, delta):
    """予算制約だけの平均分散の解析解：w = Σ^{-1}(μ - γ1)/δ。"""
    one = np.ones(len(mu))
    Si_mu, Si_1 = np.linalg.solve(S, mu), np.linalg.solve(S, one)
    gamma = (one @ Si_mu - delta) / (one @ Si_1)
    return (Si_mu - gamma * Si_1) / delta, gamma


def eff_n(w):
    return 1 / np.sum(np.asarray(w) ** 2)


# ============================================================
# 第 1 章：2 資産の例
# ============================================================
DELTA_TOY = 2.5
muA, sA, muB, sB, rho = 0.06, 0.20, 0.02, 0.10, 0.3
put('toy.muA', pct(muA, 0)); put('toy.sA', pct(sA, 0)); put('toy.muB', pct(muB, 0)); put('toy.sB', pct(sB, 0))
put('toy.rho', fmt(rho, 1))
S2 = np.array([[sA ** 2, rho * sA * sB], [rho * sA * sB, sB ** 2]])
mu2 = np.array([muA, muB])
put('toy.covAB', fmt(S2[0, 1], 4)); put('toy.varA', fmt(S2[0, 0], 2)); put('toy.varB', fmt(S2[1, 1], 2))
w50 = np.array([0.5, 0.5])
put('toy.mp', pct(w50 @ mu2, 1))
put('toy.vp', fmt(w50 @ S2 @ w50, 4)); put('toy.sp', pct(np.sqrt(w50 @ S2 @ w50), 2))
put('toy.vpA', fmt(0.25 * S2[0, 0], 4)); put('toy.vpB', fmt(0.25 * S2[1, 1], 4)); put('toy.vpAB', fmt(2 * 0.25 * S2[0, 1], 4))
rows = []
for r in [1.0, 0.5, 0.3, 0.0, -0.5, -1.0]:
    v = 0.25 * sA ** 2 + 0.25 * sB ** 2 + 2 * 0.25 * r * sA * sB
    rows.append([fmt(r, 1), pct(w50 @ mu2, 1) + '\\%', fmt(v, 4), pct(np.sqrt(v), 2) + '\\%'])
write_table('toy_rho', tabular('rrrr', ['相関 $\\rho$', '期待リターン', '分散', '標準偏差'], rows))
# 1 資産
put('toy.wone', fmt(muA / (DELTA_TOY * sA ** 2), 2))
# 制約なし
S2i = np.linalg.inv(S2)
w_free = np.linalg.solve(DELTA_TOY * S2, mu2)
put('toy.det', fmt(np.linalg.det(S2), 6))
for i, j in [(0, 0), (0, 1), (1, 1)]:
    put(f'toy.Si{i}{j}', fmt(S2i[i, j], 2))
put('toy.Simu0', fmt((S2i @ mu2)[0], 4)); put('toy.Simu1', fmt((S2i @ mu2)[1], 4))
put('toy.wfreeA', fmt(w_free[0], 4)); put('toy.wfreeB', fmt(w_free[1], 4)); put('toy.wfreesum', fmt(w_free.sum(), 4))
put('toy.wfreeApct', pct(w_free[0], 1)); put('toy.wfreeBpct', pct(w_free[1], 1)); put('toy.wfreecash', pct(1 - w_free.sum(), 1))
# 予算制約
w_bud, gamma = budget_mv(mu2, S2, DELTA_TOY)
one = np.ones(2)
put('toy.oneSimu', fmt(one @ S2i @ mu2, 4)); put('toy.oneSione', fmt(one @ S2i @ one, 2))
put('toy.gamma', pct(gamma, 2)); put('toy.gammaraw', fmt(gamma, 5))
put('toy.wbudA', pct(w_bud[0], 1)); put('toy.wbudB', pct(w_bud[1], 1))
# 感度：μA を 1% 上げる
mu2b = mu2 + np.array([0.01, 0])
w_bud2, _ = budget_mv(mu2b, S2, DELTA_TOY)
put('toy.wbudA7', pct(w_bud2[0], 1)); put('toy.wbudB7', pct(w_bud2[1], 1))

# ============================================================
# He と Litterman の G7 の例
# ============================================================
G7_JA = ['オーストラリア', 'カナダ', 'フランス', 'ドイツ', '日本', 'イギリス', 'アメリカ']
G7_EN = ['AUS', 'CAN', 'FRA', 'GER', 'JPN', 'UK', 'USA']
G7_CORR = np.array([
    [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
    [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
    [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
    [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
    [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
    [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
    [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
G7_SIG = np.array([16.0, 20.3, 24.8, 27.1, 21.0, 20.0, 18.7]) / 100
G7_W = np.array([1.6, 2.2, 5.2, 5.5, 11.6, 12.4, 61.5]) / 100
G7_PI_PAPER = [3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6]
DELTA, TAU = 2.5, 0.05
S7 = np.outer(G7_SIG, G7_SIG) * G7_CORR
Pi7 = DELTA * S7 @ G7_W
n7 = len(G7_W)
GER, FRA, UK, CAN, USA, JPN = 3, 2, 5, 1, 6, 4

# 相関の表
rows = [[G7_JA[i]] + [f'{G7_CORR[i, j]:.3f}' for j in range(n7)] for i in range(n7)]
write_table('g7_corr', tabular('l' + 'r' * n7, [''] + G7_EN, rows))

# 入力と均衡リターン
rows = [[G7_JA[i], pct(G7_SIG[i]), pct(G7_W[i]), pct(Pi7[i], 2), f'{G7_PI_PAPER[i]:.1f}'] for i in range(n7)]
write_table('g7_inputs', tabular('lrrrr', ['国', '標準偏差 $\\sigma_i$ (\\%)', '時価総額比率 $w_{\\mathrm{eq},i}$ (\\%)',
                                           '$\\Pi_i$ 再計算 (\\%)', '$\\Pi_i$ 論文 (\\%)'], rows))

# ドイツの Π を項ごとに
rows = []
for j in range(n7):
    cov = S7[GER, j]
    rows.append([G7_JA[j], f'{G7_CORR[GER, j]:.3f}', pct(G7_SIG[j]), fmt(cov, 5), pct(G7_W[j]), fmt(cov * G7_W[j], 6)])
rows.append(['合計 $(\\Sigma w_{\\mathrm{eq}})_{\\mathrm{GER}}$', '', '', '', '', fmt((S7 @ G7_W)[GER], 6)])
write_table('g7_ger', tabular('lrrrrr', ['国 $j$', '$\\rho_{\\mathrm{GER},j}$', '$\\sigma_j$ (\\%)',
                                        '$\\Sigma_{\\mathrm{GER},j}$', '$w_{\\mathrm{eq},j}$ (\\%)', '$\\Sigma_{\\mathrm{GER},j}\\,w_{\\mathrm{eq},j}$'],
                                rows, midrules=(n7,)))
put('g7.SwGER', fmt((S7 @ G7_W)[GER], 6)); put('g7.PiGER', pct(Pi7[GER], 2)); put('g7.PiGERraw', fmt(Pi7[GER], 5))
put('g7.SGERGER', fmt(S7[GER, GER], 5))

# 市場ポートフォリオとベータ
var_m = G7_W @ S7 @ G7_W
Pi_m = G7_W @ Pi7
beta7 = (S7 @ G7_W) / var_m
put('g7.varm', fmt(var_m, 5)); put('g7.sigm', pct(np.sqrt(var_m), 2)); put('g7.Pim', pct(Pi_m, 2))
put('g7.sharpe', fmt(Pi_m / np.sqrt(var_m), 3))
put('g7.deltacheck', fmt(Pi_m / var_m, 3))
rows = [[G7_JA[i], fmt(beta7[i], 3), pct(beta7[i] * Pi_m, 2), pct(Pi7[i], 2)] for i in range(n7)]
write_table('g7_beta', tabular('lrrr', ['国', '$\\beta_i$', '$\\beta_i\\,\\Pi_m$ (\\%)', '$\\Pi_i$ (\\%)'], rows))

# 第 1 章：G7 での感度
scen = {
    '均衡 $\\Pi$': Pi7,
    'ドイツだけ $+1\\%$': Pi7 + np.eye(n7)[GER] * 0.01,
    'ドイツ $+1\\%$、フランス $-1\\%$': Pi7 + np.eye(n7)[GER] * 0.01 - np.eye(n7)[FRA] * 0.01,
    '全部 7\\%': np.full(n7, 0.07),
}
cols = {}
for name, mu in scen.items():
    cols[name] = budget_mv(mu, S7, DELTA)[0]
rows = [[G7_JA[i]] + [pct(cols[n][i]) for n in scen] for i in range(n7)]
rows.append(['比率の絶対値の合計'] + [pct(np.abs(cols[n]).sum()) for n in scen])
write_table('g7_sens', tabular('lrrrr', ['国'] + list(scen), rows, midrules=(n7,)))
put('sens.gerup', pct(cols['ドイツだけ $+1\\%$'][GER])); put('sens.fraup', pct(cols['ドイツだけ $+1\\%$'][FRA]))
put('sens.ukup', pct(cols['ドイツだけ $+1\\%$'][UK]))
put('sens.gerfr', pct(cols['ドイツ $+1\\%$、フランス $-1\\%$'][GER])); put('sens.frafr', pct(cols['ドイツ $+1\\%$、フランス $-1\\%$'][FRA]))
w_eq7 = cols['全部 7\\%']
put('sens.eqmin', pct(w_eq7.min())); put('sens.eqmax', pct(w_eq7.max()))
put('sens.eqminname', G7_JA[int(np.argmin(w_eq7))]); put('sens.eqmaxname', G7_JA[int(np.argmax(w_eq7))])

# 推定誤差
rows = []
for T in [1, 5, 10, 30, 100]:
    rows.append([f'{T} 年', pct(0.20 / np.sqrt(T), 1) + '\\%', pct(0.06 - 1.96 * 0.20 / np.sqrt(T), 1) + '\\% 〜 ' + pct(0.06 + 1.96 * 0.20 / np.sqrt(T), 1) + '\\%'])
write_table('se', tabular('rrr', ['データの長さ $T$', '平均の標準誤差 $\\sigma/\\sqrt{T}$', '95\\% の範囲（真の平均 6\\% のとき）'], rows))

# 図：有効フロンティア（G7、空売りあり、Π を期待リターンとする）
one7 = np.ones(n7)
A_ = one7 @ np.linalg.solve(S7, one7); B_ = one7 @ np.linalg.solve(S7, Pi7); C_ = Pi7 @ np.linalg.solve(S7, Pi7)
D_ = A_ * C_ - B_ ** 2
m_grid = np.linspace(B_ / A_ - 0.02, 0.16, 300)
s_grid = np.sqrt((A_ * m_grid ** 2 - 2 * B_ * m_grid + C_) / D_)
# 空売りなしのフロンティア
lo_m = np.linspace(Pi7.min() + 1e-4, Pi7.max() - 1e-4, 60)
lo_s = []
for m in lo_m:
    w = cp.Variable(n7)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S7))), [cp.sum(w) == 1, w >= 0, Pi7 @ w == m]).solve(solver=cp.CLARABEL)
    lo_s.append(np.sqrt(max(w.value @ S7 @ w.value, 0)))
fig, ax = plt.subplots(figsize=(5.9, 3.4))
ax.plot(100 * s_grid, 100 * m_grid, color=C1, label='効率的フロンティア（空売りあり）', zorder=3)
ax.plot(100 * np.array(lo_s), 100 * lo_m, color=C3, label='効率的フロンティア（空売りなし）', zorder=3)
xs = np.linspace(0, 30, 10)
ax.plot(xs, xs * Pi_m / np.sqrt(var_m), color=C2, linestyle=(0, (4, 2)), label='資本市場線（無リスク資産と市場の組み合わせ）', zorder=2)
ax.scatter(100 * G7_SIG, 100 * Pi7, s=22, color=INK2, zorder=4)
label_offset = {'カナダ': (5, 2), 'イギリス': (-5, -9)}
for i in range(n7):
    if G7_JA[i] == 'アメリカ':   # 線と重ならないよう、引き出し線で離す
        ax.annotate(G7_JA[i], (100 * G7_SIG[i], 100 * Pi7[i]), xytext=(23.0, 7.25), textcoords='data', fontsize=7.5,
                    color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkA=1, shrinkB=3))
        continue
    dx, dy = label_offset.get(G7_JA[i], (5, -3))
    ax.annotate(G7_JA[i], (100 * G7_SIG[i], 100 * Pi7[i]), xytext=(dx, dy), textcoords='offset points', fontsize=7.5,
                color=INK2, ha='right' if dx < 0 else 'left')
ax.scatter([100 * np.sqrt(var_m)], [100 * Pi_m], s=60, color=C2, edgecolor='white', linewidth=1.5, zorder=5)
ax.annotate('市場ポートフォリオ', (100 * np.sqrt(var_m), 100 * Pi_m), xytext=(-8, 9), textcoords='offset points', fontsize=8, color=INK, ha='right')
ax.set_xlim(0, 30); ax.set_ylim(0, 14)
ax.set_xlabel('標準偏差（年率 %）'); ax.set_ylabel('期待超過リターン（年率 %）')
ax.legend(loc='upper left', fontsize=7.5)
save_fig(fig, 'frontier')

# ============================================================
# 第 4 章：1 資産・1 ビューの例
# ============================================================
pi0, sig0, tau0, q0 = 0.05, 0.20, 0.05, 0.10
ts0 = tau0 * sig0 ** 2
put('sc.pi', pct(pi0, 0)); put('sc.sig', pct(sig0, 0)); put('sc.tau', fmt(tau0, 2)); put('sc.q', pct(q0, 0))
put('sc.ts', fmt(ts0, 3)); put('sc.tssd', pct(np.sqrt(ts0), 2))
rows = []
for label, om in [('$\\omega = \\tau\\sigma^2/4$（強い自信）', ts0 / 4), ('$\\omega = \\tau\\sigma^2$', ts0),
                  ('$\\omega = 4\\tau\\sigma^2$（弱い自信）', 4 * ts0), ('$\\omega \\to \\infty$（ビューなし）', np.inf), ('$\\omega \\to 0$（完全な自信）', 0.0)]:
    k = ts0 / (ts0 + om) if np.isfinite(om) else 0.0
    mu = pi0 + k * (q0 - pi0)
    post_var = 1 / (1 / ts0 + 1 / om) if 0 < om < np.inf else (ts0 if om == np.inf else 0.0)
    rows.append([label, fmt(om, 4) if np.isfinite(om) else '$\\infty$', fmt(k, 2), pct(mu, 2) + '\\%', pct(np.sqrt(post_var), 2) + '\\%'])
write_table('scalar', tabular('lrrrr', ['ビューの不確かさ', '$\\omega$', '採用率 $k$', '事後の平均', '事後の標準偏差'], rows))
put('sc.post', pct(pi0 + 0.5 * (q0 - pi0), 2)); put('sc.postsd', pct(np.sqrt(ts0 / 2), 2))

# 図：事前・ビュー・事後の分布
x = np.linspace(-0.08, 0.22, 500)
dens = lambda m, v: np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
fig, ax = plt.subplots(figsize=(5.9, 2.8))
ax.plot(100 * x, dens(pi0, ts0) / 100, color=C1, label=f'事前分布 N({100*pi0:.0f}%, {100*np.sqrt(ts0):.2f}%²)')
ax.plot(100 * x, dens(q0, ts0) / 100, color=C2, label=f'ビュー N({100*q0:.0f}%, {100*np.sqrt(ts0):.2f}%²)')
ax.plot(100 * x, dens(pi0 + 0.5 * (q0 - pi0), ts0 / 2) / 100, color=C3, linewidth=2.2, label=f'事後分布 N({100*(pi0+0.5*(q0-pi0)):.1f}%, {100*np.sqrt(ts0/2):.2f}%²)')
for m, c in [(pi0, C1), (q0, C2), (pi0 + 0.5 * (q0 - pi0), C3)]:
    ax.axvline(100 * m, color=c, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
ax.set_xlabel('期待リターン μ（年率 %）'); ax.set_ylabel('確率密度')
ax.set_xlim(-8, 22); ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize=7.5)
save_fig(fig, 'scalar')

# 図：採用率
r = np.logspace(-2, 2, 300)   # 図は第 7 章の数字を重ねて、最後に作る

# ============================================================
# He と Litterman の例
# ============================================================
p1 = np.zeros(n7); p1[GER] = 1; p1[FRA] = -G7_W[FRA] / (G7_W[FRA] + G7_W[UK]); p1[UK] = -G7_W[UK] / (G7_W[FRA] + G7_W[UK])
p2 = np.zeros(n7); p2[CAN] = 1; p2[USA] = -1


def omega_hl(P, tau=TAU):
    return np.diag(np.diag(P @ (tau * S7) @ P.T))


def run_example(P, Q, Om, tau=TAU):
    mu, M = bl_update(Pi7, S7, P, Q, Om, tau)
    ws = np.linalg.solve(DELTA * (S7 + M), mu)
    lam = hl_lambda(P, Q, Om, S7, G7_W, tau, DELTA)
    assert np.allclose(ws, (G7_W + P.T @ lam) / (1 + tau))
    # 整理した形：Λ = (δA)^{-1} (Q - PΠ/(1+τ))
    A = Om / tau + P @ S7 @ P.T / (1 + tau)
    assert np.allclose(lam, np.linalg.solve(DELTA * A, Q - P @ Pi7 / (1 + tau)))
    mu_i, M_i = bl_information(Pi7, S7, P, Q, Om, tau)
    assert np.allclose(mu, mu_i) and np.allclose(M, M_i)
    return mu, M, ws, lam


def example_table(name, P, mu, ws, tau=TAU):
    rows = []
    for i in range(n7):
        prow = [fmt(100 * P[k, i], 1) if P[k, i] != 0 else '0' for k in range(P.shape[0])]
        rows.append([G7_JA[i]] + prow + [pct(Pi7[i]), pct(mu[i]), pct(G7_W[i]), pct(ws[i]), pct(ws[i] - G7_W[i] / (1 + tau))])
    hdr = ['国'] + [f'$P_{{{k+1}}}$' for k in range(P.shape[0])] + ['$\\Pi$', '$\\bar\\mu$', '$w_{\\mathrm{eq}}$', '$w^*$', '$w^* - \\frac{w_{\\mathrm{eq}}}{1+\\tau}$']
    write_table(name, tabular('l' + 'r' * (P.shape[0] + 5), hdr, rows))


# 例 1
P = p1[None, :]; Q = np.array([0.05]); Om = omega_hl(P)
mu1, M1, ws1, lam1 = run_example(P, Q, Om)
example_table('ex1', P, mu1, ws1)
put('ex1.pFRA', fmt(p1[FRA], 4)); put('ex1.pUK', fmt(p1[UK], 4))
put('ex1.pPi', pct(p1 @ Pi7, 2)); put('ex1.gap', pct(0.05 - p1 @ Pi7, 2))
put('ex1.pSp', fmt(p1 @ S7 @ p1, 5)); put('ex1.ptSp', fmt(p1 @ (TAU * S7) @ p1, 6)); put('ex1.omega', fmt(Om[0, 0], 6))
put('ex1.viewsd', pct(np.sqrt(Om[0, 0]), 2))
put('ex1.pmu', pct(p1 @ mu1, 2)); put('ex1.lam', fmt(lam1[0], 3)); put('ex1.lamscaled', fmt(lam1[0] / (1 + TAU), 3))
tSp = (TAU * S7) @ p1
gain = tSp / (p1 @ (TAU * S7) @ p1 + Om[0, 0])
rows = [[G7_JA[i], fmt(1e4 * tSp[i], 3), fmt(gain[i], 4), pct(gain[i] * (0.05 - p1 @ Pi7), 2), pct(Pi7[i], 2), pct(mu1[i], 2)] for i in range(n7)]
write_table('ex1_steps', tabular('lrrrrr', ['国', '$(\\tau\\Sigma p^\\top)_i \\times 10^4$', '$g_i$', '$g_i\\,(Q - p\\Pi)$ (\\%)', '$\\Pi_i$ (\\%)', '$\\bar\\mu_i$ (\\%)'], rows))
# 事後の不確かさ
rows = [[G7_JA[i], pct(np.sqrt(TAU * S7[i, i]), 2), pct(np.sqrt(M1[i, i]), 2), pct(np.sqrt(S7[i, i]), 2), pct(np.sqrt(S7[i, i] + M1[i, i]), 2)] for i in range(n7)]
write_table('ex1_unc', tabular('lrrrr', ['国', '事前 $\\sqrt{(\\tau\\Sigma)_{ii}}$', '事後 $\\sqrt{M_{ii}}$', '$\\sqrt{\\Sigma_{ii}}$', '$\\sqrt{(\\Sigma + M)_{ii}}$'], rows))
put('ex1.Mratio', fmt(np.sqrt(M1[GER, GER] / (TAU * S7[GER, GER])), 3))
# 1 つのビューで λ = 0 になる Q
put('ex1.qzero', pct(p1 @ Pi7 / (1 + TAU), 2))
A1 = Om[0, 0] / TAU + p1 @ S7 @ p1 / (1 + TAU)
put('ex1.A', fmt(A1, 5)); put('ex1.omtau', fmt(Om[0, 0] / TAU, 5)); put('ex1.pSptau', fmt(p1 @ S7 @ p1 / (1 + TAU), 5))
put('ex1.d', pct(0.05 - p1 @ Pi7 / (1 + TAU), 2)); put('ex1.dA', fmt(DELTA * A1, 5))
lam_check = hl_lambda(P, np.array([p1 @ Pi7 / (1 + TAU)]), Om, S7, G7_W, TAU, DELTA)
assert abs(lam_check[0]) < 1e-12
lam_at_pPi = hl_lambda(P, np.array([p1 @ Pi7]), Om, S7, G7_W, TAU, DELTA)
put('ex1.lamAtpPi', fmt(lam_at_pPi[0], 4))

# 例 2
P2 = np.vstack([p1, p2]); Q2 = np.array([0.05, 0.03]); Om2 = omega_hl(P2)
mu2_, M2_, ws2, lam2 = run_example(P2, Q2, Om2)
example_table('ex2', P2, mu2_, ws2)
put('ex2.p2Pi', pct(p2 @ Pi7, 2)); put('ex2.gap2', pct(0.03 - p2 @ Pi7, 2))
put('ex2.lamA', fmt(lam2[0], 3)); put('ex2.lamB', fmt(lam2[1], 3))
put('ex2.p1mu', pct(p1 @ mu2_, 2)); put('ex2.p2mu', pct(p2 @ mu2_, 2))
put('ex2.omegaA', fmt(Om2[0, 0], 6)); put('ex2.omegaB', fmt(Om2[1, 1], 6))
corr_views = (P2 @ S7 @ P2.T)[0, 1] / np.sqrt((P2 @ S7 @ P2.T)[0, 0] * (P2 @ S7 @ P2.T)[1, 1])
put('ex2.viewcorr', fmt(corr_views, 3))
put('ex2.wsum', pct(ws2.sum(), 1)); put('ex2.w1sum', pct(ws1.sum(), 1))
put('ex2.CAN', pct(ws2[CAN])); put('ex2.USA', pct(ws2[USA]))

# 例 3：カナダのビューを強める
Q3 = np.array([0.05, 0.04])
mu3, M3, ws3, lam3 = run_example(P2, Q3, Om2)
example_table('ex3', P2, mu3, ws3)
put('ex3.lamA', fmt(lam3[0], 3)); put('ex3.lamB', fmt(lam3[1], 3))
# 例 4：ドイツのビューの自信を下げる
Om4 = Om2.copy(); Om4[0, 0] *= 2
mu4, M4, ws4, lam4 = run_example(P2, Q3, Om4)
example_table('ex4', P2, mu4, ws4)
put('ex4.lamA', fmt(lam4[0], 3)); put('ex4.lamB', fmt(lam4[1], 3))
put('ex4.p1mu', pct(p1 @ mu4, 2)); put('ex3.p1mu', pct(p1 @ mu3, 2))
# 例 5：ビュー 2 の λ が 0 になる Q2
f = lambda q: hl_lambda(P2, np.array([0.05, q]), Om2, S7, G7_W, TAU, DELTA)[1]
slope = f(1.0) - f(0.0)
q_star = -f(0.0) / slope
mu5, M5, ws5, lam5 = run_example(P2, np.array([0.05, q_star]), Om2)
example_table('ex5', P2, mu5, ws5)
assert abs(lam5[1]) < 1e-12
put('ex5.qstar', pct(q_star, 2)); put('ex5.lamA', fmt(lam5[0], 3)); put('ex5.lamB', fmt(lam5[1], 3))
put('ex5.p2Pi', pct(p2 @ Pi7, 2)); put('ex5.p2Pitau', pct(p2 @ Pi7 / (1 + TAU), 2))
put('ex5.p2mu1', pct(p2 @ mu1, 2)); put('ex5.p2mu1tau', pct(p2 @ mu1 / (1 + TAU), 2))
put('ex5.slope', fmt(slope / 100, 3))
# まとめの表
rows = []
for name, Q_, lam_, om_ in [('例 2（基準）', Q2, lam2, Om2), ('例 3（カナダを 4\\% に）', Q3, lam3, Om2),
                            ('例 4（例 3 ＋ドイツの $\\omega$ を 2 倍）', Q3, lam4, Om4), ('例 5（$\\lambda_2 = 0$ になる $Q_2$）', np.array([0.05, q_star]), lam5, Om2)]:
    rows.append([name, pct(Q_[0], 2) + '\\%', pct(Q_[1], 2) + '\\%', sci(om_[0, 0], 2), fmt(lam_[0], 3), fmt(lam_[1], 3)])
write_table('ex_summary', tabular('lrrrrr', ['', '$Q_1$', '$Q_2$', '$\\omega_1$', '$\\lambda_1$', '$\\lambda_2$'], rows))

# 図：λ2 と Q2、ω2
fig, ax = plt.subplots(figsize=(5.9, 2.9))
qs = np.linspace(-0.02, 0.06, 50)
for mult, c, lab in [(0.5, C3, r'$\omega_2$ を半分（自信が強い）'), (1.0, C1, r'$\omega_2 = p_2\,\tau\Sigma\,p_2^{T}$（論文の設定）'), (2.0, C2, r'$\omega_2$ を 2 倍（自信が弱い）')]:
    Om_ = Om2.copy(); Om_[1, 1] *= mult
    ax.plot(100 * qs, [hl_lambda(P2, np.array([0.05, q]), Om_, S7, G7_W, TAU, DELTA)[1] for q in qs], color=c, label=lab)
ax.axvline(100 * q_star, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
ax.annotate(f'λ₂ = 0（Q₂ = {100*q_star:.2f}%）', (100 * q_star, 0), xytext=(6, -16), textcoords='offset points', fontsize=7.5, color=INK2)
ax.axhline(0, color=INK2, linewidth=0.6)
ax.set_xlabel('ビュー 2 の値 Q₂（カナダ − アメリカ、年率 %）'); ax.set_ylabel('ビュー 2 の重み λ₂')
ax.legend(loc='upper left', fontsize=7.5)
save_fig(fig, 'lambda')

# 図：例 2 の比率
fig, ax = plt.subplots(figsize=(5.9, 2.8))
bars(ax, G7_JA, [100 * G7_W, 100 * ws2], [C1, C2], [r'市場の比率 $w_{\mathrm{eq}}$', r'BL の最適比率 $w^*$（例 2）'])
ax.set_ylabel('比率（%）')
ax.legend(loc='upper left', fontsize=7.5)
save_fig(fig, 'ex2_weights')

# ============================================================
# 第 6 章：制約、τ、Σ と Σ+M
# ============================================================
Sp2 = S7 + M2_
w_lo_sp = qp(mu2_, Sp2, DELTA, long_only=True)
w_lo_s = qp(mu2_, S7, DELTA, long_only=True)
w_bud_sp, _ = budget_mv(mu2_, Sp2, DELTA)
w_free_s = np.linalg.solve(DELTA * S7, mu2_)
rows = [[G7_JA[i], pct(G7_W[i]), pct(ws2[i]), pct(w_free_s[i]), pct(w_bud_sp[i]), pct(w_lo_sp[i])] for i in range(n7)]
rows.append(['合計'] + [pct(v.sum()) for v in [G7_W, ws2, w_free_s, w_bud_sp, w_lo_sp]])
write_table('constraints', tabular('lrrrrr', ['国', '$w_{\\mathrm{eq}}$', '制約なし $\\Sigma{+}M$', '制約なし $\\Sigma$', '合計 1', '合計 1・空売りなし'], rows, midrules=(n7,)))
put('lo.CAN', pct(w_lo_sp[CAN])); put('lo.GER', pct(w_lo_sp[GER])); put('lo.USA', pct(w_lo_sp[USA]))
put('lo.diffSSp', pct(np.abs(w_lo_sp - w_lo_s).max(), 2))
# τ を変える
rows = []
for t in [0.01, 0.025, 0.05, 0.25, 1.0]:
    P_ = P2; Om_ = np.diag(np.diag(P_ @ (t * S7) @ P_.T))
    mu_, M_ = bl_update(Pi7, S7, P_, Q2, Om_, t)
    ws_ = np.linalg.solve(DELTA * (S7 + M_), mu_)
    rows.append([fmt(t, 3), pct(mu_[GER], 2), pct(mu_[CAN], 2), pct(mu_[USA], 2), pct(ws_[GER]), pct(ws_[CAN]), pct(ws_.sum())])
write_table('tau', tabular('rrrrrrr', ['$\\tau$', '$\\bar\\mu_{\\mathrm{GER}}$', '$\\bar\\mu_{\\mathrm{CAN}}$', '$\\bar\\mu_{\\mathrm{USA}}$',
                                      '$w^*_{\\mathrm{GER}}$', '$w^*_{\\mathrm{CAN}}$', '$\\sum w^*$'], rows))
# τ を変えて Ω は固定
rows = []
for t in [0.01, 0.025, 0.05, 0.25]:
    mu_, M_ = bl_update(Pi7, S7, P2, Q2, Om2, t)
    k_ = (P2 @ (t * S7) @ P2.T)[0, 0] / ((P2 @ (t * S7) @ P2.T)[0, 0] + Om2[0, 0])
    rows.append([fmt(t, 3), fmt(k_, 3), pct(p1 @ mu_, 2), pct(mu_[GER], 2), pct(mu_[CAN], 2)])
write_table('tau_fixedomega', tabular('rrrrr', ['$\\tau$', 'ビュー 1 だけのときの採用率', '$p_1\\bar\\mu$', '$\\bar\\mu_{\\mathrm{GER}}$', '$\\bar\\mu_{\\mathrm{CAN}}$'], rows))

# ============================================================
# 第 7 章：本プロジェクト
# ============================================================
conn = connect(config.DB_PATH)
D_TAU, D_DELTA = config.TAU, config.RISK_AVERSION
put('cfg.tau', fmt(D_TAU, 3)); put('cfg.delta', fmt(D_DELTA, 1)); put('cfg.window', str(config.COV_WINDOW_DAYS))
put('cfg.minhist', fmt(config.MIN_HISTORY_RATIO, 1)); put('cfg.origrf', fmt(config.ORIGINAL_RISK_FREE_RATE, 2))
put('cfg.origrw', fmt(config.ORIGINAL_RETURN_WEIGHT, 1))
DATE = '2025-01-31'
put('pj.date', DATE)
inp = load_inputs(conn, DATE, config.LLM_RUN_ID, config.COV_WINDOW_DAYS, config.MIN_HISTORY_RATIO)
rf = inp.risk_free_daily
put('pj.rf', pct(rf, 4)); put('pj.rfyear', pct(rf * 252, 2))
put('pj.N', str(len(inp.market_weights))); put('pj.days', str(len(inp.returns))); put('pj.mdays', str(len(inp.month_returns)))

# ---- 5 銘柄
T5 = ['AAPL', 'NVDA', 'JPM', 'XOM', 'JNJ']
NAMES5 = {'AAPL': 'Apple', 'NVDA': 'NVIDIA', 'JPM': 'JPMorgan Chase', 'XOM': 'Exxon Mobil', 'JNJ': 'Johnson \\& Johnson'}
r5 = inp.returns[T5]
mean5, sd5 = r5.mean(), r5.std()
rows = [[t, NAMES5[t], pct(mean5[t], 3), pct(sd5[t], 2), pct(mean5[t] * 252, 1), pct(sd5[t] * np.sqrt(252), 1)] for t in T5]
write_table('pj_stats', tabular('llrrrr', ['銘柄', '会社', '日次の平均 (\\%)', '日次の標準偏差 (\\%)', '年率の平均 (\\%)', '年率の標準偏差 (\\%)'], rows))
corr5 = r5.corr()
rows = [[t] + [f'{corr5.loc[t, u]:.2f}' for u in T5] for t in T5]
write_table('pj_corr', tabular('l' + 'r' * 5, [''] + T5, rows))
lw = LedoitWolf().fit(r5.values)
S5 = shrunk_covariance(r5)
assert np.allclose(S5.values, lw.covariance_)
S_mle = np.cov(r5.values.T, bias=True)
target = np.trace(S_mle) / 5
put('pj.alpha5', fmt(lw.shrinkage_, 3)); put('pj.target5', fmt(1e4 * target, 3))
rows = []
for t in T5:
    i = T5.index(t)
    rows.append([t, fmt(1e4 * S_mle[i, i], 3), fmt(1e4 * S5.values[i, i], 3), fmt(1e4 * S_mle[i, 0], 3), fmt(1e4 * S5.values[i, 0], 3)])
write_table('pj_lw', tabular('lrrrr', ['銘柄 $i$', '標本 $S_{ii}$', 'LW $\\hat\\Sigma_{ii}$', '標本 $S_{i,\\mathrm{AAPL}}$', 'LW $\\hat\\Sigma_{i,\\mathrm{AAPL}}$'], rows))
w5 = inp.market_weights[T5] / inp.market_weights[T5].sum()
Pi5 = implied_returns(S5, w5, D_DELTA)
var5 = w5 @ S5 @ w5
beta5 = (S5 @ w5) / var5
rows = [[t, pct(inp.market_weights[t], 2), pct(w5[t], 1), fmt(1e4 * (S5 @ w5)[t], 3), fmt(beta5[t], 2), pct(Pi5[t], 4)] for t in T5]
write_table('pj_prior', tabular('lrrrrr', ['銘柄', '指数の中の比率 (\\%)', '5 銘柄の中の比率 $w_i$ (\\%)', '$(\\hat\\Sigma w)_i \\times 10^4$', '$\\beta_i$', '$\\Pi_i$ (\\%/日)'], rows))
put('pj.var5', fmt(1e4 * var5, 3)); put('pj.Pim5', pct(w5 @ Pi5, 4)); put('pj.Pim5year', pct(252 * (w5 @ Pi5), 1))
put('pj.sigm5year', pct(np.sqrt(252 * var5), 1))
put('pj.PiAAPL', pct(Pi5['AAPL'], 4)); put('pj.SwAAPL', fmt(1e4 * (S5 @ w5)['AAPL'], 3))
put('pj.PiJNJ', pct(Pi5['JNJ'], 4)); put('pj.betaJNJ', fmt(beta5['JNJ'], 2)); put('pj.PiNVDA', pct(Pi5['NVDA'], 4))
# LLM の回答
ans = pd.read_sql("SELECT symbol, sample_no, expected_return FROM llm_views WHERE run_id = ? AND as_of_date = ? AND symbol IN (%s)"
                  % ','.join('?' * 5), conn, params=[config.LLM_RUN_ID, DATE] + T5)
aapl = ans[ans.symbol == 'AAPL'].sort_values('sample_no').expected_return.values
cells = [pct(v, 3) for v in aapl]
write_table('pj_aapl', '\\begin{tabular}{r' + 'r' * 10 + '}\n\\toprule\n'
            + '\n'.join(f'{i + 1}--{i + 10} & ' + ' & '.join(cells[i:i + 10]) + ' \\\\' for i in range(0, 30, 10))
            + '\n\\bottomrule\n\\end{tabular}\n')
mon5 = inp.month_returns[T5].mean()
rows = []
for t in T5:
    a = ans[ans.symbol == t].expected_return.dropna().values
    assert np.isclose(a.var(), inp.view_variance[t]) and np.isclose(a.mean(), inp.view_mean[t])
    rows.append([t, str(len(a)), pct(mon5[t], 3), pct(a.mean(), 3), pct(a.std(), 3), pct(a.min(), 3), pct(a.max(), 3), fmt(1e6 * a.var(), 3)])
write_table('pj_answers', tabular('lrrrrrrr', ['銘柄', '回答数', '1 月の平均 (\\%)', '回答の平均 (\\%)', '回答の標準偏差 (\\%)', '最小 (\\%)', '最大 (\\%)', '分散 $\\times 10^6$'], rows))
put('pj.aaplmean', pct(aapl.mean(), 3)); put('pj.aaplsd', pct(aapl.std(), 3)); put('pj.aaplvar', fmt(1e6 * aapl.var(), 3))
put('pj.aaplmon', pct(mon5['AAPL'], 3))
Q5 = inp.view_mean[T5] - rf
om5 = inp.view_variance[T5]
ts5 = D_TAU * np.diag(S5.values)
ratio5 = om5.values / ts5
k5 = 1 / (1 + ratio5)
rows = [[t, fmt(1e6 * ts5[i], 3), fmt(1e6 * om5[t], 3), fmt(ratio5[i], 3), fmt(k5[i], 3)] for i, t in enumerate(T5)]
write_table('pj_ratio', tabular('lrrrr', ['銘柄', '$\\tau\\hat\\Sigma_{ii} \\times 10^6$', '$\\omega_i \\times 10^6$', '$\\omega_i / (\\tau\\hat\\Sigma_{ii})$', '目安の採用率 $k_i$'], rows))
put('pj.kmin', fmt(k5.min(), 2)); put('pj.kmax', fmt(k5.max(), 2))
mu5 = posterior_returns(Pi5, S5, Q5, om5, D_TAU)
mu5_info = posterior_returns_original(Pi5, S5, Q5, om5, D_TAU)
put('pj.formdiff', sci(np.abs(mu5 - mu5_info).max(), 1))
move5 = (mu5 - Pi5) / (Q5 - Pi5)
rows = [[t, pct(Pi5[t], 4), pct(Q5[t], 4), pct(Q5[t] - Pi5[t], 4), pct(mu5[t], 4), pct(mu5[t] - Pi5[t], 4), fmt(move5[t], 3)] for t in T5]
write_table('pj_post', tabular('lrrrrrr', ['銘柄', '$\\Pi_i$', '$Q_i$', '$Q_i - \\Pi_i$', '$\\bar\\mu_i$', '$\\bar\\mu_i - \\Pi_i$', '動いた割合'], rows))
put('pj.QAAPL', pct(Q5['AAPL'], 4)); put('pj.muAAPL', pct(mu5['AAPL'], 4)); put('pj.QJPM', pct(Q5['JPM'], 4)); put('pj.muJPM', pct(mu5['JPM'], 4))
put('pj.moveXOM', fmt(move5['XOM'], 3)); put('pj.kXOM', fmt(k5[T5.index('XOM')], 3))
put('pj.QXOM', pct(Q5['XOM'], 4)); put('pj.muXOM', pct(mu5['XOM'], 4)); put('pj.PiXOM', pct(Pi5['XOM'], 4))
# AAPL を手で
iA = 0
put('pj.tsAA', fmt(1e6 * ts5[iA], 3)); put('pj.omAA', fmt(1e6 * om5['AAPL'], 3)); put('pj.kAA', fmt(k5[iA], 3))
put('pj.diagAAPL', pct(Pi5['AAPL'] + k5[iA] * (Q5['AAPL'] - Pi5['AAPL']), 4))
put('pj.gapAAPL', pct(Q5['AAPL'] - Pi5['AAPL'], 4))
# 比率
w_bl5 = optimize(mu5, S5, D_DELTA)
w_eqchk = optimize(Pi5, S5, D_DELTA)
put('pj.eqcheck', sci(np.abs(w_eqchk - w5).max(), 1))
w_free5 = np.linalg.solve(D_DELTA * S5.values, mu5.values)
w_bud5, _ = budget_mv(mu5.values, S5.values, D_DELTA)
w_mv5 = optimize(mean5 - rf, S5, D_DELTA)
P5 = np.eye(5)
mu5b, M5b = bl_update(Pi5.values, S5.values, P5, Q5.values, np.diag(om5.values), D_TAU)
assert np.allclose(mu5b, mu5.values)
w_hl5 = np.linalg.solve(D_DELTA * (S5.values + M5b), mu5b)
lam5p = (1 + D_TAU) * w_hl5 - w5.values
rows = [[t, pct(w5[t]), pct(w_free5[i]), pct(w_bud5[i]), pct(w_bl5[t]), pct(w_mv5[t])] for i, t in enumerate(T5)]
rows.append(['合計', pct(w5.sum()), pct(w_free5.sum()), pct(w_bud5.sum()), pct(w_bl5.sum()), pct(w_mv5.sum())])
write_table('pj_weights', tabular('lrrrrr', ['銘柄', '市場 $w$', 'BL 制約なし', 'BL 合計 1', 'BL 合計 1・空売りなし', '平均分散（過去 1 年）'], rows, midrules=(5,)))
put('pj.wfreesum', pct(w_free5.sum(), 0))
# 最適解での限界効用（KKT 条件）：持つ銘柄は μ̄_i - δ(Σw)_i が等しく、持たない銘柄はそれ以下
grad5 = mu5 - D_DELTA * (S5 @ w_bl5)
rows = [[t, pct(mu5[t], 4), pct(D_DELTA * (S5 @ w_bl5)[t], 4), pct(grad5[t], 4)] for t in T5]
write_table('pj_kkt', tabular('lrrr', ['銘柄', '$\\bar\\mu_i$', '$\\delta(\\hat\\Sigma w)_i$', '$\\bar\\mu_i - \\delta(\\hat\\Sigma w)_i$'], rows))
put('pj.penJPM', pct(D_DELTA * (S5 @ w_bl5)['JPM'], 4))
put('pj.corrXOMJPM', fmt(corr5.loc['XOM', 'JPM'], 2))
put('pj.wblNVDA', pct(w_bl5['NVDA'])); put('pj.wblJPM', pct(w_bl5['JPM'])); put('pj.wblJNJ', pct(w_bl5['JNJ']))
put('pj.wblAAPL', pct(w_bl5['AAPL'])); put('pj.wblXOM', pct(w_bl5['XOM']))
rows = [[t, pct(w5[t]), pct(w_hl5[i]), fmt(lam5p[i], 3)] for i, t in enumerate(T5)]
write_table('pj_hl', tabular('lrrr', ['銘柄', '$w_{\\mathrm{eq}}$', '$w^* = (\\delta(\\hat\\Sigma + M))^{-1}\\bar\\mu$', '$\\Lambda_i = (1+\\tau)w^*_i - w_{\\mathrm{eq},i}$'], rows))
put('pj.lamAAPL', fmt(lam5p[0], 2)); put('pj.lamJPM', fmt(lam5p[2], 2))
# 図：5 銘柄の Π、Q、μ̄
fig, ax = plt.subplots(figsize=(5.9, 2.8))
bars(ax, T5, [100 * Pi5.values, 100 * Q5.values, 100 * mu5.values], [C1, C2, C3],
     [r'均衡リターン $\Pi$', r'LLM のビュー $Q$', r'BL の事後リターン $\bar{\mu}$'], width=0.27)
ax.set_ylabel('1 日あたり（%）')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=3)
save_fig(fig, 'pj_post')

# ---- 元の実装の計算を 5 銘柄で
m5 = inp.month_returns[T5]
Sm5 = sample_covariance(m5)
Pi_orig5 = capm_returns_original(m5, w5, config.ORIGINAL_RISK_FREE_RATE)
mu_orig5 = posterior_returns_original(Pi_orig5, Sm5, inp.view_mean[T5], om5, D_TAU)
w_orig5 = optimize_original(mu_orig5, Sm5, config.ORIGINAL_RETURN_WEIGHT)
w_orig5_exact = optimize(mu_orig5, Sm5, 2 / config.ORIGINAL_RETURN_WEIGHT)
mkt_m = (m5 * w5).sum(axis=1)
put('or.mktmean', pct(mkt_m.mean(), 3)); put('or.prem', pct((mkt_m - config.ORIGINAL_RISK_FREE_RATE).mean(), 3))
ratio_o = om5.values / (D_TAU * np.diag(Sm5.values))
move_o = (mu_orig5 - Pi_orig5) / (inp.view_mean[T5] - Pi_orig5)
rows = [[t, fmt((m5[t].cov(mkt_m) / mkt_m.var()), 2), pct(Pi_orig5[t], 3), pct(inp.view_mean[t], 3), pct(mu_orig5[t], 3), fmt(ratio_o[i], 4), fmt(move_o[t], 4), pct(w_orig5[t]), pct(w_orig5_exact[t])]
        for i, t in enumerate(T5)]
write_table('pj_orig', tabular('lrrrrrrrr', ['銘柄', '$\\beta_i$', '$\\Pi^{\\mathrm{orig}}_i$', '$Q_i$', '$\\bar\\mu_i$', '$\\omega_i/(\\tau S_{ii})$', '動いた割合', '比率 (元の解法)', '比率 (最後まで解く)'], rows))
put('or.days', str(len(m5)))
# 元の最適化（SciPy の SLSQP、標準の設定）が何回で止まったか
from scipy.optimize import minimize
f_orig = lambda x: x @ Sm5.values @ x - config.ORIGINAL_RETURN_WEIGHT * (x @ mu_orig5.values)
res = minimize(f_orig, np.ones(5) / 5, constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}),
               bounds=tuple((0, 1) for _ in range(5)))
assert np.allclose(res.x, w_orig5.values, atol=1e-6)
put('or.nit', str(res.nit)); put('or.f0', sci(f_orig(np.ones(5) / 5), 1)); put('or.fexact', sci(f_orig(w_orig5_exact.values), 1))
put('or.wJPMexact', pct(w_orig5_exact['JPM']))
put('or.PiAAPL', pct(Pi_orig5['AAPL'], 3))

# ---- Ω = 0 の落とし穴（2 銘柄の例）
S_t = np.array([[4e-4, 1e-4], [1e-4, 4e-4]]); Pi_t = pd.Series([3e-4, 3e-4], index=['A', 'B'])
Q_t = pd.Series([2e-3, 2e-3], index=['A', 'B']); om_t = pd.Series([1e-6, 0.0], index=['A', 'B'])
mu_t = posterior_returns(Pi_t, pd.DataFrame(S_t, index=['A', 'B'], columns=['A', 'B']), Q_t, om_t, D_TAU)
mu_t_o = posterior_returns_original(Pi_t, pd.DataFrame(S_t, index=['A', 'B'], columns=['A', 'B']), Q_t, om_t, D_TAU)
put('z.muA', pct(mu_t['A'], 4)); put('z.muB', pct(mu_t['B'], 4)); put('z.muAo', pct(mu_t_o['A'], 4)); put('z.muBo', pct(mu_t_o['B'], 4))
min_var = conn.execute("SELECT MIN(variance) FROM llm_view_stats WHERE run_id = ?", (config.LLM_RUN_ID,)).fetchone()[0]
put('z.minvar', sci(min_var, 1))

# ---- 全銘柄（同じ判断日）
S_all = shrunk_covariance(inp.returns)
alpha_all = LedoitWolf().fit(inp.returns.values).shrinkage_
put('all.alpha', fmt(alpha_all, 3))
Pi_all = implied_returns(S_all, inp.market_weights, D_DELTA)
Q_all = inp.view_mean - rf
om_all = inp.view_variance
mu_all = posterior_returns(Pi_all, S_all, Q_all, om_all, D_TAU)
ts_all = D_TAU * np.diag(S_all.values)
ratio_all = om_all.values / ts_all
k_all = 1 / (1 + ratio_all)
gap = (Q_all - Pi_all)
move_all = ((mu_all - Pi_all) / gap)[gap.abs() > 1e-4]
w_all = optimize(mu_all, S_all, D_DELTA)
stored = pd.read_sql("SELECT symbol, weight FROM portfolio_weights WHERE portfolio_id = 4 AND as_of_date = ?", conn,
                     params=(DATE,), index_col='symbol')['weight']
put('all.storeddiff', pct((w_all - stored.reindex(w_all.index)).abs().max(), 3))
put('all.ratiomed', fmt(np.median(ratio_all), 3)); put('all.ratioq1', fmt(np.quantile(ratio_all, 0.25), 3)); put('all.ratioq3', fmt(np.quantile(ratio_all, 0.75), 3))
put('all.kmed', fmt(np.median(k_all), 3)); put('all.kq1', fmt(np.quantile(k_all, 0.25), 3)); put('all.kq3', fmt(np.quantile(k_all, 0.75), 3))
put('all.movemed', fmt(np.median(move_all), 3))
put('all.corrQ', fmt(np.corrcoef(mu_all, Q_all)[0, 1], 3)); put('all.corrPi', fmt(np.corrcoef(mu_all, Pi_all)[0, 1], 3))
put('all.Pimed', pct(np.median(Pi_all), 4)); put('all.Qmed', pct(np.median(Q_all), 3)); put('all.Qsd', pct(Q_all.std(), 3))
put('all.Pisd', pct(Pi_all.std(), 4))
put('all.shareBig', pct(((Q_all - Pi_all).abs() > 0.001).mean(), 0))
put('all.topQmonth', pct((1 + Q_all[mu_all.idxmax()]) ** 21 - 1, 0)); put('all.topQ', pct(Q_all[mu_all.idxmax()], 2)); put('all.topname', mu_all.idxmax())
put('all.sdmed', pct(np.median(np.sqrt(np.diag(S_all.values))), 2))
put('all.semonth', pct(np.median(np.sqrt(np.diag(S_all.values))) / np.sqrt(21), 3))
put('all.effmkt', fmt(eff_n(inp.market_weights), 1)); put('all.effbl', fmt(eff_n(w_all), 2)); put('all.nheld', str(int((w_all > 1e-4).sum())))
top = w_all[mu_all.sort_values(ascending=False).head(6).index]
rows = [[s, pct(w, 1), pct(inp.market_weights[s], 2), pct(Pi_all[s], 3), pct(Q_all[s], 3), pct(mu_all[s], 3), fmt(k_all[list(inp.market_weights.index).index(s)], 3)] for s, w in top.items()]
write_table('all_top', tabular('lrrrrrr', ['銘柄', 'BL の比率 (\\%)', '市場の比率 (\\%)', '$\\Pi_i$ (\\%/日)', '$Q_i$ (\\%/日)', '$\\bar\\mu_i$ (\\%/日)', '$k_i$'], rows))
# 図：採用率の分布
fig, ax = plt.subplots(figsize=(5.9, 2.6))
ax.hist(k_all, bins=np.linspace(0, 1, 41), color=C1, edgecolor='white', linewidth=0.8, zorder=2)
ax.axvline(0.5, color=C2, linewidth=1.2, linestyle=(0, (4, 2)))
ax.annotate('He と Litterman の設定（k = 0.5）', (0.5, ax.get_ylim()[1] * 0.85), xytext=(-5, 0), textcoords='offset points', fontsize=7.5, color=INK2, ha='right')
ax.set_xlabel(f'目安の採用率 k = τσ² / (τσ² + ω)（{DATE}、{len(k_all)} 銘柄）'); ax.set_ylabel('銘柄の数')
ax.grid(axis='x', visible=False)
save_fig(fig, 'pj_adoption')

# ---- 30 か月の判断日
rows = []
meds = []
for d in load_as_of_dates(conn, config.LLM_RUN_ID):
    x = load_inputs(conn, d, config.LLM_RUN_ID, config.COV_WINDOW_DAYS, config.MIN_HISTORY_RATIO)
    Sx = shrunk_covariance(x.returns)
    rx = x.view_variance.values / (D_TAU * np.diag(Sx.values))
    meds.append((d, len(rx), np.median(rx), np.median(1 / (1 + rx))))
meds = pd.DataFrame(meds, columns=['date', 'n', 'ratio', 'k'])
put('dates.n', str(len(meds)))
put('dates.kmin', fmt(meds.k.min(), 2)); put('dates.kmax', fmt(meds.k.max(), 2))
put('dates.rmin', fmt(meds.ratio.min(), 2)); put('dates.rmax', fmt(meds.ratio.max(), 2))
# 採用率の図に、この範囲を重ねる
ax_k = plt.figure(figsize=(5.9, 2.7)).add_subplot()
fig = ax_k.figure
ax_k.axvspan(meds.ratio.min(), meds.ratio.max(), color=C2, alpha=0.15, linewidth=0, zorder=1)
ax_k.plot(r, 1 / (1 + r), color=C1, zorder=3)
ax_k.scatter([1], [0.5], s=40, color=C1, edgecolor='white', linewidth=1.2, zorder=4)
ax_k.annotate('He と Litterman の設定\n' + r'$\omega = p\,\tau\Sigma\,p^{T}$ → $k = 0.5$', (1, 0.5), xytext=(10, 8), textcoords='offset points', fontsize=7.5, color=INK2)
ax_k.annotate(f'本プロジェクトの中央値の範囲\n（{len(meds)} か月、k = {meds.k.min():.2f}〜{meds.k.max():.2f}）',
              (np.sqrt(meds.ratio.min() * meds.ratio.max()), 0.25), ha='center', fontsize=7.5, color=INK2)
ax_k.set_xscale('log'); ax_k.set_ylim(0, 1.02); ax_k.set_xlim(0.01, 100)
ax_k.set_xlabel('ビューの不確かさ ÷ 均衡の不確かさ　ω / (τσ²)'); ax_k.set_ylabel('採用率 k')
save_fig(fig, 'adoption')

# ============================================================
# 書き出し
# ============================================================
lines = ['% BLM解説_計算.py が作ったファイル（手で直さない）']
for k, v in VALUES.items():
    lines.append(f'\\expandafter\\def\\csname v@{k}\\endcsname{{{v}}}')
(OUT / 'values.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'{len(VALUES)} values, tables and figures written to {OUT}')
