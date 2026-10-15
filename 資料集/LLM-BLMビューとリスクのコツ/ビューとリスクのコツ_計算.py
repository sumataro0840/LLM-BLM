"""
「Black-Litterman のビューとリスクのコツ」（勉強用資料）の数値・表・図を作る。

    （リポジトリの直下で）.venv\\Scripts\\python.exe 資料集\\LLM-BLMビューとリスクのコツ\\ビューとリスクのコツ_計算.py

作るもの（生成/ に置く。手で直さない）:
    values.tex   本文で使う数値（本文では \\V{名前} で参照する）
    tab_*.tex    表
    fig_*.pdf    図
使うデータ:
    .env の DB_PATH の DB   株価・指数・構成銘柄・LLM の回答（実行番号 1）
    データ/*.csv              Fama-French のファクター（月次）、モメンタム（日次）、EPU、GPR（出どころは本文の付録）
"""
import sys
import time
from pathlib import Path

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd
from scipy import optimize, stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
OUT = HERE / '生成'
OUT.mkdir(exist_ok=True)
DATA = HERE / 'データ'

from marketdb import config as mconfig
from marketdb.db.connection import connect
from marketdb.db.market import load_prices
from marketdb.db.sector import load_member_caps, load_month_ends, load_sector_etfs

RNG = np.random.default_rng(20261015)
VALUES = {}
SHARED = {}   # 節をまたいで使う数値


# ============================================================
# 書式
# ============================================================
def fmt(x, d):
    """小数 d 桁。負の数のマイナスは数式の記号にする（-0.0 は 0.0 にする）。"""
    x = float(x)
    if round(x, d) == 0:
        x = 0.0
    s = f'{abs(x):,.{d}f}'.replace(',', '{,}')
    return ('\\ensuremath{-}' if x < 0 else '') + s


def sfmt(x, d):
    """符号つき（+ も付ける）。"""
    return ('+' if round(float(x), d) > 0 else '') + fmt(x, d)


def pct(x, d=1):
    return fmt(100 * x, d)


def spct(x, d=1):
    return sfmt(100 * x, d)


def sci(x, d=2):
    """a × 10^b の形。"""
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f'\\ensuremath{{{fmt(m, d)}\\times 10^{{{e}}}}}'


def put(key, text):
    VALUES[key] = str(text)


def write_table(name, text):
    (OUT / f'tab_{name}.tex').write_text(text, encoding='utf-8')


def tabular(colspec, header, rows, midrules=()):
    """header: セルのリスト。rows: セルのリストのリスト。midrules: その行の前に \\midrule を入れる行番号。"""
    lines = [f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for i, row in enumerate(rows):
        if i in midrules:
            lines.append('\\midrule')
        lines.append(' & '.join(str(c) for c in row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


# ============================================================
# 図の体裁（総合報告書・BLM解説と同じ配色。系列は固定の順で、3〜4 色まで）
# ============================================================
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
NEUTRAL = '#b9b8b2'
plt.rcParams.update({
    'font.family': 'Yu Gothic', 'font.size': 8.5,
    'axes.edgecolor': BASE, 'axes.labelcolor': INK, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 1.6,
    'pdf.fonttype': 3, 'axes.unicode_minus': False, 'axes.titlesize': 8.5, 'axes.titleweight': 'normal',
})


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'fig_{name}.pdf')
    check = __import__('os').environ.get('FIG_CHECK_DIR')
    if check:   # 目で確かめるための PNG（資料には使わない）
        fig.savefig(Path(check) / f'fig_{name}.png', dpi=110)
    plt.close(fig)


def bars(ax, labels, series, colors, names, width=0.8):
    """グループの棒グラフ。棒どうしに小さなすき間を空ける。"""
    x = np.arange(len(labels))
    k = len(series)
    w = width / k
    for j, (vals, color, name) in enumerate(zip(series, colors, names)):
        ax.bar(x + (j - (k - 1) / 2) * w, vals, w * 0.88, color=color, label=name, zorder=2)
    ax.set_xticks(x, labels)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.grid(axis='x', visible=False)


def end_labels(ax, x, items, gap_frac=0.06):
    """線の右端に値を書く。近い値どうしは上下にずらして重ならないようにする。items: [(y, 文字列)]"""
    lo, hi = ax.get_ylim()
    gap = gap_frac * (hi - lo)
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, text), y in zip(items, ys):
        ax.text(x, y, text, fontsize=7, color=INK2, va='center')


def date_axis(ax, start='2015', end='2027', step=2):
    ticks = pd.date_range(f'{start}-01-01', f'{end}-01-01', freq=f'{12 * step}MS')
    ax.set_xticks(ticks, [t.strftime('%Y') for t in ticks])


# ============================================================
# Black-Litterman の計算
# ============================================================
def bl(Pi, S, P, Q, Om, tau):
    """元の BL（期待値へのビュー）。事後の平均と、平均の不確かさ M を返す。"""
    tS = tau * S
    A = P @ tS @ P.T + Om
    K = tS @ P.T @ np.linalg.inv(A)
    return Pi + K @ (Q - P @ Pi), tS - K @ P @ tS


def bl_market(pi, S, P, v, Om):
    """Meucci の市場版（実現値へのビュー）。事後の平均と共分散を返す（τ は出てこない）。"""
    A = P @ S @ P.T + Om
    K = S @ P.T @ np.linalg.inv(A)
    return pi + K @ (v - P @ pi), S - K @ P @ S


def mv_long_only(mu, S, lam, cap=None):
    """w'mu - lam/2 w'Sw の最大化（全額投資・空売りなし・上限 cap）。"""
    n = len(mu)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0] + ([w <= cap] if cap is not None else [])
    scale = 1 / max(np.abs(mu).max(), np.abs(np.diag(S)).max())
    prob = cp.Problem(cp.Maximize(scale * (mu @ w - lam / 2 * cp.quad_form(w, cp.psd_wrap(S)))), cons)
    prob.solve(solver=cp.CLARABEL)
    x = np.clip(w.value, 0, None)
    return x / x.sum()


def eff_n(w):
    return 1 / np.sum(np.asarray(w) ** 2)


def ols(y, X, white=True):
    """定数項を自分で入れる OLS。係数と（White の）標準誤差。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    if white:
        meat = X.T @ (X * (e ** 2)[:, None])
        V = XtX_inv @ meat @ XtX_inv
    else:
        V = XtX_inv * (e @ e) / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(V)), e


# ============================================================
# A. Idzorek（2004/2007）の 8 資産の例
# ============================================================
A_NAMES = ['米国債券', '外国債券', '米国大型グロース', '米国大型バリュー', '米国小型グロース', '米国小型バリュー',
           '先進国株式', '新興国株式']
A_W = np.array([0.1934, 0.2613, 0.1209, 0.1209, 0.0134, 0.0134, 0.2418, 0.0349])
A_PI_TABLE = np.array([0.08, 0.67, 6.41, 4.08, 7.43, 3.70, 4.80, 6.60]) / 100
A_S = np.array([
    [0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437],
    [0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535],
    [-0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039],
    [-0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854],
    [0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994],
    [0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235],
    [-0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064],
    [-0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.079958]])
A_P = np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                [-1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.9, -0.9, 0.1, -0.1, 0, 0]], float)
A_Q = np.array([0.0525, 0.0025, 0.02])
A_TAU = 0.025
A_C = np.array([0.25, 0.50, 0.65])
# Idzorek の表 6（新しい期待リターンと比率）。照合用
A_ER_TABLE = np.array([0.07, 0.50, 6.50, 4.32, 7.59, 3.94, 4.93, 6.84]) / 100
A_W_TABLE = np.array([29.88, 15.59, 9.35, 14.82, 1.04, 1.65, 27.81, 3.49]) / 100


def idz_weights(Pi, S, P, Q, Om, tau, lam):
    mu, _ = bl(Pi, S, P, Q, Om, tau)
    return mu, np.linalg.solve(lam * S, mu)


def idz_full_conf(Pi, S, P, Q, tau, lam):
    """Ω = 0（100% の確信）。"""
    tS = tau * S
    mu = Pi + tS @ P.T @ np.linalg.solve(P @ tS @ P.T, Q - P @ Pi)
    return mu, np.linalg.solve(lam * S, mu)


def sec_a():
    S, w = A_S, A_W
    Sw = S @ w
    lam = float(A_PI_TABLE @ Sw / (Sw @ Sw))
    Pi = lam * Sw
    put('idz.lam', fmt(lam, 3))
    put('idz.pidiff', fmt(100 * np.abs(Pi - A_PI_TABLE).max(), 3))
    put('idz.mktvar', fmt(w @ S @ w, 5))
    put('idz.mktsd', pct(np.sqrt(w @ S @ w), 2))
    put('idz.mktprem', pct(w @ Pi, 2))
    pSp = np.einsum('ij,jk,ik->i', A_P, S, A_P)
    Om_hl = np.diag(A_TAU * pSp)
    for k in range(3):
        put(f'idz.pSp{k + 1}', pct(pSp[k], 3))
        put(f'idz.omhl{k + 1}', fmt(Om_hl[k, k], 6))
        put(f'idz.ppi{k + 1}', pct(A_P[k] @ Pi, 2))
    mu_hl, w_hl = idz_weights(Pi, S, A_P, A_Q, Om_hl, A_TAU, lam)
    mu_100, w_100 = idz_full_conf(Pi, S, A_P, A_Q, A_TAU, lam)
    # 3 つの書き方（情報形・更新形・混合推定）が同じ値になることを確かめる
    tSi, Omi = np.linalg.inv(A_TAU * S), np.linalg.inv(Om_hl)
    mu_info = np.linalg.solve(tSi + A_P.T @ Omi @ A_P, tSi @ Pi + A_P.T @ Omi @ A_Q)
    Xg = np.vstack([np.eye(8), A_P])
    Vg = np.block([[A_TAU * S, np.zeros((8, 3))], [np.zeros((3, 8)), Om_hl]])
    Vgi = np.linalg.inv(Vg)
    mu_gls = np.linalg.solve(Xg.T @ Vgi @ Xg, Xg.T @ Vgi @ np.concatenate([Pi, A_Q]))
    put('idz.formdiff', sci(max(np.abs(mu_info - mu_hl).max(), np.abs(mu_gls - mu_hl).max()), 1))
    # He-Litterman の形：比率の変化は P の行の組み合わせ（ビューのない資産は動かない）
    Lam = (A_TAU / lam) * np.linalg.solve(A_P @ (A_TAU * S) @ A_P.T + Om_hl, A_Q - A_P @ Pi)
    put('idz.lam1', fmt(Lam[0], 4))
    put('idz.lam2', fmt(Lam[1], 4))
    put('idz.lam3', fmt(Lam[2], 4))
    put('idz.hldiff', sci(np.abs(w + A_P.T @ Lam - w_hl).max(), 1))
    put('idz.erdiff', fmt(100 * np.abs(mu_hl - A_ER_TABLE).max(), 3))
    put('idz.wdiff', fmt(100 * np.abs(w_hl - A_W_TABLE).max(), 2))
    put('idz.wsum', pct(w_hl.sum(), 2))
    # 黙示の確信度（3 つのビューを同時に入れたとき）
    implied = (w_hl - w) / np.where(np.abs(w_100 - w) > 1e-12, w_100 - w, np.nan)
    put('idz.impl1', pct(implied[6], 2))
    put('idz.impl2', pct(implied[1], 2))
    put('idz.impl3', pct(implied[2], 2))
    # 1 つずつ入れたとき（He-Litterman の Ω なら 50%）
    single = []
    for k in range(3):
        pk, qk = A_P[[k]], A_Q[[k]]
        _, wk = idz_weights(Pi, S, pk, qk, np.diag([A_TAU * pSp[k]]), A_TAU, lam)
        _, wk100 = idz_full_conf(Pi, S, pk, qk, A_TAU, lam)
        j = int(np.argmax(np.abs(pk[0])))
        single.append((wk[j] - w[j]) / (wk100[j] - w[j]))
    put('idz.single', ', '.join(pct(s, 1) for s in single))
    # Idzorek の方法：(1) 論文の手順 6（数値で探す）と (2) 閉じた式 ω = τ pΣp' (1-C)/C
    om_closed = A_TAU * pSp * (1 - A_C) / A_C
    om_num = []
    for k in range(3):
        pk, qk = A_P[[k]], A_Q[[k]]
        _, wk100 = idz_full_conf(Pi, S, pk, qk, A_TAU, lam)
        target = w + A_C[k] * (wk100 - w)

        def loss(log_om):
            _, wk = idz_weights(Pi, S, pk, qk, np.diag([np.exp(log_om)]), A_TAU, lam)
            return np.sum((target - wk) ** 2)

        res = optimize.minimize_scalar(loss, bounds=(np.log(1e-9), np.log(10)), method='bounded',
                                       options={'xatol': 1e-12})
        om_num.append(np.exp(res.x))
    om_num = np.array(om_num)
    put('idz.omnumdiff', fmt(100 * np.max(np.abs(om_num / om_closed - 1)), 4))
    for k in range(3):
        put(f'idz.omidz{k + 1}', fmt(om_closed[k], 6))
        put(f'idz.omratio{k + 1}', fmt(om_closed[k] / Om_hl[k, k], 2))
    mu_i, w_i = idz_weights(Pi, S, A_P, A_Q, np.diag(om_closed), A_TAU, lam)
    achieved = (w_i - w) / np.where(np.abs(w_100 - w) > 1e-12, w_100 - w, np.nan)
    put('idz.ach1', pct(achieved[6], 1))
    put('idz.ach2', pct(achieved[1], 1))
    put('idz.ach3', pct(achieved[2], 1))
    # 3 つの ω をまとめて探し、3 つ同時に入れたときの傾きを確信度に合わせる
    idx = [6, 1, 2]

    def resid(log_om):
        _, wj = idz_weights(Pi, S, A_P, A_Q, np.diag(np.exp(log_om)), A_TAU, lam)
        return (wj - w)[idx] / (w_100 - w)[idx] - A_C

    sol = optimize.least_squares(resid, np.log(om_closed), xtol=1e-14, ftol=1e-14)
    om_joint = np.exp(sol.x)
    _, w_j = idz_weights(Pi, S, A_P, A_Q, np.diag(om_joint), A_TAU, lam)
    ach_j = (w_j - w)[idx] / (w_100 - w)[idx]
    for k in range(3):
        put(f'idz.omjoint{k + 1}', fmt(om_joint[k] * 1e4, 2))
        put(f'idz.achj{k + 1}', pct(ach_j[k], 1))
    put('idz.joint.res', sci(np.abs(sol.fun).max(), 1))
    # 表：入力
    rows = [[A_NAMES[i], pct(w[i], 2), pct(np.sqrt(S[i, i]), 1), pct(A_PI_TABLE[i], 2), pct(Pi[i], 3)] for i in range(8)]
    write_table('idz_inputs', tabular('lrrrr', ['資産クラス', '時価総額の比率', '標準偏差', '$\\Pi$（論文）',
                                                 '$\\Pi=\\lambda\\Sigma w$'], rows))
    # 表：ビュー
    vnames = ['先進国株式は 5.25\\%', '外国債券 $-$ 米国債券 $=$ 0.25\\%', 'グロース $-$ バリュー $=$ 2\\%']
    rows = [[f'{k + 1}', vnames[k], pct(A_P[k] @ Pi, 2), pct(A_Q[k], 2), pct(pSp[k], 3), fmt(Om_hl[k, k] * 1e4, 2),
             pct(A_C[k], 0), fmt(om_closed[k] * 1e4, 2)] for k in range(3)]
    write_table('idz_views', tabular('clrrrrrr', ['', 'ビュー', '$p_k\\Pi$', '$Q_k$', "$p_k\\Sigma p_k'$",
                                                   '$\\omega_k^{\\mathrm{HL}}\\times10^4$', '$C_k$',
                                                   '$\\omega_k^{\\mathrm{Idz}}\\times10^4$'], rows))
    # 表：結果
    rows = [[A_NAMES[i], pct(Pi[i], 2), pct(mu_hl[i], 2), pct(mu_i[i], 2), pct(w[i], 2), pct(w_hl[i], 2),
             pct(w_i[i], 2), pct(w_100[i], 2)] for i in range(8)]
    rows.append(['合計', '', '', '', pct(w.sum(), 2), pct(w_hl.sum(), 2), pct(w_i.sum(), 2), pct(w_100.sum(), 2)])
    write_table('idz_result', tabular('lrrrrrrr', ['資産クラス', '$\\Pi$', '$\\mu$（HL）', '$\\mu$（Idz）',
                                                    '$w_{\\mathrm{mkt}}$', '$w$（HL）', '$w$（Idz）', '$w$（100\\%）'],
                                       rows, midrules=(8,)))
    # 表：確信度
    rows = []
    for k, (name, j) in enumerate([('ビュー 1（先進国株式）', 6), ('ビュー 2（外国債券）', 1), ('ビュー 3（大型グロース）', 2)]):
        rows.append([name, pct(single[k], 1), pct(implied[j], 2), pct(A_C[k], 0), pct(achieved[j], 1)])
    write_table('idz_conf', tabular('lrrrr', ['', '1 つだけ（HL）', '3 つ同時（HL）', '指定 $C_k$',
                                               '3 つ同時（Idz）'], rows))
    # 事前の指標（論文の表 8 と照合）
    act = w_hl - w
    put('idz.blsd', pct(np.sqrt(w_hl @ S @ w_hl), 3))
    put('idz.actrisk', pct(np.sqrt(act @ S @ act), 3))
    put('idz.blret', pct(w_hl @ mu_hl, 3))
    put('idz.mktret', pct(w @ mu_hl, 3))
    put('idz.actret', pct(act @ mu_hl, 3))
    put('idz.ir', fmt(act @ mu_hl / np.sqrt(act @ S @ act), 3))
    # 図：比率の傾き
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    bars(ax, ['米国\n債券', '外国\n債券', '大型\nグロース', '大型\nバリュー', '小型\nグロース', '小型\nバリュー', '先進国\n株式',
              '新興国\n株式'],
         [100 * (w_hl - w), 100 * (w_i - w), 100 * (w_100 - w)], [C1, C2, C3],
         ['He–Litterman の Ω（τpΣp′）', 'Idzorek の Ω（確信度 25/50/65%）', 'Ω = 0（100% の確信）'])
    ax.set_ylabel('時価総額の比率からの差（%）')
    ax.legend(loc='upper right', ncol=1)
    save_fig(fig, 'idz_tilts')
    SHARED['idz'] = dict(Pi=Pi, lam=lam, Om_hl=Om_hl, om_idz=om_closed, mu_hl=mu_hl, w_hl=w_hl, mu_i=mu_i, w_i=w_i,
                         pSp=pSp)


# ============================================================
# B. τ と Ω の関係
# ============================================================
def sec_b():
    d = SHARED['idz']
    Pi, lam, S, P, Q = d['Pi'], d['lam'], A_S, A_P, A_Q
    om_fixed = np.diag(d['Om_hl'])
    taus = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0]
    rows = []
    for t in taus:
        mu_a, M_a = bl(Pi, S, P, Q, np.diag(t * d['pSp']), t)
        mu_b, M_b = bl(Pi, S, P, Q, np.diag(om_fixed), t)
        adopt_b = (mu_b[6] - Pi[6]) / (Q[0] - Pi[6])
        rows.append([fmt(t, 3), pct(mu_a[6], 3), pct(np.sqrt(M_a[6, 6]), 2), pct(mu_b[6], 3), pct(adopt_b, 1),
                     pct(np.sqrt(M_b[6, 6]), 2)])
    write_table('tau', tabular('rrrrrr', ['$\\tau$', '$\\mu$（$\\Omega\\propto\\tau$）', '$\\sqrt{M}$（$\\Omega\\propto\\tau$）', '$\\mu$（$\\Omega$ 固定）',
                                          '$Q$ への移動', '$\\sqrt{M}$（$\\Omega$ 固定）'], rows))
    tt = np.logspace(-3, 0.3, 80)
    mu_fixed = [bl(Pi, S, P, Q, np.diag(om_fixed), t)[0][6] for t in tt]
    mu_prop = [bl(Pi, S, P, Q, np.diag(t * d['pSp']), t)[0][6] for t in tt]
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.plot(tt, 100 * np.array(mu_prop), color=C1, label='Ω を τ に比例させる（He–Litterman）')
    ax.plot(tt, 100 * np.array(mu_fixed), color=C2, label='Ω を固定する（τ = 0.025 のときの値）')
    ax.axhline(100 * Pi[6], color=NEUTRAL, lw=0.8, ls='--')
    ax.axhline(100 * Q[0], color=NEUTRAL, lw=0.8, ls='--')
    ax.text(tt[-1], 100 * Pi[6] + 0.01, f'均衡 Π = {100 * Pi[6]:.2f}%', fontsize=7, color=INK2, va='bottom',
            ha='right')
    ax.text(tt[0], 100 * Q[0] - 0.01, f'ビュー Q = {100 * Q[0]:.2f}%', fontsize=7, color=INK2, va='top')
    ax.set_xscale('log')
    ax.set_xlabel('τ（対数目盛）')
    ax.set_ylabel('先進国株式の事後の期待リターン（%）')
    ax.legend(loc='center right')
    save_fig(fig, 'tau')
    # 採用率の曲線（1 資産・1 ビュー）
    x = np.logspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.plot(x, 1 / (1 + x), color=C1)
    marks = [(1.0, 'He–Litterman（ω = τσ²）', (8, 4)), (1 / 0.10 ** 2 - 1, 'IC = 0.10 に合わせた ω', (-150, 26)),
             (1 / 0.05 ** 2 - 1, 'IC = 0.05 に合わせた ω', (8, 20))]
    if 'llm_ratio_med' in SHARED:
        marks.insert(0, (SHARED['llm_ratio_med'], 'LLM の回答の分散（中央値）', (8, 4)))
    for xv, lab, off in marks:
        yv = 1 / (1 + xv)
        ax.plot([xv], [yv], 'o', ms=5, color=C2, zorder=3)
        ax.annotate(f'{lab}\n採用率 {100 * yv:.1f}%', (xv, yv), xytext=off, textcoords='offset points', fontsize=7,
                    color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5) if off[1] > 10 else None)
    ax.set_xscale('log')
    ax.set_xlabel('ω ÷ τσ²（ビューの不確かさ ÷ 均衡の不確かさ、対数目盛）')
    ax.set_ylabel('ビューの採用率')
    ax.set_ylim(-0.03, 1.05)
    save_fig(fig, 'adoption_curve')


# ============================================================
# C. Meucci（2010）の 6 か国の例：元の BL と市場版
# ============================================================
M_NAMES = ['イタリア', 'スペイン', 'スイス', 'カナダ', 'アメリカ', 'ドイツ']
M_VOL = np.array([0.21, 0.24, 0.24, 0.25, 0.29, 0.31])
M_CORR_UPPER = [[1, .54, .62, .25, .41, .59], [1, .69, .29, .36, .83], [1, .15, .46, .65], [1, .47, .39], [1, .38],
                [1]]
M_W = np.array([0.04, 0.04, 0.05, 0.08, 0.71, 0.08])


def meucci_inputs():
    C = np.zeros((6, 6))
    for i, row in enumerate(M_CORR_UPPER):
        for k, v in enumerate(row):
            C[i, i + k] = C[i + k, i] = v
    S = np.outer(M_VOL, M_VOL) * C
    return C, S


def sec_c():
    C, S = meucci_inputs()
    lam_m = 1.2
    pi = 2 * lam_m * S @ M_W
    tau = 0.4
    P = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1]], float)
    v = np.array([0.12, -0.10])
    Om = P @ S @ P.T
    mu_bl, M = bl(pi, S, P, v, Om, tau)
    S_bl = S + M
    mu_m, S_m = bl_market(pi, S, P, v, Om)
    mu0_bl, M0 = bl(pi, S, P, v, 1e-12 * np.eye(2), tau)
    mu0_m, S0_m = bl_market(pi, S, P, v, 1e-12 * np.eye(2))
    put('meu.pi', ', '.join(pct(x, 1) for x in pi))
    put('meu.ppi1', pct(P[0] @ pi, 2))
    put('meu.ppi2', pct(P[1] @ pi, 2))
    put('meu.vsd1', pct(np.sqrt(Om[0, 0]), 1))
    put('meu.vsd2', pct(np.sqrt(Om[1, 1]), 1))
    w_prior = mv_long_only(pi, S, 2 * lam_m)
    w_bl = mv_long_only(mu_bl, S_bl, 2 * lam_m)
    w_m = mv_long_only(mu_m, S_m, 2 * lam_m)
    rows = []
    for i in range(6):
        rows.append([M_NAMES[i], pct(M_W[i], 0), pct(pi[i], 2), pct(mu_bl[i], 2), pct(mu_m[i], 2),
                     pct(np.sqrt(S[i, i]), 1), pct(np.sqrt(S_bl[i, i]), 1), pct(np.sqrt(S_m[i, i]), 1),
                     pct(w_prior[i], 1), pct(w_bl[i], 1), pct(w_m[i], 1)])
    write_table('meucci', tabular('lrrrrrrrrrr', ['国', '$w$', '$\\pi$', '$\\mu_{\\mathrm{BL}}$', '$\\mu_{\\mathrm{mBL}}$',
                                                  '$\\sigma$', '$\\sigma_{\\mathrm{BL}}$', '$\\sigma_{\\mathrm{mBL}}$',
                                                  '$w^*_{\\pi}$', '$w^*_{\\mathrm{BL}}$', '$w^*_{\\mathrm{mBL}}$'], rows))
    put('meu.blspain', pct(mu_bl[1], 2))
    put('meu.mspain', pct(mu_m[1], 2))
    put('meu.full.bl', pct(mu0_bl[1], 2))
    put('meu.full.m', pct(mu0_m[1], 2))
    put('meu.full.blsd', pct(np.sqrt((S + M0)[1, 1]), 1))
    put('meu.full.msd', pct(np.sqrt(max(S0_m[1, 1], 0)), 2))
    put('meu.unc.blsd', pct(np.sqrt((1 + tau) * S[1, 1]), 1))
    # 定性的なビュー（η）
    rows = []
    for k, name in enumerate(['スペイン', 'アメリカ $-$ ドイツ']):
        sd = np.sqrt(Om[k, k])
        base = P[k] @ pi
        rows.append([name, pct(base, 2), pct(sd, 1)] + [pct(base + eta * sd, 1) for eta in (-2, -1, 1, 2)])
    write_table('meucci_eta', tabular('lrrrrrr', ['ビュー', '$(P\\pi)_k$', "$\\sqrt{(P\\Sigma P')_{kk}}$",
                                                  'とても弱気', '弱気', '強気', 'とても強気'], rows))


# ============================================================
# D. IC と Ω（Grinold の式との対応）
# ============================================================
def sec_d():
    rows = []
    for ic in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        s = ic ** 2
        rows.append([fmt(ic, 2), pct(s, 2), fmt(1 / s - 1, 0), fmt(1 / ic, 1), pct(ic * 0.08, 2)])
    write_table('ic', tabular('rrrrr', ['IC', '採用率 $=\\mathrm{IC}^2$', '$\\omega/\\sigma^2=1/\\mathrm{IC}^2-1$',
                                        '生の予想の広がり $\\sigma_v/\\sigma$', '$z=1$ の $\\alpha$（$\\sigma=8\\%$/月）'],
                                rows))
    # シミュレーション：ノイズの多い予想と、その較正
    rng = np.random.default_rng(101)   # 節ごとに種を決める（実行する節の組み合わせで結果が変わらないように）
    n, sigma, ic = 400_000, 0.08, 0.10
    x = rng.normal(0, sigma, n)
    om = sigma ** 2 * (1 / ic ** 2 - 1)
    v = x + rng.normal(0, np.sqrt(om), n)
    z = (v - v.mean()) / v.std()
    post = ic * sigma * z
    put('sim.n', f'{n:,}'.replace(',', '{,}'))
    put('sim.ic', fmt(np.corrcoef(v, x)[0, 1], 3))
    put('sim.slope.raw', fmt(np.polyfit(v, x, 1)[0], 4))
    put('sim.slope.post', fmt(np.polyfit(post, x, 1)[0], 3))
    q = pd.qcut(v, 20, labels=False)
    df = pd.DataFrame({'v': v, 'post': post, 'x': x, 'q': q}).groupby('q').mean()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axs[0]
    ax.plot(100 * df.v, 100 * df.x, 'o', ms=4, color=C1)
    lim = 100 * np.array([df.v.min(), df.v.max()])
    ax.plot(lim, lim, color=NEUTRAL, lw=0.8, ls='--')
    ax.set_ylim(-3, 3)
    ax.set_xlabel('生の予想 v（%、20 分位の平均）')
    ax.set_ylabel('実際の値の平均（%）')
    ax.set_title(f'生の予想：傾き {np.polyfit(v, x, 1)[0]:.3f}（= IC²）')
    ax = axs[1]
    ax.plot(100 * df.post, 100 * df.x, 'o', ms=4, color=C2)
    lim = 100 * np.array([df.post.min(), df.post.max()])
    ax.plot(lim, lim, color=NEUTRAL, lw=0.8, ls='--')
    ax.set_xlabel('IC × σ × z（%、20 分位の平均）')
    ax.set_title(f'Grinold の式で縮めた値：傾き {np.polyfit(post, x, 1)[0]:.2f}')
    save_fig(fig, 'ic_sim')


# ============================================================
# E. 本プロジェクトの LLM の回答：当たり具合と Ω
# ============================================================
def load_llm_panel(conn):
    views = pd.read_sql("SELECT as_of_date, symbol, mean_return, variance, n_answers FROM llm_view_stats "
                        "WHERE run_id = 1", conn)
    syms = sorted(views.symbol.unique())
    px = load_prices(conn, syms, '2023-01-01', '2026-09-30')
    ret = px / px.shift(1) - 1
    rows = []
    for d in sorted(views.as_of_date.unique()):
        t = pd.Timestamp(d)
        # 判断日の翌月の末日（判断日は月の最後の取引日）
        nxt_end = t.replace(day=1) + pd.offsets.MonthEnd(2)
        cur_start = t.replace(day=1)
        win_next = ret[(ret.index > t) & (ret.index <= nxt_end)]
        win_cur = ret[(ret.index >= cur_start) & (ret.index <= t)]
        hist = ret[ret.index <= t].tail(252)
        v = views[views.as_of_date == d].set_index('symbol')
        df = pd.DataFrame({
            'date': d, 'Q': v.mean_return, 'om': v.variance,
            'r_next': win_next.mean().reindex(v.index), 'n_next': win_next.notna().sum().reindex(v.index),
            'r_in': win_cur.mean().reindex(v.index), 'sd': hist.std().reindex(v.index),
            'R_next': (1 + win_next.fillna(0)).prod().reindex(v.index) - 1,
        })
        rows.append(df)
    panel = pd.concat(rows).reset_index().rename(columns={'index': 'symbol'})
    panel = panel[(panel.n_next >= 15) & panel.sd.notna() & panel.r_in.notna()]
    return panel


def sec_e(conn):
    panel = load_llm_panel(conn)
    SHARED['llm_panel'] = panel
    months = sorted(panel.date.unique())
    put('llm.months', len(months))
    put('llm.rows', f'{len(panel):,}'.replace(',', '{,}'))
    per = []
    for d in months:
        g = panel[panel.date == d]
        b = np.polyfit(g.Q, g.r_next, 1)[0]
        per.append(dict(date=d, n=len(g), ic=stats.spearmanr(g.Q, g.r_next)[0], icp=np.corrcoef(g.Q, g.r_next)[0, 1],
                        cin=np.corrcoef(g.Q, g.r_in)[0, 1], slope=b,
                        ratio=np.median(g.om / (g.Q - g.r_next) ** 2),
                        mse=np.mean((g.Q - g.r_next) ** 2), noise=np.mean(g.sd ** 2 / g.n_next),
                        om=np.mean(g.om), sdQ=g.Q.std(), sdr=g.r_next.std()))
    per = pd.DataFrame(per)
    SHARED['llm_per'] = per
    T = len(per)
    put('llm.ic.mean', sfmt(per.ic.mean(), 3))
    put('llm.ic.t', fmt(per.ic.mean() / per.ic.std() * np.sqrt(T), 2))
    put('llm.ic.pos', int((per.ic > 0).sum()))
    put('llm.icp.mean', sfmt(per.icp.mean(), 3))
    put('llm.cin.mean', fmt(per.cin.mean(), 2))
    put('llm.slope.mean', sfmt(per.slope.mean(), 3))
    put('llm.slope.t', fmt(per.slope.mean() / per.slope.std() * np.sqrt(T), 2))
    put('llm.slope.med', sfmt(per.slope.median(), 3))
    # 月ごとに平均を引いて、まとめて傾きを取る
    dm = panel.copy()
    dm['Qd'] = dm.Q - dm.groupby('date').Q.transform('mean')
    dm['rd'] = dm.r_next - dm.groupby('date').r_next.transform('mean')
    pooled = float((dm.Qd * dm.rd).sum() / (dm.Qd ** 2).sum())
    put('llm.slope.pooled', sfmt(pooled, 3))
    ratio_all = panel.om / (panel.Q - panel.r_next) ** 2
    put('llm.ratio.med', fmt(ratio_all.median(), 3))
    put('llm.ratio.q25', fmt(ratio_all.quantile(0.25), 3))
    put('llm.ratio.q75', fmt(ratio_all.quantile(0.75), 3))
    put('llm.om.mean', sci(panel.om.mean()))
    put('llm.mse', sci(per.mse.mean()))
    put('llm.noise', sci(per.noise.mean()))
    put('llm.sdQ', pct(per.sdQ.mean(), 3))
    put('llm.sdr', pct(per.sdr.mean(), 3))
    put('llm.omsd', pct(np.sqrt(panel.om).median(), 3))
    put('llm.sd.med', pct(panel.sd.median(), 2))
    # 回答の分散と、τσ² の比（採用率の目安）
    tau = 0.025
    r = panel.om / (tau * panel.sd ** 2)
    SHARED['llm_ratio_med'] = float(r.median())
    put('llm.omtau.med', sci(r.median()))
    put('llm.adopt.med', pct(1 / (1 + r.median()), 2))
    # 表：月ごと
    rows = [[d[:7], f'{int(n)}', sfmt(ic, 3), sfmt(icp, 3), sfmt(s, 3), fmt(c, 2), fmt(ra, 3)]
            for d, n, ic, icp, s, c, ra in per[['date', 'n', 'ic', 'icp', 'slope', 'cin', 'ratio']].itertuples(index=False)]
    write_table('llm_monthly', tabular('lrrrrrr', ['判断日', '銘柄数', '順位相関', '相関', '傾き $b$',
                                                   '入力との相関', '分散 ÷ 誤差$^2$'], rows))
    # 図：分散 ÷ 誤差^2 の分布
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    lr = np.log10(ratio_all[ratio_all > 0])
    ax.hist(np.clip(lr, -5, 3), bins=np.arange(-5, 3.01, 0.1), color=C1, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color=INK2, lw=0.8, ls='--')
    ax.axvline(np.median(lr), color=C2, lw=1.2)
    ax.text(0.05, ax.get_ylim()[1] * 0.92, '1（分散 = 誤差²）', fontsize=7, color=INK2)
    ax.annotate(f'中央値 {ratio_all.median():.3f}', (np.median(lr), ax.get_ylim()[1] * 0.62),
                xytext=(-3.6, ax.get_ylim()[1] * 0.62), textcoords='data', fontsize=7, color=INK2, va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.set_xlabel('log10（回答の分散 ÷ 実際の誤差の 2 乗）')
    ax.set_ylabel('銘柄・月の数')
    save_fig(fig, 'llm_ratio')
    # 図：月ごとの順位相関と、10 分位の散布
    dm['dec'] = dm.groupby('date').Qd.transform(lambda s: pd.qcut(s, 10, labels=False))
    dec = dm.groupby('dec')[['Qd', 'rd']].mean()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axs[0]
    ax.bar(range(T), per.ic, color=[C1 if x >= 0 else C2 for x in per.ic], width=0.72)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(0, T, 6), [d[:7] for d in per.date[::6]])
    ax.set_ylabel('順位相関')
    ax.set_title(f'予想と翌月の実際の順位相関（平均 {per.ic.mean():+.3f}）')
    ax.grid(axis='x', visible=False)
    ax = axs[1]
    ax.plot(100 * dec.Qd, 100 * dec.rd, 'o-', ms=4, color=C1, lw=1)
    lim = 100 * np.array([dec.Qd.min(), dec.Qd.max()])
    ax.plot(lim, lim, color=NEUTRAL, lw=0.8, ls='--')
    ax.set_xlabel('予想（月の平均からの差、%/日）')
    ax.set_ylabel('翌月の実際（同、%/日）')
    ax.set_title(f'10 分位の平均（傾き {pooled:+.3f}）')
    save_fig(fig, 'llm_ic')


TE_TARGET = 0.03   # 年率のトラッキングエラーの上限（E2 の 1 つの戦略）


def optimize_te(mu, S, delta, wb, te_daily):
    """blm.optimizer.optimize と同じ目的関数に、指数からのずれ（事前の TE）の上限を足したもの。"""
    n = len(mu)
    w = cp.Variable(n)
    Sv = cp.psd_wrap(S.values)
    obj = mu.values @ w - delta / 2 * cp.quad_form(w, Sv)
    cons = [cp.sum(w) == 1, w >= 0, cp.quad_form(w - wb.values, Sv) <= te_daily ** 2]
    prob = cp.Problem(cp.Maximize(1e4 * obj), cons)
    prob.solve(solver=cp.CLARABEL)
    x = np.clip(w.value, 0, None)
    return pd.Series(x / x.sum(), index=mu.index)


def sec_e2(conn):
    """Ω の決め方を 3 通り変えて、30 か月の比率を計算し直す（本プロジェクトのコードを使う）。"""
    from blm import config as bconfig
    from blm.black_litterman import posterior_returns
    from blm.covariance import shrunk_covariance
    from blm.equilibrium import implied_returns
    from blm.inputs import load_inputs
    from blm.optimizer import optimize as blm_optimize
    per = SHARED['llm_per']
    panel = SHARED['llm_panel']
    tau, delta = bconfig.TAU, bconfig.RISK_AVERSION
    put('blm.tau', fmt(tau, 3))
    put('blm.delta', fmt(delta, 1))
    results, detail = [], []
    prev = {}
    dates = list(per.date)
    for i, d in enumerate(dates):
        inp = load_inputs(conn, d, 1, bconfig.COV_WINDOW_DAYS, bconfig.MIN_HISTORY_RATIO)
        S = shrunk_covariance(inp.returns)
        Pi = implied_returns(S, inp.market_weights, delta)
        Q = inp.view_mean - inp.risk_free_daily
        var_i = pd.Series(np.diag(S.values), index=S.index)
        # 過去の月だけで測った傾き（月ごとに平均を引いてまとめる）。データがないうちは 0。
        # 1 つ前の判断日の「翌月」は、この判断日で終わる月なので、結果はもう出ている
        past = panel[panel.date.isin(dates[:i])]
        if len(past):
            qd = past.Q - past.groupby('date').Q.transform('mean')
            rd = past.r_next - past.groupby('date').r_next.transform('mean')
            b = float(np.clip((qd * rd).sum() / (qd ** 2).sum(), 0, 1))
        else:
            b = 0.0
        R = panel[panel.date == d].set_index('symbol').R_next.reindex(S.index)
        # Ω を Σ に比例させると（誤差が銘柄間で Σ と同じように相関する）、事後の平均は Π + b (Q - Π)
        mu_full = Pi + b * (Q - Pi)
        # 相対ビュー：予想と均衡から、それぞれの時価総額加重の平均を引いた差だけを使う（全体の水準は動かさない）
        wm = inp.market_weights
        gap_rel = (Q - float(wm @ Q)) - (Pi - float(wm @ Pi))
        mu_rel = Pi + b * gap_rel
        variants = {
            'sample': posterior_returns(Pi, S, Q, inp.view_variance, tau),
            'hl': posterior_returns(Pi, S, Q, tau * var_i, tau),
            'calib': posterior_returns(Pi, S, Q, tau * var_i * (1 / b - 1), tau) if b > 1e-9 else Pi.copy(),
            'full': mu_full,
            'fullte': mu_full,
            'relte': mu_rel,
        }
        for name, mu in variants.items():
            if name in ('fullte', 'relte'):
                w = optimize_te(mu, S, delta, inp.market_weights, TE_TARGET / np.sqrt(252))
            else:
                w = blm_optimize(mu, S, delta)
            adopt = ((mu - Pi) / (Q - Pi)).where((Q - Pi).abs() > 1e-5).median()
            ret = float((w * R.fillna(0)).sum())
            if name in prev:
                idx = w.index.union(prev[name].index)
                to = 0.5 * float((w.reindex(idx).fillna(0) - prev[name].reindex(idx).fillna(0)).abs().sum())
            else:
                to = np.nan
            prev[name] = w
            results.append(dict(date=d, var=name, ret=ret, adopt=adopt, effn=eff_n(w),
                                active=0.5 * float((w - inp.market_weights).abs().sum()), turnover=to, b=b))
        mret = float((inp.market_weights * R.fillna(0)).sum())
        results.append(dict(date=d, var='mkt', ret=mret, adopt=0, effn=eff_n(inp.market_weights), active=0,
                            turnover=np.nan, b=b))
        rf = inp.risk_free_daily * 21
        detail.append(dict(date=d, rf=rf, b=b))
    res = pd.DataFrame(results)
    det = pd.DataFrame(detail).set_index('date')
    SHARED['blm_res'] = res
    mk = res[res['var'] == 'mkt'].set_index('date').ret
    labels = {'sample': '回答の分散（元の方法）', 'hl': 'He–Litterman（$\\tau\\,\\mathrm{diag}\\,\\Sigma$）',
              'calib': '傾きで較正（対角）', 'full': '傾きで較正（$\\Sigma$ に比例）',
              'fullte': '同上 ＋ TE 3\\% 以内', 'relte': '相対ビュー ＋ TE 3\\% 以内', 'mkt': '時価総額加重'}
    plain = {'sample': '回答の分散（元の方法）', 'calib': '傾きで較正（対角）', 'full': '傾きで較正（Σ に比例）',
             'fullte': '同上 ＋ TE 3% 以内', 'relte': '相対ビュー ＋ TE 3% 以内'}
    put('bt.te.target', pct(TE_TARGET, 0))
    rows = []
    for name in ['sample', 'hl', 'calib', 'full', 'fullte', 'relte', 'mkt']:
        g = res[res['var'] == name].set_index('date')
        r = g.ret
        ex = r - det.rf
        act = r - mk
        T = len(r)
        ann = (1 + r).prod() ** (12 / T) - 1
        te = act.std() * np.sqrt(12)
        rows.append([labels[name], pct(g.adopt.median(), 1) if name != 'mkt' else '---', fmt(g.effn.mean(), 1),
                     pct(g.active.mean(), 0) if name != 'mkt' else '---', pct(ann, 1), pct(r.std() * np.sqrt(12), 1),
                     fmt(ex.mean() / ex.std() * np.sqrt(12), 2),
                     pct(te, 1) if name != 'mkt' else '---',
                     fmt(act.mean() * 12 / te, 2) if name != 'mkt' else '---',
                     pct(g.turnover.mean(), 0) if name != 'mkt' else '---'])
        put(f'bt.{name}.ann', pct(ann, 1))
        put(f'bt.{name}.effn', fmt(g.effn.mean(), 1))
        put(f'bt.{name}.adopt', pct(g.adopt.median(), 1))
        if name != 'mkt':
            put(f'bt.{name}.te', pct(te, 1))
            put(f'bt.{name}.ir', fmt(act.mean() * 12 / te, 2))
            put(f'bt.{name}.tstat', fmt(act.mean() / act.std() * np.sqrt(T), 2))
    write_table('blm_omega', tabular('lrrrrrrrrr', ['$\\Omega$ の決め方', '採用率', '実質銘柄数', '指数との差',
                                                    '年率', 'リスク', 'シャープ', 'TE', 'IR', '回転率'], rows))
    bcal = det.b
    put('bt.b.last', fmt(bcal.iloc[-1], 3))
    put('bt.b.zero', int((bcal <= 1e-9).sum()))
    put('bt.months', len(det))
    # 図：時価総額加重に対する累積の差
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = pd.to_datetime(list(det.index))
    ends = []
    for name, col in [('sample', C1), ('full', C2), ('fullte', C3)]:
        g = res[res['var'] == name].set_index('date')
        cum = np.cumprod(1 + g.ret.values) / np.cumprod(1 + mk.values)
        ax.plot(x, cum, color=col, label=plain[name])
        ends.append((cum[-1], f'{cum[-1]:.2f}'))
    ax.axhline(1, color=INK2, lw=0.6)
    end_labels(ax, x[-1] + pd.Timedelta(days=10), ends)
    ax.set_ylabel('時価総額加重に対する比（開始時 = 1）')
    ax.legend(loc='lower left')
    save_fig(fig, 'blm_omega')
    # 図：較正の傾き b_t と、実質銘柄数の推移
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    axs[0].step(x, det.b.values, where='post', color=C1)
    axs[0].set_ylabel('判断日までの傾き $b_t$')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('較正の傾き（それまでの月で計算）')
    for name, col in [('sample', C1), ('calib', C2), ('full', C3), ('fullte', C4)]:
        g = res[res['var'] == name].set_index('date')
        axs[1].plot(x, g.effn.values, color=col, label=plain[name])
    mkn = res[res['var'] == 'mkt'].set_index('date').effn
    axs[1].plot(x, mkn.values, color=NEUTRAL, lw=1.0, ls='--', label='時価総額加重')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('実質銘柄数（対数目盛）')
    axs[1].set_title('実質銘柄数の推移')
    axs[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
    for ax in axs:
        ax.tick_params(axis='x', labelrotation=0)
        ax.set_xticks(pd.to_datetime(['2024-07-01', '2025-07-01', '2026-07-01']), ['2024-07', '2025-07', '2026-07'])
    save_fig(fig, 'blm_omega_time')


# ============================================================
# セクター ETF・指数の月次データ（F〜K で共通）
# ============================================================
ETF_JA = {'XLC': '通信サービス', 'XLY': '一般消費財', 'XLP': '生活必需品', 'XLE': 'エネルギー', 'XLF': '金融',
          'XLV': 'ヘルスケア', 'XLI': '資本財', 'XLK': '情報技術', 'XLB': '素材', 'XLRE': '不動産', 'XLU': '公益'}
ETF_ORDER = ['XLK', 'XLF', 'XLV', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLU', 'XLRE', 'XLB']
OLD9 = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLU', 'XLB']


def load_daily(conn):
    if 'daily' not in SHARED:
        etfs = load_sector_etfs(conn)
        px = load_prices(conn, list(etfs) + ['^GSPC'], '2014-01-01', '2026-09-30')
        irx = load_prices(conn, ['^IRX'], '2014-01-01', '2026-09-30', column='close')['^IRX']
        vix = load_prices(conn, ['^VIX'], '2014-01-01', '2026-09-30', column='close')['^VIX']
        SHARED['daily'] = (px, irx, vix, etfs)
    return SHARED['daily']


def monthly(conn):
    """月末の値から作る月次のリターン（期間の添字）と、月次の無リスク金利。"""
    if 'monthly' not in SHARED:
        px, irx, vix, etfs = load_daily(conn)
        me = px.resample('ME').last()
        r = me / me.shift(1) - 1
        r.index = r.index.to_period('M')
        rf = (irx.resample('ME').last() / 100 / 12)
        rf.index = rf.index.to_period('M')
        rf = rf.shift(1)   # 月初の金利でその月の無リスク金利とする
        SHARED['monthly'] = (r, rf)
    return SHARED['monthly']


def read_ff():
    ff = pd.read_csv(DATA / 'ff_factors_monthly.csv')
    ff.index = pd.PeriodIndex([f'{str(m)[:4]}-{str(m)[4:]}' for m in ff.month], freq='M')
    return ff.drop(columns='month') / 100


# ============================================================
# F. 政治・政策の不確かさ（EPU、GPR）
# ============================================================
def sec_f(conn):
    px, irx, vix, etfs = load_daily(conn)
    r, rf = monthly(conn)
    epu = pd.read_csv(DATA / 'epu_monthly.csv')
    epu = pd.Series(epu.epu.values, index=pd.PeriodIndex(epu.month, freq='M'))
    gpr = pd.read_csv(DATA / 'gpr_monthly.csv')
    gpr.index = pd.PeriodIndex(gpr.month, freq='M')
    d = px['^GSPC'].pct_change(fill_method=None)
    rv = d.groupby(d.index.to_period('M')).std() * np.sqrt(252)
    vixm = vix.resample('ME').last()
    vixm.index = vixm.index.to_period('M')
    idx = pd.period_range('2015-01', '2026-09', freq='M')
    df = pd.DataFrame({'epu': epu.reindex(idx), 'gpr': gpr.GPR.reindex(idx), 'gprt': gpr.GPRT.reindex(idx),
                       'gpra': gpr.GPRA.reindex(idx), 'rv': rv.reindex(idx), 'vix': vixm.reindex(idx)})
    df = df.dropna()
    SHARED['unc'] = df
    put('unc.start', str(df.index[0]))
    put('unc.end', str(df.index[-1]))
    put('unc.epu.med', fmt(df.epu.median(), 0))
    put('unc.epu.max', fmt(df.epu.max(), 0))
    put('unc.epu.argmax', str(df.epu.idxmax()))
    put('unc.gpr.med', fmt(df.gpr.median(), 0))
    put('unc.gpr.max', fmt(df.gpr.max(), 0))
    put('unc.gpr.argmax', str(df.gpr.idxmax()))
    lg = np.log(df)
    nxt = lg.rv.shift(-1)
    ok = nxt.notna()
    cors = {k: np.corrcoef(lg[k][ok], nxt[ok])[0, 1] for k in ['epu', 'gpr', 'vix', 'rv']}
    for k, v in cors.items():
        put(f'unc.cor.{k}', fmt(v, 2))
    X = np.column_stack([np.ones(ok.sum()), lg.rv[ok], lg.epu[ok], lg.gpr[ok]])
    b, se, _ = ols(nxt[ok], X)
    rows = [[n, fmt(bi, 3), fmt(si, 3), fmt(bi / si, 2)] for n, bi, si in
            zip(['定数', '$\\log RV_t$', '$\\log \\mathrm{EPU}_t$', '$\\log \\mathrm{GPR}_t$'], b, se)]
    write_table('unc_reg', tabular('lrrr', ['説明変数', '係数', '標準誤差（White）', '$t$ 値'], rows))
    put('unc.reg.n', int(ok.sum()))
    put('unc.reg.epu.t', fmt(b[2] / se[2], 2))
    put('unc.reg.gpr.t', fmt(b[3] / se[3], 2))
    put('unc.reg.rv', fmt(b[1], 2))
    X2 = np.column_stack([np.ones(ok.sum()), lg.rv[ok], lg.vix[ok], lg.epu[ok]])
    b2, se2, _ = ols(nxt[ok], X2)
    put('unc.reg2.epu.t', fmt(b2[3] / se2[3], 2))
    put('unc.reg2.vix.t', fmt(b2[2] / se2[2], 2))
    # 図：EPU と GPR
    x = df.index.to_timestamp()
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True)
    axs[0].plot(x, df.epu, color=C1, lw=1.3)
    axs[0].set_ylabel('EPU（新聞ベース）')
    axs[0].annotate(f'{df.epu.idxmax()}', (df.epu.idxmax().to_timestamp(), df.epu.max()), xytext=(6, -4),
                    textcoords='offset points', fontsize=7, color=INK2)
    axs[1].plot(x, df.gprt, color=C1, lw=1.1, label='脅威（GPRT）')
    axs[1].plot(x, df.gpra, color=C2, lw=1.1, label='行動（GPRA）')
    axs[1].set_ylabel('GPR の内訳')
    axs[1].legend(loc='upper left', ncol=2)
    date_axis(axs[1])
    save_fig(fig, 'unc_series')
    # セクターの感応度（市場を入れたうえでの EPU・GPR の変化への反応）
    ex = r.sub(rf, axis=0)
    dle = np.log(epu).diff()
    dlg = np.log(gpr.GPR).diff()
    rows, betas = [], {}
    for s in ETF_ORDER:
        y = ex[s].dropna()
        y = y[(y.index >= pd.Period('2015-02', 'M')) & (y.index <= pd.Period('2026-09', 'M'))]
        m = ex['^GSPC'].reindex(y.index)
        Xs = np.column_stack([np.ones(len(y)), m, dle.reindex(y.index), dlg.reindex(y.index)])
        okk = ~np.isnan(Xs).any(axis=1)
        bb, ss, _ = ols(y.values[okk], Xs[okk])
        betas[s] = bb
        rows.append([ETF_JA[s], f'{okk.sum()}', fmt(bb[1], 2), sfmt(bb[2] * 10, 2), fmt(bb[2] / ss[2], 1),
                     sfmt(bb[3] * 10, 2), fmt(bb[3] / ss[3], 1)])
    write_table('unc_sector', tabular('lrrrrrr', ['セクター', '月数', '$\\beta_{\\mathrm{mkt}}$',
                                                  'EPU +10\\%（\\%）', '$t$', 'GPR +10\\%（\\%）', '$t$'], rows))
    # 外部の変数（GPR の変化）へのビュー：市場版 BL で条件付ける
    cols = [s for s in ETF_ORDER]
    Xm = ex[cols].copy()
    Xm['dgpr'] = dlg
    Xm['depu'] = dle
    Xm = Xm[(Xm.index >= pd.Period('2018-07', 'M')) & (Xm.index <= pd.Period('2026-09', 'M'))].dropna()
    put('scen.n', len(Xm))
    S = np.cov(Xm.values.T)
    n = len(cols)
    pi = np.zeros(n + 2)
    rows = []
    shifts = {}
    for k, (name, val) in enumerate([('dgpr', np.log(1.5)), ('depu', np.log(1.5))]):
        P = np.zeros((1, n + 2))
        P[0, n + k] = 1
        sv = S[n + k, n + k]
        mu0, _ = bl_market(pi, S, P, np.array([val]), np.array([[1e-12]]))
        mu1, _ = bl_market(pi, S, P, np.array([val]), np.array([[sv]]))
        shifts[name] = (mu0[:n], mu1[:n])
        put(f'scen.{name}.sd', pct(np.sqrt(sv), 1))
    for i, s in enumerate(cols):
        c_g = S[i, n] / np.sqrt(S[i, i] * S[n, n])
        rows.append([ETF_JA[s], fmt(c_g, 2), spct(shifts['dgpr'][0][i], 2), spct(shifts['dgpr'][1][i], 2),
                     spct(shifts['depu'][0][i], 2)])
    write_table('unc_scen', tabular('lrrrr', ['セクター', 'GPR の変化との相関', 'GPR $+50\\%$（確実）',
                                              'GPR $+50\\%$（$c=1$）', 'EPU $+50\\%$（確実）'], rows))
    SHARED['epu_med'] = float(df.epu.median())


# ============================================================
# G. EGARCH-M（マクロのボラティリティからビューと Ω を作る例）
# ============================================================
def egarch_m(theta, y):
    c, g, om, a, xi, b = theta
    n = len(y)
    ls = np.empty(n + 1)
    ls[0] = np.log(np.var(y))
    ez = np.sqrt(2 / np.pi)
    ll = 0.0
    for t in range(n):
        s2 = np.exp(ls[t])
        e = y[t] - (c + g * np.sqrt(s2))
        z = e / np.sqrt(s2)
        ll -= 0.5 * (1.8378770664093453 + ls[t] + z * z)
        ls[t + 1] = min(max(om + b * ls[t] + a * (abs(z) - ez) + xi * z, -30.0), 30.0)
    return ll, ls


def sec_g(conn):
    px, irx, vix, etfs = load_daily(conn)
    d = px['^GSPC'].pct_change(fill_method=None)
    rf = (irx / 100 / 252).shift(1).reindex(d.index).ffill()
    y = (100 * (d - rf)).dropna()
    y = y[y.index >= '2015-01-05']
    yv = y.values
    x0 = np.array([0.05, 0.0, 0.0, 0.12, -0.12, 0.97])
    bounds = [(-1, 1), (-2, 2), (-2, 2), (0, 1), (-1, 1), (0.5, 0.9999)]
    res = optimize.minimize(lambda th: -egarch_m(th, yv)[0], x0, method='L-BFGS-B', bounds=bounds)
    th = res.x
    ll, ls = egarch_m(th, yv)
    # 数値の 2 階微分で標準誤差
    k = len(th)
    H = np.zeros((k, k))
    h = 1e-4 * np.maximum(np.abs(th), 1e-2)
    f0 = -ll
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            fpp = -egarch_m(th + ei + ej, yv)[0]
            fpm = -egarch_m(th + ei - ej, yv)[0]
            fmp = -egarch_m(th - ei + ej, yv)[0]
            fmm = -egarch_m(th - ei - ej, yv)[0]
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    names = ['$c$（平均の定数）', '$\\gamma$（$\\sigma_t$ の係数）', '$\\omega$', '$\\alpha$（大きさ）',
             '$\\xi$（向き）', '$\\beta$（持続性）']
    rows = [[nm, fmt(v, 4), fmt(s, 4), fmt(v / s, 2)] for nm, v, s in zip(names, th, se)]
    write_table('egarch', tabular('lrrr', ['パラメータ', '推定値', '標準誤差', '$t$ 値'], rows))
    put('eg.n', f'{len(yv):,}'.replace(',', '{,}'))
    put('eg.ll', fmt(ll, 1))
    put('eg.gamma', fmt(th[1], 3))
    put('eg.gamma.t', fmt(th[1] / se[1], 2))
    put('eg.xi', fmt(th[4], 3))
    put('eg.beta', fmt(th[5], 4))
    put('eg.start', str(y.index[0].date()))
    put('eg.end', str(y.index[-1].date()))
    sig = np.sqrt(np.exp(ls[1:]))   # 翌日の条件付き標準偏差（その日の終わりに分かる）
    sig = pd.Series(sig, index=y.index)
    ann = sig * np.sqrt(252)
    rv252 = (y.rolling(252).std() * np.sqrt(252))
    put('eg.max', fmt(ann.max(), 0))
    put('eg.argmax', str(ann.idxmax().date()))
    mu_next = th[0] + th[1] * sig
    put('eg.mu.min', fmt(21 * mu_next.min(), 2))
    put('eg.mu.max', fmt(21 * mu_next.max(), 2))
    put('eg.mu.med', fmt(21 * mu_next.median(), 2))
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.plot(ann.index, ann, color=C1, lw=0.8, label='EGARCH-M の翌日の条件付きボラ（年率）')
    ax.plot(rv252.index, rv252, color=C2, lw=1.4, label='過去 252 日の標準偏差（年率）')
    ax.set_ylabel('年率の標準偏差（%）')
    ax.legend(loc='upper left')
    date_axis(ax)
    save_fig(fig, 'egarch_vol')
    # 採用率の時系列：Ω を条件付き分散、τΣ を 252 日の分散に比例させる。EPU で Ω を伸縮する場合と比べる
    me = y.groupby(y.index.to_period('M')).apply(lambda s: s.index[-1])
    s_next = sig.reindex(me.values) ** 2
    v252 = (y.rolling(252).var()).reindex(me.values)
    adopt_g = pd.Series(v252.values / (v252.values + s_next.values), index=me.index)
    unc = SHARED.get('unc')
    adopt_e = 1 / (1 + unc.epu / SHARED['epu_med']) if unc is not None else None
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ag = adopt_g.dropna()
    ax.plot(ag.index.to_timestamp(), ag, color=C1, label='Ω ∝ EGARCH-M の分散、τΣ ∝ 252 日の分散')
    if adopt_e is not None:
        ax.plot(adopt_e.index.to_timestamp(), adopt_e, color=C2, label='Ω ∝ EPU（中央値で 50%）')
    ax.axhline(0.5, color=NEUTRAL, lw=0.8, ls='--')
    ax.set_ylabel('ビューの採用率')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    date_axis(ax)
    save_fig(fig, 'adoption_time')
    put('eg.adopt.min', pct(ag.min(), 0))
    put('eg.adopt.argmin', str(ag.idxmin()))
    put('eg.adopt.max', pct(ag.max(), 0))
    put('eg.adopt.argmax', str(ag.idxmax()))
    if adopt_e is not None:
        put('epu.adopt.min', pct(adopt_e.min(), 0))
        put('epu.adopt.argmin', str(adopt_e.idxmin()))


# ============================================================
# H. モメンタム：長い歴史（急落とボラ調整）と、セクターのビュー
# ============================================================
def mom_stats(r):
    r = r.dropna()
    w = (1 + r).cumprod()
    dd = (w / w.cummax() - 1).min()
    return dict(mean=r.mean() * 12, sd=r.std() * np.sqrt(12), sharpe=r.mean() / r.std() * np.sqrt(12),
                skew=stats.skew(r), kurt=stats.kurtosis(r), min=r.min(), dd=dd)


def sec_h1(conn):
    ff = read_ff()
    md = pd.read_csv(DATA / 'mom_daily.csv')
    md.index = pd.to_datetime(md.date.astype(str))
    rd = md['Mom'] / 100
    rm = pd.read_csv(DATA / 'mom_monthly.csv')
    rm = pd.Series(rm['Mom'].values / 100, index=pd.PeriodIndex([f'{str(m)[:4]}-{str(m)[4:]}' for m in rm.month],
                                                                freq='M'))
    # 月 t の前月末までの 126 取引日で、月次の分散を見積もる（Barroso and Santa-Clara 2015）
    sq = rd ** 2
    roll = sq.rolling(126).mean() * 21
    end_prev = roll.groupby(roll.index.to_period('M')).last().shift(1)
    sig = np.sqrt(end_prev).reindex(rm.index)
    target = 0.12 / np.sqrt(12)
    scale = target / sig
    scaled = rm * scale
    ok = sig.notna()
    plain, sc = rm[ok], scaled[ok]
    SHARED['mom_scale'] = scale
    st_all = (mom_stats(plain), mom_stats(sc))
    sub = plain.index <= pd.Period('2011-12', 'M')
    st_sub = (mom_stats(plain[sub]), mom_stats(sc[sub]))
    rec = plain.index >= pd.Period('2015-01', 'M')
    st_rec = (mom_stats(plain[rec]), mom_stats(sc[rec]))
    rows = []
    for lab, (a, b) in [(f'{plain.index[0]}〜{plain.index[-1]}', st_all), (f'{plain.index[0]}〜2011-12', st_sub),
                        ('2015-01〜' + str(plain.index[-1]), st_rec)]:
        rows.append([lab, '元のまま', pct(a['mean'], 1), pct(a['sd'], 1), fmt(a['sharpe'], 2), fmt(a['skew'], 2),
                     fmt(a['kurt'], 1), pct(a['min'], 1), pct(a['dd'], 0)])
        rows.append(['', 'ボラで調整', pct(b['mean'], 1), pct(b['sd'], 1), fmt(b['sharpe'], 2), fmt(b['skew'], 2),
                     fmt(b['kurt'], 1), pct(b['min'], 1), pct(b['dd'], 0)])
    write_table('mom_bsc', tabular('llrrrrrrr', ['期間', '', '年率平均', '年率標準偏差', 'シャープ', '歪度', '尖度',
                                                 '最悪の月', '最大下落'], rows, midrules=(2, 4)))
    put('mom.start', str(plain.index[0]))
    put('mom.end', str(plain.index[-1]))
    put('mom.sh.all', fmt(st_all[0]['sharpe'], 2))
    put('mom.sh.all.sc', fmt(st_all[1]['sharpe'], 2))
    put('mom.sh.sub', fmt(st_sub[0]['sharpe'], 2))
    put('mom.sh.sub.sc', fmt(st_sub[1]['sharpe'], 2))
    put('mom.dd', pct(st_all[0]['dd'], 0))
    put('mom.dd.sc', pct(st_all[1]['dd'], 0))
    worst = plain.sort_values().head(6)
    rows = [[str(p), pct(v, 1), fmt(scale[p], 2), pct(sc[p], 1)] for p, v in worst.items()]
    write_table('mom_worst', tabular('lrrr', ['月', 'モメンタム', '調整の倍率', '調整後'], rows))
    put('mom.worst.month', str(worst.index[0]))
    put('mom.worst', pct(worst.iloc[0], 1))
    p09 = pd.Period('2009-04', 'M')
    put('mom.2009', pct(rm[p09], 1))
    put('mom.2009.sc', pct(scaled[p09], 1))
    put('mom.2009.scale', fmt(scale[p09], 2))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    x = plain.index.to_timestamp()
    ax.plot(x, (1 + plain).cumprod(), color=C1, lw=1.1, label='モメンタム（元のまま）')
    ax.plot(x, (1 + sc).cumprod(), color=C2, lw=1.1, label='ボラで調整（年率 12% に合わせる）')
    ax.set_yscale('log')
    ax.set_ylabel('資産（開始時 = 1、対数目盛）')
    for yr, lab in [('1932-07', '1932 年'), ('2009-04', '2009 年')]:
        p = pd.Period(yr, 'M')
        ax.annotate(lab, (p.to_timestamp(), (1 + plain).cumprod()[p]), xytext=(0, -22), textcoords='offset points',
                    fontsize=7, color=INK2, ha='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.legend(loc='upper left')
    ticks = pd.date_range('1930-01-01', '2030-01-01', freq='120MS')
    ax.set_xticks(ticks, [t.strftime('%Y') for t in ticks])
    save_fig(fig, 'mom_long')


def sec_h2(conn):
    from sklearn.covariance import LedoitWolf
    px, irx, vix, etfs = load_daily(conn)
    r, rf = monthly(conn)
    sector2etf = {v: k for k, v in etfs.items()}
    cols = [s for s in ETF_ORDER if s in px.columns]
    dates = [d for d in load_month_ends(conn) if '2016-01-01' <= d <= '2026-08-31']
    delta = 2.5
    scale = SHARED.get('mom_scale')
    hist_ic = []
    rows, prev = [], {}
    for d in dates:
        p = px.loc[:d, cols].tail(253)
        avail = [s for s in cols if p[s].notna().sum() >= 253]
        rets = (p[avail] / p[avail].shift(1) - 1).iloc[1:]
        S = LedoitWolf().fit(rets.values).covariance_ * 21
        caps = load_member_caps(conn, d).groupby('gics_sector').market_cap.sum()
        wb = np.array([caps.get(etfs[s], 0.0) for s in avail])
        wb = wb / wb.sum()
        Pi = delta * S @ wb
        m = (p[avail].iloc[-22] / p[avail].iloc[-253] - 1).values
        z = (m - m.mean()) / m.std()
        sig = np.sqrt(np.diag(S))
        per = pd.Period(d[:7], 'M') + 1
        R = r.loc[per, avail].values if per in r.index else None
        if R is None or np.isnan(R).any():
            continue
        n_past = len(hist_ic)
        ic_t = max(0.0, (np.sum(hist_ic) + 24 * 0.05) / (n_past + 24))
        k_t = float(np.clip(scale.get(pd.Period(d[:7], 'M') + 1, np.nan), 0, 1.5)) if scale is not None else 1.0
        if np.isnan(k_t):
            k_t = 1.0
        mus = {'icv': Pi + ic_t * sig * z, 'naive': Pi + m / 11, 'vol': Pi + ic_t * k_t * sig * z}
        rr = R - R.mean()
        hist_ic.append(np.corrcoef(z, rr)[0, 1])
        rec = dict(date=d, ic=hist_ic[-1], ic_t=ic_t, k=k_t, bench=float(wb @ R), rf=float(rf.get(per, 0.0)),
                   n=len(avail))
        for name, mu in mus.items():
            w = mv_long_only(mu, S, delta)
            rec[name] = float(w @ R)
            rec[name + '_effn'] = eff_n(w)
            ws = pd.Series(w, index=avail)
            if name in prev:
                idx = ws.index.union(prev[name].index)
                rec[name + '_to'] = 0.5 * float((ws.reindex(idx).fillna(0) - prev[name].reindex(idx).fillna(0)).abs().sum())
            prev[name] = ws
            rec[name + '_act'] = 0.5 * float(np.abs(w - wb).sum())
        rows.append(rec)
    res = pd.DataFrame(rows).set_index('date')
    SHARED['secmom'] = res
    T = len(res)
    put('sm.months', T)
    put('sm.start', res.index[0][:7])
    put('sm.end', res.index[-1][:7])
    put('sm.cic.mean', sfmt(res.ic.mean(), 3))
    put('sm.cic.t', fmt(res.ic.mean() / res.ic.std() * np.sqrt(T), 2))
    put('sm.cic.last', fmt(res.ic_t.iloc[-1], 3))
    put('sm.cic.pos', int((res.ic > 0).sum()))
    labels = {'bench': '時価総額加重（ベンチマーク）', 'icv': 'IC で縮めたビュー', 'naive': '過去 11 か月の平均がそのまま続く',
              'vol': 'IC ＋ モメンタムのボラで調整'}
    out = []
    for name in ['icv', 'vol', 'naive', 'bench']:
        rr = res[name]
        ex = rr - res.rf
        act = rr - res.bench
        ann = (1 + rr).prod() ** (12 / T) - 1
        te = act.std() * np.sqrt(12)
        wealth = (1 + rr).cumprod()
        dd = (wealth / wealth.cummax() - 1).min()
        row = [labels[name], pct(ann, 1), pct(rr.std() * np.sqrt(12), 1), fmt(ex.mean() / ex.std() * np.sqrt(12), 2),
               pct(dd, 0)]
        if name != 'bench':
            row += [pct(te, 1), fmt(act.mean() * 12 / te, 2), fmt(act.mean() / act.std() * np.sqrt(T), 2),
                    pct(res[name + '_to'].mean(), 0), fmt(res[name + '_effn'].mean(), 1)]
            put(f'sm.{name}.ir', fmt(act.mean() * 12 / te, 2))
            put(f'sm.{name}.te', pct(te, 1))
            put(f'sm.{name}.t', fmt(act.mean() / act.std() * np.sqrt(T), 2))
            put(f'sm.{name}.to', pct(res[name + '_to'].mean(), 0))
        else:
            row += ['---'] * 5
        put(f'sm.{name}.ann', pct(ann, 1))
        out.append(row)
    write_table('secmom', tabular('lrrrrrrrrr', ['戦略', '年率', 'リスク', 'シャープ', '最大下落', 'TE', 'IR',
                                                 '$t$ 値', '回転率', '実質数'], out))
    # 図：IC
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    x = pd.to_datetime(res.index)
    ax.bar(x, res.ic, width=20, color=[C1 if v >= 0 else C2 for v in res.ic])
    ax.plot(x, res.ic.expanding().mean(), color=INK2, lw=1.2, label='それまでの平均')
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel('セクターの順位の当たり（相関）')
    ax.legend(loc='upper right')
    date_axis(ax, '2016', '2027')
    save_fig(fig, 'secmom_ic')
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ends = []
    for name, col in [('icv', C1), ('vol', C2), ('naive', C3)]:
        cum = (1 + res[name]).cumprod() / (1 + res.bench).cumprod()
        ax.plot(x, cum, color=col, label=labels[name])
        ends.append((cum.iloc[-1], f'{cum.iloc[-1]:.2f}'))
    ax.axhline(1, color=INK2, lw=0.6)
    end_labels(ax, x[-1] + pd.Timedelta(days=20), ends)
    ax.set_ylabel('ベンチマークに対する比（開始時 = 1）')
    ax.legend(loc='upper left')
    date_axis(ax, '2016', '2027')
    save_fig(fig, 'secmom_cum')
    # 最後の判断日の入力を、I〜K の例で使う
    d = dates[-1]
    p = px.loc[:d, cols].tail(253)
    rets = (p / p.shift(1) - 1).iloc[1:]
    S = LedoitWolf().fit(rets.values).covariance_ * 21
    caps = load_member_caps(conn, d).groupby('gics_sector').market_cap.sum()
    wb = np.array([caps.get(etfs[s], 0.0) for s in cols])
    wb = wb / wb.sum()
    SHARED['last'] = dict(date=d, cols=cols, S=S, wb=wb, Pi=delta * S @ wb)


# ============================================================
# I. エントロピー・プーリング（Meucci 2008）
# ============================================================
def entropy_pooling(p, Aeq, beq, Aineq=None, bineq=None):
    """min Σ q log(q/p)  s.t.  Aeq q = beq, Aineq q <= bineq。双対を解いて q を返す。"""
    A = Aeq if Aineq is None else np.vstack([Aeq, Aineq])
    b = beq if Aineq is None else np.concatenate([beq, bineq])
    ne = len(beq)
    sc = np.maximum(A.std(axis=1), 1e-12)
    sc[np.all(A == A[:, :1], axis=1)] = 1.0
    A = A / sc[:, None]
    b = b / sc
    logp = np.log(p)

    def f(lam):
        x = logp - 1 - A.T @ lam
        x = np.minimum(x, 700)
        q = np.exp(x)
        return q.sum() + lam @ b, b - A @ q

    bnds = [(None, None)] * ne + [(0, None)] * (len(b) - ne)
    res = optimize.minimize(f, np.zeros(len(b)), jac=True, method='L-BFGS-B', bounds=bnds,
                            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
    q = np.exp(logp - 1 - A.T @ res.x)
    return q / q.sum()


def wstats(X, q):
    m = X.T @ q
    sd = np.sqrt(((X - m) ** 2).T @ q)
    return m, sd


def sec_i(conn):
    L = SHARED['last']
    cols, S, Pi = L['cols'], L['S'], L['Pi']
    n = len(cols)
    J, nu = 20000, 5
    rng = np.random.default_rng(202)
    Z = rng.multivariate_normal(np.zeros(n), S * (nu - 2) / nu, J)
    W = rng.chisquare(nu, J) / nu
    X = Pi + Z / np.sqrt(W)[:, None]
    p = np.full(J, 1 / J)
    SHARED['scen'] = dict(X=X, p=p)
    iK, iU, iE, iF = cols.index('XLK'), cols.index('XLU'), cols.index('XLE'), cols.index('XLF')
    m0, sd0 = wstats(X, p)
    tail0 = float(p @ (X[:, iF] <= -0.10))
    spread0 = m0[iK] - m0[iU]
    put('ep.J', f'{J:,}'.replace(',', '{,}'))
    put('ep.date', L['date'])
    put('ep.spread0', pct(spread0, 2))
    put('ep.tail0', pct(tail0, 2))
    put('ep.sdE0', pct(sd0[iE], 2))
    Aeq = np.vstack([np.ones(J), X[:, iE], X[:, iE] ** 2, (X[:, iF] <= -0.10).astype(float)])
    beq = np.array([1, m0[iE], m0[iE] ** 2 + (1.3 * sd0[iE]) ** 2, 0.03])
    Ain = -(X[:, iK] - X[:, iU])[None, :]
    bin_ = np.array([-0.01])
    q = entropy_pooling(p, Aeq, beq, Ain, bin_)
    m1, sd1 = wstats(X, q)
    SHARED['scen']['q'] = q
    put('ep.spread1', pct(m1[iK] - m1[iU], 2))
    put('ep.tail1', pct(float(q @ (X[:, iF] <= -0.10)), 2))
    put('ep.sdE1', pct(sd1[iE], 2))
    put('ep.ens', pct(np.exp(-np.sum(q * np.log(q))) / J, 1))
    put('ep.kl', fmt(np.sum(q * np.log(q / p)), 3))
    qc = 0.5 * p + 0.5 * q
    mc, _ = wstats(X, qc)
    put('ep.c.spread', pct(mc[iK] - mc[iU], 2))
    put('ep.c.tail', pct(float(qc @ (X[:, iF] <= -0.10)), 2))
    # 平均のビューだけ（等式）なら、市場版 BL の条件付きとほぼ同じになる
    q_eq = entropy_pooling(p, np.vstack([np.ones(J), X[:, iK] - X[:, iU]]), np.array([1, 0.01]))
    m_eq, _ = wstats(X, q_eq)
    P = np.zeros((1, n))
    P[0, iK], P[0, iU] = 1, -1
    mu_c, _ = bl_market(m0, np.cov(X.T, aweights=p), P, np.array([0.01]), np.array([[1e-12]]))
    rows = []
    for i, s in enumerate(cols):
        rows.append([ETF_JA[s], pct(m0[i], 2), pct(m1[i], 2), pct(sd0[i], 1), pct(sd1[i], 1),
                     pct(float(p @ (X[:, i] <= -0.10)), 1), pct(float(q @ (X[:, i] <= -0.10)), 1),
                     spct(m_eq[i] - m0[i], 2), spct(mu_c[i] - m0[i], 2)])
    write_table('ep', tabular('lrrrrrrrr', ['セクター', '平均（前）', '平均（後）', '標準偏差（前）', '（後）',
                                            '$\\le-10\\%$（前）', '（後）', '平均のみ：EP', '同：BL'], rows))
    put('ep.maxdiff', fmt(100 * np.max(np.abs((m_eq - m0) - (mu_c - m0))), 3))
    # 図
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, i, title in [(axs[0], iF, '金融：−10% 以下の確率を 3% に'), (axs[1], iE, 'エネルギー：標準偏差を 1.3 倍に')]:
        bins = np.linspace(np.percentile(X[:, i], 0.2), np.percentile(X[:, i], 99.8), 60)
        ax.hist(100 * X[:, i], bins=100 * bins, weights=p, color=C1, alpha=0.55, label='事前（シナリオは同じ）')
        ax.hist(100 * X[:, i], bins=100 * bins, weights=q, histtype='step', color=C2, lw=1.4, label='事後（確率を変更）')
        ax.set_title(title)
        ax.set_xlabel('月次リターン（%）')
        ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('確率')
    axs[0].axvline(-10, color=INK2, lw=0.6, ls='--')
    axs[1].legend(loc='upper left')
    save_fig(fig, 'ep')


# ============================================================
# J. コピュラ・オピニオン・プーリング（Meucci 2006）
# ============================================================
def sec_j(conn):
    from scipy.linalg import null_space
    L = SHARED['last']
    cols, S = L['cols'], L['S']
    X, p = SHARED['scen']['X'], SHARED['scen']['p']
    J, n = X.shape
    iK, iU = cols.index('XLK'), cols.index('XLU')
    P = np.zeros(n)
    P[iK], P[iU] = 1, -1
    Pbar = np.vstack([P, null_space(P[None, :]).T])
    V = X @ Pbar.T
    v1 = V[:, 0]
    u = stats.rankdata(v1) / (J + 1)
    c, loc, scl, df_ = 0.5, 0.02, 0.02, 4
    grid = np.linspace(min(v1.min(), loc - 15 * scl), max(v1.max(), loc + 15 * scl), 20001)
    Femp = np.searchsorted(np.sort(v1), grid, side='right') / J
    Fmix = (1 - c) * Femp + c * stats.t.cdf((grid - loc) / scl, df_)
    Fmix = np.maximum.accumulate(Fmix + np.linspace(0, 1e-12, len(grid)))
    v1n = np.interp(u, Fmix, grid)
    Vn = V.copy()
    Vn[:, 0] = v1n
    Xn = Vn @ np.linalg.inv(Pbar.T)
    m0, sd0 = wstats(X, p)
    m1, sd1 = wstats(Xn, p)
    put('cop.c', fmt(c, 1))
    put('cop.loc', pct(loc, 0))
    put('cop.scale', pct(scl, 0))
    put('cop.m0', pct(v1.mean(), 2))
    put('cop.m1', pct(v1n.mean(), 2))
    put('cop.sd0', pct(v1.std(), 2))
    put('cop.sd1', pct(v1n.std(), 2))
    put('cop.sk0', fmt(stats.skew(v1), 2))
    put('cop.sk1', fmt(stats.skew(v1n), 2))
    put('cop.q05', pct(np.percentile(v1n, 5), 1))
    put('cop.q95', pct(np.percentile(v1n, 95), 1))
    put('cop.q05.0', pct(np.percentile(v1, 5), 1))
    put('cop.q95.0', pct(np.percentile(v1, 95), 1))
    put('cop.otherdiff', fmt(100 * np.max(np.abs(np.delete(m1 - m0, [iK, iU]))), 4))
    # 比べる：市場版 BL で、同じ平均（＝ COP の事後の平均）を c=1 相当の不確かさで
    Om = np.array([[P @ S @ P]])
    mu_b, _ = bl_market(m0, np.cov(X.T), P[None, :], np.array([v1n.mean()]), 1e-12 * Om)
    rows = [[ETF_JA[s], pct(m0[i], 2), spct(m1[i] - m0[i], 2), spct(mu_b[i] - m0[i], 2)] for i, s in enumerate(cols)]
    write_table('cop', tabular('lrrr', ['セクター', '事前の平均', 'COP の変化', '市場版 BL の変化（同じ平均へ）'], rows))
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    bins = np.linspace(-0.12, 0.16, 90)
    ax.hist(100 * v1, bins=100 * bins, density=True, color=C1, alpha=0.55, label='事前の分布（スプレッド）')
    ax.hist(100 * v1n, bins=100 * bins, density=True, histtype='step', color=C2, lw=1.4,
            label=f'事後（確信度 c = {c}）')
    xs = np.linspace(-12, 16, 400)
    ax.plot(xs, stats.t.pdf((xs / 100 - loc) / scl, df_) / scl / 100, color=C3, lw=1.1, ls='--',
            label='ビューの分布（t 分布）')
    ax.set_xlabel('情報技術 − 公益 の月次リターン（%）')
    ax.set_ylabel('密度')
    ax.legend(loc='upper left')
    ax.grid(axis='x', visible=False)
    save_fig(fig, 'cop')


# ============================================================
# K. CVaR の最適化（Rockafellar and Uryasev 2000）
# ============================================================
def min_risk(X, q, mu, target, cap, kind, beta=0.95):
    n = X.shape[1]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap, mu @ w >= target]
    if kind == 'cvar':
        zeta = cp.Variable()
        u = cp.Variable(len(q))
        cons += [u >= 0, u >= -X @ w - zeta]
        obj = zeta + (q @ u) / (1 - beta)
    else:
        C = np.cov(X.T, aweights=q)
        obj = cp.quad_form(w, cp.psd_wrap(C))
    cp.Problem(cp.Minimize(100 * obj), cons).solve(solver=cp.CLARABEL)
    x = np.clip(w.value, 0, None)
    return x / x.sum()


def risk_of(X, q, w, beta=0.95):
    r = X @ w
    order = np.argsort(r)
    cq = np.cumsum(q[order])
    var = -r[order][np.searchsorted(cq, 1 - beta)]
    tail = r <= -var
    cvar = -(q[tail] @ r[tail]) / q[tail].sum()
    m = q @ r
    sd = np.sqrt(q @ (r - m) ** 2)
    return m, sd, var, cvar


def sec_k(conn):
    L = SHARED['last']
    cols, wb = L['cols'], L['wb']
    X, q = SHARED['scen']['X'], SHARED['scen']['q']
    mu = X.T @ q
    target = float(mu @ wb)
    put('cv.target', pct(target, 2))
    sols = {'mv40': min_risk(X, q, mu, target, 0.4, 'mv'), 'cv30': min_risk(X, q, mu, target, 0.3, 'cvar'),
            'cv40': min_risk(X, q, mu, target, 0.4, 'cvar'), 'cv50': min_risk(X, q, mu, target, 0.5, 'cvar')}
    rows = [[ETF_JA[s], pct(wb[i], 1)] + [pct(sols[k][i], 1) for k in ['mv40', 'cv30', 'cv40', 'cv50']]
            for i, s in enumerate(cols)]
    stats_rows = []
    vals ={k: risk_of(X, q, w) for k, w in [('wb', wb)] + list(sols.items())}
    for j, lab in enumerate(['期待リターン', '標準偏差', 'VaR 95\\%', 'CVaR 95\\%']):
        stats_rows.append([lab] + [pct(vals[k][j], 2) for k in ['wb', 'mv40', 'cv30', 'cv40', 'cv50']])
    write_table('cvar', tabular('lrrrrr', ['セクター', 'ベンチマーク', '分散最小（上限 40\\%）', 'CVaR 最小（30\\%）',
                                           '（40\\%）', '（50\\%）'], rows + stats_rows, midrules=(len(rows),)))
    put('cv.mv.cvar', pct(vals['mv40'][3], 2))
    put('cv.cv.cvar', pct(vals['cv40'][3], 2))
    put('cv.mv.sd', pct(vals['mv40'][1], 2))
    put('cv.cv.sd', pct(vals['cv40'][1], 2))


# ============================================================
# L. トラッキングエラーと情報レシオ（Idzorek の例）
# ============================================================
def active_ir(alpha, S, wb=None):
    """1'w = 0（と、必要なら β = 0）の下での最大の情報レシオと、その向き。"""
    Si = np.linalg.inv(S)
    A = [np.ones(len(alpha))]
    if wb is not None:
        A.append(S @ wb)
    A = np.array(A)
    c = A @ Si @ alpha
    M = A @ Si @ A.T
    h = Si @ alpha - Si @ A.T @ np.linalg.solve(M, c)
    ir = np.sqrt(alpha @ Si @ alpha - c @ np.linalg.solve(M, c))
    return ir, h


def sec_l():
    d = SHARED['idz']
    S, w = A_S, A_W
    rows = []
    for lab, mu in [('He–Litterman の Ω', d['mu_hl']), ('Idzorek の Ω', d['mu_i'])]:
        alpha = mu - d['Pi']
        ir, h = active_ir(alpha, S)
        ir_b, h_b = active_ir(alpha, S, w)
        wa = 0.02 / np.sqrt(h @ S @ h) * h
        beta_a = wa @ S @ w / (w @ S @ w)
        rows.append([lab, fmt(ir, 3), pct(0.02 * ir, 3), fmt(beta_a, 3), fmt(ir_b, 3)])
    write_table('te', tabular('lrrrr', ['事後の期待リターン', '最大の IR', 'TE 2\\% の超過リターン',
                                        'そのときの $\\beta$ のずれ', 'IR（$\\beta$ 中立）'], rows))
    alpha = d['mu_hl'] - d['Pi']
    ir, h = active_ir(alpha, S)
    put('te.ir.hl', fmt(ir, 3))
    # τ と TE（Ω 固定）、確信度と IR
    om_fixed = np.diag(d['Om_hl'])
    tt = np.logspace(-3, 0, 60)
    te_tau = []
    for t in tt:
        mu, _ = bl(d['Pi'], S, A_P, A_Q, np.diag(om_fixed), t)
        wt = np.linalg.solve(d['lam'] * S, mu)
        te_tau.append(np.sqrt((wt - w) @ S @ (wt - w)))
    cs = np.linspace(0.02, 0.98, 60)
    ir_c, te_c = [], []
    for cc in cs:
        om = A_TAU * d['pSp'] * (1 - cc) / cc
        mu, _ = bl(d['Pi'], S, A_P, A_Q, np.diag(om), A_TAU)
        ir_c.append(active_ir(mu - d['Pi'], S)[0])
        wt = np.linalg.solve(d['lam'] * S, mu)
        te_c.append(np.sqrt((wt - w) @ S @ (wt - w)))
    te_tau = np.array(te_tau)
    i2 = int(np.argmin(np.abs(te_tau - 0.02)))
    put('te.tau2', fmt(tt[i2], 3))
    mu, _ = bl(d['Pi'], S, A_P, A_Q, np.diag(om_fixed), A_TAU)
    wt = np.linalg.solve(d['lam'] * S, mu)
    put('te.tau.base', pct(np.sqrt((wt - w) @ S @ (wt - w)), 3))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    axs[0].plot(tt, 100 * te_tau, color=C1)
    axs[0].axhline(2, color=NEUTRAL, lw=0.8, ls='--')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('τ（Ω は固定、対数目盛）')
    axs[0].set_ylabel('BL の傾きの TE（年率 %）')
    axs[0].set_title('τ を動かして TE を合わせる')
    axs[1].plot(100 * cs, ir_c, color=C2)
    axs[1].set_xlabel('3 つのビューに共通の確信度 C（%）')
    axs[1].set_ylabel('事前の最大 IR')
    axs[1].set_title('確信度と事前の情報レシオ')
    save_fig(fig, 'te')
    put('te.ir.c90', fmt(ir_c[int(np.argmin(np.abs(cs - 0.9)))], 2))
    put('te.ir.c50', fmt(ir_c[int(np.argmin(np.abs(cs - 0.5)))], 2))


# ============================================================
# M. ファクターへのビュー（Fama-French の 5 ファクター ＋ モメンタム）
# ============================================================
def sec_m(conn):
    r, rf = monthly(conn)
    ff = read_ff()
    fac = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    fac_ja = ['市場', '規模', 'バリュー', '収益性', '投資', 'モメンタム']
    idx = pd.period_range('2015-02', '2026-08', freq='M')
    F = ff.loc[ff.index.isin(idx), fac]
    idx = F.index
    Y = r.loc[idx, OLD9].sub(ff.loc[idx, 'RF'], axis=0)
    B, D = [], []
    for s in OLD9:
        Xs = np.column_stack([np.ones(len(idx)), F.values])
        b, se, e = ols(Y[s].values, Xs, white=False)
        B.append(b[1:])
        D.append(e.var(ddof=Xs.shape[1]))
    B = np.array(B)
    D = np.diag(D)
    Fcov = np.cov(F.values.T)
    long = ff[(ff.index >= pd.Period('1963-07', 'M')) & (ff.index <= pd.Period('2014-12', 'M'))][fac]
    xi = long.mean().values
    Flong = np.cov(long.values.T)
    T_long = len(long)
    put('fac.Tlong', T_long)
    put('fac.n', len(idx))
    iM = fac.index('Mom')
    put('fac.mom.prior', pct(xi[iM] * 12, 1))
    rows_t = []
    res = {}
    for T_eff in [T_long, 120, 36]:
        V = Flong / T_eff
        p = np.zeros((1, len(fac)))
        p[0, iM] = 1
        om = np.array([[Flong[iM, iM] / 12]])
        post = xi + V @ p.T @ np.linalg.solve(p @ V @ p.T + om, np.array([0.0]) - p @ xi)
        res[T_eff] = post
        adopt = V[iM, iM] / (V[iM, iM] + om[0, 0])
        rows_t.append([f'{T_eff}', pct(adopt, 1), pct(post[iM] * 12, 2)] +
                      [spct((post[j] - xi[j]) * 12, 2) for j in range(len(fac)) if j != iM])
    write_table('fac_prior', tabular('rrr' + 'r' * (len(fac) - 1),
                                     ['$T_{\\mathrm{eff}}$（月）', '採用率', 'モメンタム（年率）'] +
                                     [f'{fac_ja[j]}の変化' for j in range(len(fac)) if j != iM], rows_t))
    post = res[120]
    rows = []
    for i, s in enumerate(OLD9):
        before = B[i] @ xi * 12
        after = B[i] @ post * 12
        rows.append([ETF_JA[s], fmt(B[i, 0], 2), sfmt(B[i, iM], 2), pct(before, 2), pct(after, 2),
                     spct(after - before, 2)])
    write_table('fac_assets', tabular('lrrrrr', ['セクター', '市場 $\\beta$', 'モメンタム $\\beta$', '事前（年率）',
                                                 '事後（年率）', '変化'], rows))
    put('fac.adopt120', pct((Flong[iM, iM] / 120) / (Flong[iM, iM] / 120 + Flong[iM, iM] / 12), 1))
    Sig = B @ Fcov @ B.T + D
    put('fac.share', pct(np.mean(np.diag(B @ Fcov @ B.T) / np.diag(Sig)), 0))


# ============================================================
# N. 複数の予想をまとめる（対象論文の表 5 の誤差を使う）
# ============================================================
def sec_n():
    names = ['Gemma', 'Qwen', 'Llama', 'GPT']
    mse = np.array([1.2373, 0.5125, 0.5288, 0.6505])
    w = (1 / mse) / np.sum(1 / mse)
    rows = [[n, fmt(m, 4), pct(wi, 1)] for n, m, wi in zip(names, mse, w)]
    write_table('fusion_w', tabular('lrr', ['モデル', 'MSE（論文の表 5）', '精度加重の重み'], rows))
    best = mse.min()
    sd = np.sqrt(mse)
    rows = []
    for rho in [0.0, 0.5, 0.8, 0.9, 0.95]:
        R = np.full((4, 4), rho)
        np.fill_diagonal(R, 1)
        C = np.outer(sd, sd) * R
        Ci = np.linalg.inv(C)
        one = np.ones(4)
        var_opt = 1 / (one @ Ci @ one)
        var_pw = w @ C @ w
        rows.append([fmt(rho, 2), fmt(var_pw, 4), fmt(var_pw / best, 3), fmt(var_opt, 4), fmt(var_opt / best, 3)])
    write_table('fusion_rho', tabular('rrrrr', ['誤差の相関 $\\rho$', '精度加重の MSE', '最良との比',
                                                '最適な組み合わせの MSE', '最良との比'], rows))
    put('fu.pw.indep', fmt(1 / np.sum(1 / mse), 4))
    put('fu.best', fmt(best, 4))


# ============================================================
# O. ESG で共分散を膨らませる案の検討（例）
# ============================================================
def sec_o():
    sig = np.array([0.20, 0.22, 0.25, 0.30])
    R = np.full((4, 4), 0.5)
    np.fill_diagonal(R, 1)
    S = np.outer(sig, sig) * R
    w = np.array([0.4, 0.3, 0.2, 0.1])
    esg = np.array([0.9, 0.7, 0.4, 0.1])
    delta, tau = 2.5, 0.05
    Dm = np.diag(np.sqrt(2 - esg))
    St = Dm @ S @ Dm
    Pi = delta * S @ w
    Pit = delta * St @ w
    w1 = np.linalg.solve(delta * St, Pit)
    w2 = np.linalg.solve(delta * St, Pi)
    w2c = mv_long_only(Pi, St, delta)
    om = 0.04 ** 2
    ad = tau * np.diag(S) / (tau * np.diag(S) + om)
    adt = tau * np.diag(St) / (tau * np.diag(St) + om)
    lam_esg = 0.25
    w_ar = (1 - lam_esg) * w + lam_esg * esg / esg.sum()
    rows = [[f'{chr(65 + i)}', fmt(esg[i], 1), fmt(np.sqrt(2 - esg[i]), 3), pct(w[i], 0), pct(w1[i], 1),
             pct(w2c[i], 1), pct(ad[i], 1), pct(adt[i], 1), pct(w_ar[i], 1)] for i in range(4)]
    write_table('esg', tabular('crrrrrrrr', ['資産', 'ESG', '$\\sqrt{2-\\mathrm{ESG}}$', '$w_{\\mathrm{mkt}}$',
                                             '(1) $\\tilde\\Pi$ で', '(2) $\\Pi$ のまま', '採用率（$\\Sigma$）',
                                             '採用率（$\\tilde\\Sigma$）', 'Alpern–Rachev'], rows))
    put('esg.w1diff', fmt(100 * np.abs(w1 - w).max(), 6))


# ============================================================
# P. 立場の確率からビューを作る（neural predicate の考え方）
# ============================================================
def sec_p():
    comps = [('A', 0.80, 0.15, 0.05), ('B', 0.40, 0.30, 0.30), ('C', 0.10, 0.20, 0.70), ('D', 0.34, 0.33, 0.33)]
    r_bull, r_bear, pi_i, sig, tau = 0.02, -0.02, 0.006, 0.08, 0.05
    a_min, a_max = 1.0, 20.0
    rows = []
    for name, pb, pn, pr in comps:
        pr_ = np.array([pb, pn, pr])
        q = pb * r_bull + pr * r_bear + pn * pi_i
        H = -np.sum(pr_ * np.log(pr_)) / np.log(3)
        ratio = a_min + (a_max - a_min) * H
        adopt = 1 / (1 + ratio)
        mu = pi_i + adopt * (q - pi_i)
        rows.append([name, fmt(pb, 2), fmt(pn, 2), fmt(pr, 2), pct(q, 2), fmt(H, 2), fmt(ratio, 1), pct(adopt, 1),
                     pct(mu, 2)])
    write_table('stance', tabular('lrrrrrrrr', ['銘柄', '強気', '中立', '弱気', '$q_i$', '$H/\\log 3$',
                                                '$\\omega/\\tau\\sigma^2$', '採用率', '事後の平均'], rows))


# ============================================================
# S. ビューが多いときの確信の積み上がり（等相関のモデルと、本プロジェクトの外れの共通部分）
# ============================================================
def stack_adoption(K, sig2, rho, om, rho_e, rho_e_assumed):
    """リターンも外れも等相関のとき、K 個の絶対ビューの「平均の部分」と「平均からの差の部分」の採用率（市場版）。"""
    vx_c = sig2 * (rho + (1 - rho) / K)
    ve_c_true = om * (rho_e + (1 - rho_e) / K)
    ve_c_asm = om * (rho_e_assumed + (1 - rho_e_assumed) / K)
    vx_d = sig2 * (1 - rho)
    ve_d_asm = om * (1 - rho_e_assumed)
    ve_d_true = om * (1 - rho_e)
    return (vx_c / (vx_c + ve_c_asm), vx_c / (vx_c + ve_c_true), vx_d / (vx_d + ve_d_asm), vx_d / (vx_d + ve_d_true))


def sec_s():
    sig, rho, ic = 0.08, 0.3, 0.05
    sig2 = sig ** 2
    om = sig2 * (1 / ic ** 2 - 1)
    put('stk.sig', pct(sig, 0))
    put('stk.rho', fmt(rho, 1))
    put('stk.ic', fmt(ic, 2))
    # 行列で直接計算して、分解の式と一致することを確かめる（K = 50、外れの相関 0.3、対角と仮定）
    K, re_ = 50, 0.3
    S = sig2 * ((1 - rho) * np.eye(K) + rho * np.ones((K, K)))
    Om_diag = om * np.eye(K)
    G = S @ np.linalg.inv(S + Om_diag)      # 市場版の BL で、v - π にかかる行列
    one = np.ones(K) / np.sqrt(K)
    s_common_mat = one @ G @ one
    s_c_formula = stack_adoption(K, sig2, rho, om, re_, 0.0)[0]
    put('stk.check', sci(abs(s_common_mat - s_c_formula), 1))
    # 表：K = 1, 10, 100, 500 で、平均の部分と差の部分の採用率
    rows = []
    for K in [1, 10, 100, 500]:
        for re_ in [0.0, 0.3, 0.6]:
            a_c, t_c, a_d, t_d = stack_adoption(K, sig2, rho, om, re_, 0.0)
            dev = ['---', '---'] if K == 1 else [pct(a_d, 2), pct(t_d, 2)]   # ビューが 1 つなら差の部分はない
            rows.append([f'{K}' if re_ == 0 else '', fmt(re_, 1), pct(a_c, 2), pct(t_c, 2)] + dev)
    write_table('stack', tabular('rrrrrr', ['$K$', '外れの相関 $\\rho_e$', '平均の部分（対角）', '平均の部分（正しい）',
                                            '差の部分（対角）', '差の部分（正しい）'], rows, midrules=(3, 6, 9)))
    a500 = stack_adoption(500, sig2, rho, om, 0.3, 0.0)
    put('stk.k500.diag', pct(a500[0], 1))
    put('stk.k500.true', pct(a500[1], 2))
    put('stk.single', pct(ic ** 2, 2))
    Ks = np.unique(np.round(np.logspace(0, np.log10(500), 80)).astype(int))
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.plot(Ks, [100 * stack_adoption(k, sig2, rho, om, 0.3, 0.0)[0] for k in Ks], color=C1,
            label='Ω を対角とした場合（外れは独立と仮定）')
    ax.plot(Ks, [100 * stack_adoption(k, sig2, rho, om, 0.3, 0.3)[1] for k in Ks], color=C2,
            label='外れの相関 0.3 を正しく入れた場合')
    ax.plot(Ks, [100 * stack_adoption(k, sig2, rho, om, 0.6, 0.6)[1] for k in Ks], color=C3,
            label='外れの相関 0.6 を正しく入れた場合')
    ax.set_xscale('log')
    ax.set_xlabel('ビューの数 K（対数目盛）')
    ax.set_ylabel('平均の部分の採用率（%）')
    ax.legend(loc='upper left')
    save_fig(fig, 'stack')
    # 本プロジェクトの外れの共通部分：月ごとの平均の外れの分散 ÷ 銘柄ごとの外れの分散
    panel = SHARED.get('llm_panel')
    if panel is not None:
        e = panel.Q - panel.r_next
        g = pd.DataFrame({'date': panel.date, 'e': e})
        mean_e = g.groupby('date').e.mean()
        var_i = g.groupby('date').e.var().mean()
        n_i = g.groupby('date').e.size().mean()
        rho_hat = (mean_e.var() - var_i / n_i) / var_i
        put('stk.rhoe', fmt(rho_hat, 2))
        put('stk.common.share', pct(mean_e.var() / (mean_e.var() + var_i), 0))
        # 予想の外れのうち、実現値の側（翌月のリターンの平均）と予想の側を分ける
        q_mean = panel.groupby('date').Q.mean()
        r_mean = panel.groupby('date').r_next.mean()
        put('stk.qlevel.sd', pct(q_mean.std(), 3))
        put('stk.rlevel.sd', pct(r_mean.std(), 3))
        put('stk.level.corr', fmt(np.corrcoef(q_mean, r_mean)[0, 1], 2))
    # 2 資産の手計算の例（第 2 章）：相関 0.5、資産 1 だけに絶対ビュー
    s1, s2, r12 = 0.20, 0.15, 0.5
    S2 = np.array([[s1 ** 2, r12 * s1 * s2], [r12 * s1 * s2, s2 ** 2]])
    w2 = np.array([0.6, 0.4])
    d2, t2 = 2.5, 0.05
    Pi2 = d2 * S2 @ w2
    P2 = np.array([[1.0, 0.0]])
    Q2 = np.array([Pi2[0] + 0.02])
    Om2 = np.array([[t2 * S2[0, 0]]])
    mu2, _ = bl(Pi2, S2, P2, Q2, Om2, t2)
    w2s = np.linalg.solve(d2 * S2, mu2)
    put('two.pi1', pct(Pi2[0], 2))
    put('two.pi2', pct(Pi2[1], 2))
    put('two.q', pct(Q2[0], 2))
    put('two.mu1', pct(mu2[0], 2))
    put('two.mu2', pct(mu2[1], 2))
    put('two.w1', pct(w2s[0], 1))
    put('two.w2', pct(w2s[1], 1))
    put('two.shift2', pct(mu2[1] - Pi2[1], 2))
    put('two.beta21', fmt(S2[0, 1] / S2[0, 0], 3))


# ============================================================
# T. 銘柄の指標の当たり具合（stock_metrics と翌月のリターンの順位相関）
# ============================================================
T_METRICS = [('momentum_12m', 'モメンタム（12--1 か月）', 1), ('return_1m', '直近 1 か月のリターン', 1),
             ('relative_momentum_12m', 'セクター内のモメンタム', 1), ('volatility_30d', 'ボラティリティ（30 日）', 1),
             ('beta_1y', 'ベータ（1 年）', 1), ('pe_trailing', '割安さ（PER が低い順）', -1),
             ('pb', '割安さ（PBR が低い順）', -1), ('roe', 'ROE', 1), ('net_margin', '純利益率', 1),
             ('revenue_growth_yoy', '売上の伸び（前年比）', 1), ('debt_to_equity', '負債比率', 1)]


def sec_t(conn):
    names = [m for m, _, _ in T_METRICS]
    mt = pd.read_sql(f"SELECT as_of_date, symbol, metric, value FROM stock_metrics WHERE metric IN "
                     f"({','.join('?' * len(names))})", conn, params=names)
    wide = mt.pivot_table(index=['as_of_date', 'symbol'], columns='metric', values='value')
    dates = sorted(mt.as_of_date.unique())
    syms = sorted(mt.symbol.unique())
    px = load_prices(conn, syms, '2015-01-01', '2026-09-30')
    me = px.resample('ME').last()
    me.index = me.index.to_period('M')
    nxt = me.shift(-1) / me - 1
    rows, series = [], {}
    for m, ja, sign in T_METRICS:
        ics = []
        for d in dates:
            per = pd.Period(d[:7], 'M')
            if per not in nxt.index or d not in wide.index.get_level_values(0):
                continue
            x = wide.loc[d][m].dropna() * sign
            r = nxt.loc[per].reindex(x.index)
            ok = r.notna()
            if ok.sum() < 100:
                continue
            ics.append(stats.spearmanr(x[ok], r[ok])[0])
        ics = np.array(ics)
        series[m] = ics
        mean, sd, T = ics.mean(), ics.std(ddof=1), len(ics)
        rows.append([ja, f'{T}', sfmt(mean, 3), fmt(mean / sd * np.sqrt(T), 2), pct((ics > 0).mean(), 0),
                     pct(mean ** 2, 2), pct(abs(mean) * 0.08, 2)])
        put(f'tic.{m}.mean', sfmt(mean, 3))
        put(f'tic.{m}.t', fmt(mean / sd * np.sqrt(T), 2))
    write_table('stock_ic', tabular('lrrrrrr', ['指標', '月数', 'IC の平均', '$t$ 値', 'プラスの月', '採用率 $\\mathrm{IC}^2$',
                                                '$|\\alpha|$（$z=1$、$\\sigma=8\\%$）'], rows))
    put('tic.start', dates[0][:7])
    put('tic.end', dates[-2][:7])
    means = pd.Series({ja: series[m].mean() for m, ja, _ in T_METRICS})
    ses = pd.Series({ja: series[m].std(ddof=1) / np.sqrt(len(series[m])) for m, ja, _ in T_METRICS})
    order = means.sort_values().index
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(order))
    ax.barh(y, means[order], xerr=2 * ses[order], color=[C1 if v >= 0 else C2 for v in means[order]], height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_yticks(y, order)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel('翌月のリターンとの順位相関の平均（線は ±2 標準誤差）')
    ax.grid(axis='y', visible=False)
    save_fig(fig, 'stock_ic')


# ============================================================
# 実行
# ============================================================
SECTIONS = [('F', lambda: sec_f(SHARED['conn'])), ('G', lambda: sec_g(SHARED['conn'])),
            ('H1', lambda: sec_h1(SHARED['conn'])), ('H2', lambda: sec_h2(SHARED['conn'])),
            ('I', lambda: sec_i(SHARED['conn'])), ('J', lambda: sec_j(SHARED['conn'])),
            ('K', lambda: sec_k(SHARED['conn'])), ('L', sec_l), ('M', lambda: sec_m(SHARED['conn'])),
            ('N', sec_n), ('O', sec_o), ('P', sec_p), ('S', sec_s), ('T', lambda: sec_t(SHARED['conn']))]


def main():
    conn = connect(str(ROOT / mconfig.DB_PATH))
    SHARED['conn'] = conn
    order =[('A', sec_a), ('C', sec_c), ('D', sec_d), ('E', lambda: sec_e(conn)), ('B', sec_b),
             ('E2', lambda: sec_e2(conn))] + SECTIONS
    only = set(sys.argv[1:])
    for name, fn in order:
        if only and name not in only:
            continue
        t = time.time()
        fn()
        print(f'[{name}] {time.time() - t:.1f}s', flush=True)
    lines = ['% ビューとリスクのコツ_計算.py が作ったファイル（手で直さない）']
    old = {}
    vt = OUT / 'values.tex'
    if only and vt.exists():
        for line in vt.read_text(encoding='utf-8').splitlines()[1:]:
            key = line.split('v@', 1)[1].split('\\endcsname', 1)[0]
            old[key] = line
    for k, v in VALUES.items():
        old[k] = f'\\expandafter\\def\\csname v@{k}\\endcsname{{{v}}}'
    lines += list(old.values())
    vt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'values: {len(old)}')


if __name__ == '__main__':
    main()
