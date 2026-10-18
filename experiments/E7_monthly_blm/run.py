"""
E7: the BLM built from everything learned so far, rebalanced monthly, with the LLM's reading of each stock's
latest earnings release as the views. Everything follows preregistration.md. Run once, after the pre-registration
commit:

    python experiments/E7_monthly_blm/run.py

Decision dates: month ends 2024-01 .. 2026-07, held to the next month end (gpt-4o-mini scores exist from 2023-11).
"""
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from blm.black_litterman import posterior_returns_general
from blm.optimizer import optimize_with_tracking_error
from experiments.common import data as D

OUT = Path(__file__).resolve().parent / 'results'
PANEL = REPO / 'experiments' / 'E2_event_reaction' / 'results' / 'panel.csv'
X02_DIR = REPO / 'experiments' / 'X02_covariance_window'
_spec = importlib.util.spec_from_file_location('x02_run', X02_DIR / 'run.py')
x02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x02)

FIRST, LAST = '2024-01', '2026-07'
VIEW_DAYS = 92                 # a release counts as a view for 92 calendar days after its event date
IC = 0.05
TAU, DELTA = 0.025, 2.5
TARGET = 0.03
LOOKBACK = 36
COST = 0.001                   # one way, of the traded amount
SEEDS = 20                     # permuted controls


def blom(values):
    r = stats.rankdata(values, method='average')
    return stats.norm.ppf((r - 0.375) / (len(r) + 0.25))


def latest_scores(panel, date, universe):
    """LLM composite of each stock's latest release whose event date is on or before the decision date
    (its reaction had started by that close) and at most VIEW_DAYS earlier."""
    lo = (pd.Timestamp(date) - pd.Timedelta(days=VIEW_DAYS)).strftime('%Y-%m-%d')
    p = panel[(panel.event_date <= date) & (panel.event_date > lo) & panel.symbol.isin(universe)]
    p = p.sort_values(['event_date', 'accepted_at']).drop_duplicates('symbol', keep='last')
    return p.set_index('symbol').llm_composite


def te_limits():
    """X03's rule on X02's lw252 months: 3 % / mean realized-to-ex-ante TE ratio of the previous 36 months."""
    df = pd.read_csv(X02_DIR / 'results' / 'by_date.csv')
    df = df[df.estimator == 'lw252'].set_index('date').sort_index()
    ratio = (df.active_te / df.active_te_pred).rolling(LOOKBACK).mean().shift(1)
    return TARGET / ratio


def blm_weights(cov, w_mkt, sigma_daily, z, limit_annual):
    """Black-Litterman with one absolute view per stock that has a score (q = prior + Grinold alpha
    IC * sigma * z), He-Litterman Omega = tau diag(P Sigma P'), then the TE-limited optimizer."""
    prior = DELTA * cov @ w_mkt
    names = z.index
    pick = pd.DataFrame(0.0, index=range(len(names)), columns=cov.index)
    for k, s in enumerate(names):
        pick.loc[k, s] = 1.0
    q = pd.Series(prior[names].values + IC * sigma_daily[names].values * z.values)
    omega = TAU * np.diag(np.diag(pick.values @ cov.values @ pick.values.T))
    mu = posterior_returns_general(prior, cov, pick, q, omega, TAU)
    w, status = optimize_with_tracking_error(mu, cov, DELTA, w_mkt, limit_annual / np.sqrt(252))
    return w, status


def one_date(args):
    date, next_end, limit, panel = args
    conn = D.connect()
    returns, _, future, w_mkt = x02.inputs(conn, date, next_end)
    r252 = returns.iloc[-252:]
    cov = x02.lw(r252)
    sigma = r252.std()
    scores = latest_scores(panel, date, set(cov.index))
    z = pd.Series(blom(scores.values), index=scores.index)
    month = (1 + future).prod() - 1                         # each stock's return to the next month end
    out = {'n_views': int(len(z)), 'n_stocks': int(len(cov)), 'limit': float(limit), 'month': month,
           'w_mkt': w_mkt, 'daily': future}
    w, status = blm_weights(cov, w_mkt, sigma, z, limit)
    out['llm'] = (w, status)
    for s in range(SEEDS):
        rng = np.random.default_rng([s, int(date.replace('-', ''))])
        zp = pd.Series(rng.permutation(z.values), index=z.index)
        out[f'perm{s}'] = blm_weights(cov, w_mkt, sigma, zp, limit)
    act = w - w_mkt
    out['te_pred'] = float(np.sqrt(252 * act.values @ cov.values @ act.values))
    return date, out


def backtest(results, key):
    """Monthly holding with drifted weights; cost on the traded amount (one way)."""
    rows, held = [], None
    for date, x in results:
        w = x[key][0]
        R = x['month'].reindex(w.index).fillna(0.0)
        before = x['w_mkt'] if held is None else held
        idx = w.index.union(before.index)
        traded = float((w.reindex(idx).fillna(0) - before.reindex(idx).fillna(0)).abs().sum())
        gross = float((w * R).sum())
        grown = w * (1 + R)
        held = grown / grown.sum()
        bench = float((x['w_mkt'] * x['month'].reindex(x['w_mkt'].index).fillna(0.0)).sum())
        act_daily = x['daily'][w.index] @ (w - x['w_mkt'].reindex(w.index).fillna(0))
        rows.append({'date': date, 'gross': gross, 'net': gross - COST * traded, 'bench': bench,
                     'turnover': traded / 2, 'te_realized': float(np.sqrt(252) * act_daily.std()),
                     'status': x[key][1]})
    df = pd.DataFrame(rows).set_index('date')
    df['active_net'] = df.net - df.bench
    df['active_gross'] = df.gross - df.bench
    return df


def summarize(df):
    a = df.active_net
    return {'mean_active_net_monthly': float(a.mean()), 'nw_t': float(x02.newey_west_t(a)),
            'p_one_sided': float(stats.t.sf(x02.newey_west_t(a), len(a) - 1)),
            'mean_active_gross_monthly': float(df.active_gross.mean()),
            'information_ratio_net': float(a.mean() * 12 / (a.std() * np.sqrt(12))),
            'cumulative_net': float((1 + df.net).prod() - 1), 'cumulative_bench': float((1 + df.bench).prod() - 1),
            'realized_te_mean': float(df.te_realized.mean()), 'turnover_mean': float(df.turnover.mean()),
            'share_months_ahead': float((a > 0).mean()), 'status': df.status.value_counts().to_dict()}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    panel = pd.read_csv(PANEL, usecols=['accession', 'symbol', 'event_date', 'accepted_at', 'llm_composite'])
    panel = panel[panel.llm_composite.notna()]
    dates = x02.decision_dates(conn)
    nxt = dict(zip(dates[:-1], dates[1:]))
    limits = te_limits()
    jobs = [(d, nxt[d], float(limits.loc[d]), panel) for d in dates[:-1] if FIRST <= d[:7] <= LAST]
    print(f'{len(jobs)} decision dates {jobs[0][0]} .. {jobs[-1][0]}', flush=True)
    with ProcessPoolExecutor(5) as pool:
        results = list(pool.map(one_date, jobs))
    llm = backtest(results, 'llm')
    perms = [backtest(results, f'perm{s}') for s in range(SEEDS)]
    llm.to_csv(OUT / 'by_month.csv')
    main_result = summarize(llm)
    main_result['verdict'] = '支持' if main_result['p_one_sided'] < 0.05 else '支持されない'
    perm_gross = pd.concat([p.active_gross for p in perms], axis=1).mean(axis=1)
    diff = llm.active_gross - perm_gross
    summary = {
        'code_commit': commit, 'n_months': len(llm), 'first': llm.index[0], 'last': llm.index[-1],
        'views_per_month': {'mean': float(np.mean([x['n_views'] for _, x in results])),
                            'min': int(min(x['n_views'] for _, x in results))},
        'stocks_per_month': float(np.mean([x['n_stocks'] for _, x in results])),
        'te_limit': {'mean': float(np.mean([x['limit'] for _, x in results]))},
        'te_pred_mean': float(np.mean([x['te_pred'] for _, x in results])),
        'primary_llm_blm_vs_market': main_result,
        'secondary_llm_vs_permuted': {'mean_diff_gross_monthly': float(diff.mean()), 'nw_t': float(x02.newey_west_t(diff)),
                                      'permuted_mean_active_net': float(np.mean([p.active_net.mean() for p in perms]))},
        'secondary_te': {'realized_mean': main_result['realized_te_mean'], 'target': TARGET},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
