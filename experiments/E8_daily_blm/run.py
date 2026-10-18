"""
E8: is any of the LLM's information still unpriced at the close of the event day (day 0), so that a portfolio
trading at that close can earn it? Everything follows preregistration.md. Run once, after the pre-registration
commit:

    python experiments/E8_daily_blm/run.py

Part 1 (the verdict): the event regression of the next day's abnormal return (close of day 0 to close of day 1)
on the LLM score, with E3's controls and the day-0 reaction. Part 2 (reported, not the verdict): a daily BLM that
puts views only on the stocks whose day 0 is today, trades at that close and holds one day.
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
from events import config
from events.evaluate import holm, permutation_test, regress
from events.returns import load_closes, reactions, sector_benchmarks
from experiments.common import data as D
from marketdb.db.connection import connect
from marketdb.db.market import load_trading_days

OUT = Path(__file__).resolve().parent / 'results'
N_PERMUTATIONS = 1000
NEXT_WINDOWS = {'r_1_1': (1, 1), 'drift_1_5': (1, 5)}
IC = 0.05
TAU, DELTA = 0.025, 2.5
TE = 0.03
COST = 0.001
MIN_PAST = 500                 # the daily strategy starts once 500 scored events lie in the past


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2 = _load('e2_run', 'experiments/E2_event_reaction/run.py')
e3 = _load('e3_run', 'experiments/E3_analyst_surprise/run.py')
x02 = _load('x02_run', 'experiments/X02_covariance_window/run.py')


# ---------- part 1: the event regression ----------

def next_day_reactions(conn, A):
    days = load_trading_days(conn, '2023-01-01', '2026-12-31')
    bench = sector_benchmarks(conn)
    symbols = sorted(set(A.symbol) | {b for b in bench.values() if b} | {'^GSPC'})
    closes = load_closes(conn, symbols, days[0], days[-1])
    return reactions(A[['accession', 'symbol', 'event_date']], closes, days, bench, NEXT_WINDOWS)


def fit(d, focal, controls, permute=False):
    d = d.dropna(subset=['y', focal])
    controls = [c for c in controls if d[c].std() > 0]
    full = regress(d, 'y', [focal, *controls], e2.FIXED)
    row = full.loc[focal]
    out = {'coef_bp_per_sd': float(row.coef * 1e4), 'se_bp': float(row.se * 1e4), 't': float(row.t),
           'p_one_sided': float(e2.one_sided(row)), 'n': int(row.n), 'event_dates': int(row.clusters),
           'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4), float((row.coef + 1.96 * row.se) * 1e4)]}
    if 'day0' in full.index and focal != 'day0':
        out['day0'] = {'coef_bp_per_sd': float(full.loc['day0'].coef * 1e4), 't': float(full.loc['day0'].t)}
    if permute:
        out['permutation_p_one_sided'] = float(permutation_test(d, 'y', focal, controls, e2.FIXED,
                                                                n=N_PERMUTATIONS, seed=0)['p_one_sided'])
    return out


def regression_part(conn):
    A = e3.sample_a(conn)
    nd = next_day_reactions(conn, A)
    A = A.merge(nd[['accession', 'abn_r_1_1', 'abn_drift_1_5']], on='accession', how='left')
    out = {}
    for name, y in (('P', 'abn_r_1_1'), ('S1', 'abn_drift_1_5'), ('S2', 'abn_r_1_1')):
        d = e3.prepare(A, 'llm_composite', y)
        d['day0'] = e2.standardize(e2.winsorize(A['abn_r_0_0']))
        controls = e3.M1 + ([] if name == 'S2' else ['day0'])
        out[name] = fit(d, 'llm_composite', controls, permute=(name == 'P'))
    p = out['P']
    p['verdict'] = '支持' if p['p_one_sided'] < 0.05 and p['permutation_p_one_sided'] < 0.05 else '支持されない'
    for r, ph in zip((out['S1'], out['S2']), holm([out['S1']['p_one_sided'], out['S2']['p_one_sided']])):
        r['p_holm'] = float(ph)
    out['n_with_next_day'] = int(A.abn_r_1_1.notna().sum())
    return out, A


# ---------- part 2: the daily BLM ----------

def standardized_scores(events, day):
    """Scores of the events whose day 0 is `day`, standardized with the mean and sd of all scored events whose
    day 0 was earlier (none of today's or later). None before MIN_PAST past events."""
    past = events[events.event_date < day].llm_composite
    if len(past) < MIN_PAST:
        return None
    today = events[events.event_date == day].drop_duplicates('symbol').set_index('symbol').llm_composite
    return (today - past.mean()) / past.std()


def month_inputs(conn, month_end, next_end):
    """Covariance (LW, 252 days), daily sigma and cap weights at a month end, and the daily returns to the next."""
    returns, _, future, w_mkt = x02.inputs(conn, month_end, next_end)
    r252 = returns.iloc[-252:]
    return x02.lw(r252), r252.std(), w_mkt, future


def daily_weights(cov, sigma, bench, z):
    prior = DELTA * cov @ bench
    if z is None or len(z) == 0:
        return bench.copy(), 'no views'
    names = list(z.index)
    pick = pd.DataFrame(np.eye(len(cov))[[cov.index.get_loc(s) for s in names]], columns=cov.index)
    q = pd.Series(prior[names].values + IC * sigma[names].values * z.values)
    omega = TAU * np.diag(np.diag(pick.values @ cov.values @ pick.values.T))
    mu = posterior_returns_general(prior, cov, pick, q, omega, TAU)
    return optimize_with_tracking_error(mu, cov, DELTA, bench, TE / np.sqrt(252))


def one_month(args):
    """Daily decisions from the close of a month end to the close before the next month end (that close is the
    next month's first decision): the benchmark is the month-end cap weights moved by the returns. The month's first
    trade is counted from the benchmark (the previous month's last holdings are in another process)."""
    month_end, next_end, events, seed = args
    conn = D.connect()
    cov, sigma, w_mkt, future = month_inputs(conn, month_end, next_end)
    rows, bench, held = [], w_mkt.copy(), None
    days = [month_end] + list(future.index.strftime('%Y-%m-%d'))
    for k, day in enumerate(days[:-1]):             # k: position of `day` in days; future.iloc[k] is the next day
        if k > 0:
            r_today = future.iloc[k - 1]
            bench = bench * (1 + r_today)
            bench = bench / bench.sum()                 # cap weights at today's close
            held = held * (1 + r_today)
            held = held / held.sum()
        z = standardized_scores(events, day)
        if z is not None:
            z = z[z.index.isin(cov.index)]
            if seed is not None and len(z) > 1:
                rng = np.random.default_rng([seed, int(day.replace('-', ''))])
                z = pd.Series(rng.permutation(z.values), index=z.index)
        w, status = daily_weights(cov, sigma, bench, z)
        before = bench if held is None else held
        traded = float((w - before).abs().sum())
        r_next = future.iloc[k]
        gross = float((w * r_next).sum())
        rows.append({'day': day, 'n_views': 0 if z is None else len(z), 'gross': gross,
                     'net': gross - COST * traded, 'bench': float((bench * r_next).sum()),
                     'turnover': traded / 2, 'status': status, 'started': z is not None})
        held = w
    return rows


def strategy_part(conn, events, seed=None):
    dates = x02.decision_dates(conn)
    nxt = dict(zip(dates[:-1], dates[1:]))
    months = [d for d in dates[:-1] if '2023-10' <= d[:7] <= '2026-07']
    with ProcessPoolExecutor(5) as pool:
        rows = [r for rs in pool.map(one_month, [(m, nxt[m], events, seed) for m in months]) for r in rs]
    df = pd.DataFrame(rows).set_index('day')
    df = df[df.started]
    df['active_net'] = df.net - df.bench
    df['active_gross'] = df.gross - df.bench
    return df


def summarize(df):
    a = df.active_net
    t = x02.newey_west_t(a, lag=5)
    return {'days': len(df), 'first': df.index[0], 'last': df.index[-1],
            'mean_active_net_daily_bp': float(a.mean() * 1e4), 'nw_t': float(t),
            'mean_active_gross_daily_bp': float(df.active_gross.mean() * 1e4),
            'annual_active_net': float(a.mean() * 252), 'annual_active_gross': float(df.active_gross.mean() * 252),
            'information_ratio_net': float(a.mean() / a.std() * np.sqrt(252)),
            'realized_te': float(a.std() * np.sqrt(252)), 'turnover_daily_mean': float(df.turnover.mean()),
            'days_with_views': int((df.n_views > 0).sum()), 'status': df.status.value_counts().to_dict()}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    conn = connect(config.DB_PATH)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    reg, A = regression_part(conn)
    print(json.dumps(reg, indent=1, ensure_ascii=False), flush=True)
    events = A[['symbol', 'event_date', 'llm_composite']].dropna()
    llm = strategy_part(D.connect(), events)
    llm.to_csv(OUT / 'by_day.csv')
    perm = strategy_part(D.connect(), events, seed=0)
    diff = llm.active_gross - perm.active_gross.reindex(llm.index)
    summary = {'code_commit': commit, 'part1_regression': reg,
               'part2_daily_blm': summarize(llm), 'part2_permuted_control': summarize(perm),
               'part2_llm_minus_permuted_gross_daily_bp': {'mean': float(diff.mean() * 1e4),
                                                          'nw_t': float(x02.newey_west_t(diff.dropna(), lag=5))}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'part1_regression'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
