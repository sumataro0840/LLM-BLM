"""
X02: which estimation window (or exponential weighting) of the covariance predicts next month's risk best?
X01 found that the realized tracking error is about 1.5 times the ex-ante limit with the 252-day window the BLM uses.
(No LLM; S&P500 members, decision dates = month ends 2017-01 .. 2026-08, so that 504 trading days of history exist.)

Fixed before the first run:
- Estimators, all Ledoit-Wolf shrinkage (as the BLM) on daily returns:
    lw63, lw126, lw252 (now), lw504   the last N trading days, equal weights
    ewma63, ewma126                   the last 504 days with exponential weights of half-life 63 / 126 days
                                      (LW on the weighted, re-centred returns: row t scaled by sqrt(T w_t))
- Universe of a date: as X01 (index members with a weight and at least half of the last 252 days' prices);
  missing days filled with the cross-sectional mean, for every window.
- Portfolios held to the next month end (daily returns, a stock that stops trading counts 0):
    fixed (the same weights for every estimator, so that only the prediction differs):
        mkt      market-cap weights
        ew       equal weights
        act      active weights of the BLM optimizer built with lw252 (X01's 'active': equilibrium + IC 0.05
                 momentum alpha, ex-ante TE at most 3 %) minus the market-cap weights
    built with each estimator:
        minvar   long-only minimum variance
        active   the BLM optimizer with that estimator (realized TE against its 3 % target)
- Measures per date and portfolio: predicted daily variance h = w' S w, realized daily variance RV = mean of
  squared daily returns over the holding month; QLIKE = RV/h - log(RV/h) - 1 (0 when exact), and the bias
  ratio sqrt(RV / h).
- Primary: (1) QLIKE of the fixed 'act' portfolio (the risk the BLM's TE limit relies on),
            (2) realized annualized TE of 'active' built with each estimator.
  Each alternative against lw252: mean paired difference with a Newey-West (lag 3) t value.
- Selection with past data only: on each date pick the estimator with the lowest mean QLIKE of 'act' over the
  previous 36 months whose holding period has ended (months before the date); from the 37th date on.
  Compare its QLIKE and the realized TE of its 'active' with lw252 over the same months.

    python experiments/X02_covariance_window/run.py
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from blm.optimizer import optimize_with_tracking_error
from experiments.common import data as D
from marketdb.db.index import load_index_weights, load_members
from marketdb.db.market import last_trading_days, load_prices, load_trading_days

OUT = Path(__file__).resolve().parent / 'results'
HISTORY = 504
UNIVERSE_WINDOW = 252
MOMENTUM_DAYS = (252, 21)
DELTA = 2.5
IC = 0.05
TE = 0.03 / np.sqrt(252)
LOOKBACK = 36
ESTIMATORS = {'lw63': ('window', 63), 'lw126': ('window', 126), 'lw252': ('window', 252), 'lw504': ('window', 504),
              'ewma63': ('halflife', 63), 'ewma126': ('halflife', 126)}
BASE = 'lw252'


def lw(returns):
    est = LedoitWolf().fit(returns.values)
    return pd.DataFrame(est.covariance_, index=returns.columns, columns=returns.columns)


def ewma_lw(returns, halflife):
    """LW shrinkage of the exponentially weighted covariance: the weighted sample covariance is X'X / T of the
    rows x_t = sqrt(T w_t) (r_t - weighted mean); LW then picks the shrinkage from those rows."""
    T = len(returns)
    w = 0.5 ** (np.arange(T)[::-1] / halflife)
    w = w / w.sum()
    r = returns.values
    x = (r - w @ r) * np.sqrt(T * w)[:, None]
    est = LedoitWolf(assume_centered=True).fit(x)
    return pd.DataFrame(est.covariance_, index=returns.columns, columns=returns.columns)


def estimate(name, returns):
    kind, n = ESTIMATORS[name]
    return lw(returns.iloc[-n:]) if kind == 'window' else ewma_lw(returns, n)


def decision_dates(conn):
    days = pd.Series(pd.to_datetime(load_trading_days(conn, '2016-12-01', '2026-09-30')))
    ends = days.groupby(days.dt.to_period('M')).max()
    return [d.strftime('%Y-%m-%d') for d in ends if '2017-01' <= d.strftime('%Y-%m') <= '2026-08']


def inputs(conn, date, next_end):
    weights = load_index_weights(conn, date)
    members = [s for s in load_members(conn, date) if s in weights.index]
    days = last_trading_days(conn, date, HISTORY + 1)
    prices = load_prices(conn, members, days[0], next_end)
    past = prices[prices.index <= pd.Timestamp(date)]
    returns = past.pct_change().iloc[1:].iloc[-HISTORY:]
    enough = returns.iloc[-UNIVERSE_WINDOW:].notna().mean() >= 0.5
    universe = sorted(s for s in members if enough.get(s, False) and pd.notna(past[s].iloc[-1]))
    returns = returns[universe]
    returns = returns.apply(lambda c: c.fillna(returns.mean(axis=1)))
    mom_start = past[universe].iloc[-MOMENTUM_DAYS[0] - 1]
    mom = past[universe].iloc[-MOMENTUM_DAYS[1] - 1] / mom_start - 1
    future = prices[universe][prices.index >= pd.Timestamp(date)].ffill().pct_change().iloc[1:].fillna(0.0)
    w_mkt = weights[universe] / weights[universe].sum()
    return returns, mom, future, w_mkt


def normal_scores(x):
    from scipy.stats import norm
    r = x.rank(method='average')
    return pd.Series(norm.ppf((r - 0.375) / (x.notna().sum() + 0.25)), index=x.index).fillna(0.0)


def min_variance_long_only(cov):
    w = cp.Variable(len(cov))
    problem = cp.Problem(cp.Minimize(1e4 * cp.quad_form(w, cp.psd_wrap(cov.values))), [cp.sum(w) == 1, w >= 0])
    problem.solve(solver=cp.CLARABEL)
    v = np.clip(w.value, 0, None)
    return pd.Series(v / v.sum(), index=cov.index)


def qlike(rv, h):
    q = rv / h
    return float(q - np.log(q) - 1)


def one_date(args):
    date, next_end = args
    conn = D.connect()
    returns, mom, future, w_mkt = inputs(conn, date, next_end)
    alpha = IC * returns.iloc[-252:].std() * normal_scores(mom)
    covs = {name: estimate(name, returns) for name in ESTIMATORS}
    base = covs[BASE]
    active_base, _ = optimize_with_tracking_error(DELTA * base @ w_mkt + alpha, base, DELTA, w_mkt, TE)
    fixed = {'mkt': w_mkt, 'ew': pd.Series(1 / len(w_mkt), index=w_mkt.index), 'act': active_base - w_mkt}
    out = {}
    for name, cov in covs.items():
        S = cov.values
        row = {'n_stocks': len(cov)}
        for p, w in fixed.items():
            h = float(w.values @ S @ w.values)
            rv = float(((future[w.index] @ w) ** 2).mean())
            row[f'{p}_qlike'] = qlike(rv, h)
            row[f'{p}_bias'] = float(np.sqrt(rv / h))
        mv = min_variance_long_only(cov)
        row['minvar_vol'] = float(np.sqrt(252) * (future[mv.index] @ mv).std())
        row['minvar_pred'] = float(np.sqrt(252 * mv.values @ S @ mv.values))
        active, status = optimize_with_tracking_error(DELTA * cov @ w_mkt + alpha, cov, DELTA, w_mkt, TE)
        act = active - w_mkt
        row['active_te'] = float(np.sqrt(252) * (future[act.index] @ act).std())
        row['active_te_pred'] = float(np.sqrt(252 * act.values @ S @ act.values))
        row['active_status'] = status
        out[name] = row
    return date, out


def newey_west_t(x, lag=3):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n, m = len(x), x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(s / n)


def compare(wide, measure):
    out = {}
    for est in ESTIMATORS:
        entry = {'mean': float(wide[measure][est].mean()), 'median': float(wide[measure][est].median())}
        if est != BASE:
            diff = wide[measure][est] - wide[measure][BASE]
            entry.update(diff_vs_base=float(diff.mean()), t_vs_base=float(newey_west_t(diff)),
                         share_better=float((diff < 0).mean()))
        out[est] = entry
    return out


def past_only_selection(wide):
    """Estimator chosen on each date from the mean 'act' QLIKE of the previous LOOKBACK dates (all ended)."""
    q = wide['act_qlike']
    rows = []
    for i in range(LOOKBACK, len(q)):
        pick = q.iloc[i - LOOKBACK:i].mean().idxmin()
        date = q.index[i]
        rows.append({'date': date, 'pick': pick, 'act_qlike': q.iloc[i][pick], 'base_act_qlike': q.iloc[i][BASE],
                     'active_te': wide['active_te'].iloc[i][pick], 'base_active_te': wide['active_te'].iloc[i][BASE]})
    sel = pd.DataFrame(rows)
    d_q = sel.act_qlike - sel.base_act_qlike
    d_te = sel.active_te - sel.base_active_te
    return sel, {'n_dates': len(sel), 'picks': sel.pick.value_counts().to_dict(),
                 'act_qlike': float(sel.act_qlike.mean()), 'base_act_qlike': float(sel.base_act_qlike.mean()),
                 'qlike_diff': float(d_q.mean()), 'qlike_t': float(newey_west_t(d_q)),
                 'active_te': float(sel.active_te.mean()), 'base_active_te': float(sel.base_active_te.mean()),
                 'te_diff': float(d_te.mean()), 'te_t': float(newey_west_t(d_te))}


def main():
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    dates = decision_dates(conn)
    pairs = list(zip(dates[:-1], dates[1:]))
    print(f'{len(pairs)} decision dates', flush=True)
    rows = []
    with ProcessPoolExecutor(5) as pool:          # leave cores for the ChronoGPT job running on the GPU
        for i, (date, out) in enumerate(pool.map(one_date, pairs), 1):
            rows += [{'date': date, 'estimator': name, **row} for name, row in out.items()]
            if i % 20 == 0:
                print(f'  {i}/{len(pairs)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'by_date.csv', index=False)
    wide = df.pivot(index='date', columns='estimator')

    summary = {'n_dates': len(wide), 'n_stocks_mean': float(wide['n_stocks'][BASE].mean()), 'base': BASE}
    for m in ['act_qlike', 'active_te', 'mkt_qlike', 'ew_qlike', 'act_bias', 'mkt_bias', 'ew_bias',
              'minvar_vol', 'minvar_pred', 'active_te_pred']:
        summary[m] = compare(wide, m)
    summary['active_te_ratio'] = {e: float((wide['active_te'][e] / wide['active_te_pred'][e]).mean()) for e in ESTIMATORS}
    summary['active_status'] = {e: wide['active_status'][e].value_counts().to_dict() for e in ESTIMATORS}
    sel, summary['past_only_selection'] = past_only_selection(wide)
    sel.to_csv(OUT / 'selection.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
