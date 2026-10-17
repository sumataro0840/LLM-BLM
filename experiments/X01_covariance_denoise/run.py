"""
X01: does Marcenko-Pastur denoising give a better covariance for the optimizer than the Ledoit-Wolf shrinkage
used now? (No LLM; S&P500 members, decision dates = month ends 2016-01 .. 2026-08.)

Fixed before the first run:
- Estimators (window 252 trading days, as BLM_COV_WINDOW_DAYS):
    sample   plain sample covariance (singular when stocks > days; pseudo-inverse where an inverse is needed;
             negative eigenvalues from rounding are set to 0 so that the solver accepts it)
    lw       Ledoit-Wolf shrinkage (blm.covariance.shrunk_covariance, what the BLM uses now)
    mp       Marcenko-Pastur constant-residual-eigenvalue denoising (blm.covariance.denoised_covariance)
- Universe of a date: index members with a market-cap weight and at least half of the window's prices;
  missing days filled with the cross-sectional mean (as blm.inputs.load_inputs).
- Portfolios built on each date and held to the next month end (daily returns; a stock that stops trading counts 0):
    gmv        unconstrained global minimum variance  w = S^-1 1 / 1' S^-1 1
    minvar     long-only minimum variance (Clarabel)
    active     the BLM's optimizer (blm.optimizer.optimize_with_tracking_error): expected returns
               = equilibrium (delta = 2.5) + Grinold alpha IC 0.05 * daily vol * normal score of 12-1 momentum,
               ex-ante tracking error at most 3 % a year against the market-cap weights
- Measures per date: realized volatility (annualized), predicted volatility, realized / predicted,
  turnover (sum |w_t - w_{t-1}|, drift ignored), largest weight, gross exposure (gmv), realized tracking error (active),
  condition number of the estimate.
- Comparison: mean over dates and the paired difference against lw with a Newey-West (lag 3) t value.
  Primary: realized volatility of minvar and realized tracking error of active (the two the BLM depends on).

    python experiments/X01_covariance_denoise/run.py
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
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blm.covariance import denoised_covariance, sample_covariance, shrunk_covariance
from blm.optimizer import optimize_with_tracking_error
from experiments.common import data as D
from marketdb.db.index import load_index_weights, load_members
from marketdb.db.market import last_trading_days, load_prices, load_trading_days

OUT = Path(__file__).resolve().parent / 'results'
WINDOW = 252
MOMENTUM_DAYS = (252, 21)          # 12-1 momentum: return from 252 to 21 trading days ago
DELTA = 2.5
IC = 0.05
TE = 0.03 / np.sqrt(252)
ESTIMATORS = {'sample': sample_covariance, 'lw': shrunk_covariance, 'mp': denoised_covariance}


def decision_dates(conn):
    days = pd.Series(pd.to_datetime(load_trading_days(conn, '2015-12-01', '2026-09-30')))
    ends = days.groupby(days.dt.to_period('M')).max()
    return [d.strftime('%Y-%m-%d') for d in ends if '2016-01' <= d.strftime('%Y-%m') <= '2026-08']


def inputs(conn, date, next_end):
    weights = load_index_weights(conn, date)
    members = [s for s in load_members(conn, date) if s in weights.index]
    days = last_trading_days(conn, date, MOMENTUM_DAYS[0] + 1)
    prices = load_prices(conn, members, days[0], next_end)
    past = prices[prices.index <= pd.Timestamp(date)]
    returns = past.pct_change().iloc[1:].iloc[-WINDOW:]
    enough = returns.notna().mean() >= 0.5
    universe = sorted(s for s in members if enough.get(s, False) and pd.notna(past[s].iloc[-1]))
    returns = returns[universe]
    returns = returns.apply(lambda c: c.fillna(returns.mean(axis=1)))
    mom = past[universe].iloc[-MOMENTUM_DAYS[1] - 1] / past[universe].iloc[0] - 1
    future = prices[universe][prices.index >= pd.Timestamp(date)].ffill().pct_change().iloc[1:].fillna(0.0)
    w_mkt = weights[universe] / weights[universe].sum()
    return returns, mom, future, w_mkt


def normal_scores(x):
    r = x.rank(method='average')
    return pd.Series(norm.ppf((r - 0.375) / (x.notna().sum() + 0.25)), index=x.index).fillna(0.0)


def ann(x):
    return float(np.sqrt(252) * x)


def min_variance_long_only(cov):
    w = cp.Variable(len(cov))
    problem = cp.Problem(cp.Minimize(1e4 * cp.quad_form(w, cp.psd_wrap(cov.values))), [cp.sum(w) == 1, w >= 0])
    problem.solve(solver=cp.CLARABEL)
    v = np.clip(w.value, 0, None)
    return pd.Series(v / v.sum(), index=cov.index)


def psd(cov):
    """Symmetric, with rounding-level negative eigenvalues set to 0 (changes only the singular sample estimate)."""
    values, vectors = np.linalg.eigh((cov.values + cov.values.T) / 2)
    return pd.DataFrame(vectors @ np.diag(np.clip(values, 0, None)) @ vectors.T, index=cov.index, columns=cov.columns)


def one_date(args):
    date, next_end = args
    conn = D.connect()
    returns, mom, future, w_mkt = inputs(conn, date, next_end)
    alpha = IC * returns.std() * normal_scores(mom)
    out, weights = {}, {}
    for name, estimate in ESTIMATORS.items():
        raw = estimate(returns)
        cov = psd(raw)
        S = cov.values
        eig = np.linalg.eigvalsh(S)
        row = {'n_stocks': len(cov), 'condition': float(eig.max() / max(eig.min(), 1e-18))}
        if name == 'mp':
            row['n_signal'] = raw.attrs['n_signal']
        inv1 = np.linalg.pinv(S) @ np.ones(len(S))
        gmv = pd.Series(inv1 / inv1.sum(), index=cov.index)
        minvar = min_variance_long_only(cov)
        prior = DELTA * cov @ w_mkt
        active, status = optimize_with_tracking_error(prior + alpha, cov, DELTA, w_mkt, TE)
        for pname, w in [('gmv', gmv), ('minvar', minvar), ('active', active)]:
            realized = future[w.index] @ w
            row[f'{pname}_vol'] = ann(realized.std())
            row[f'{pname}_pred'] = ann(np.sqrt(w.values @ S @ w.values))
            row[f'{pname}_max_w'] = float(w.abs().max())
            weights[(name, pname)] = w
        row['gmv_gross'] = float(gmv.abs().sum())
        act = active - w_mkt
        row['active_te'] = ann((future[act.index] @ act).std())
        row['active_te_pred'] = ann(np.sqrt(act.values @ S @ act.values))
        row['active_status'] = status
        out[name] = row
    return date, out, weights


def newey_west_t(x, lag=3):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n, m = len(x), x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(s / n)


def main():
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    dates = decision_dates(conn)
    pairs = list(zip(dates[:-1], dates[1:]))
    print(f'{len(pairs)} decision dates', flush=True)
    rows, prev = [], {}
    with ProcessPoolExecutor(6) as pool:
        for i, (date, out, weights) in enumerate(pool.map(one_date, pairs), 1):
            for name, row in out.items():
                for pname in ('gmv', 'minvar', 'active'):
                    w = weights[(name, pname)]
                    old = prev.get((name, pname))
                    row[f'{pname}_turnover'] = (float(w.sub(old, fill_value=0).abs().sum()) if old is not None else np.nan)
                    prev[(name, pname)] = w
                rows.append({'date': date, 'estimator': name, **row})
            if i % 20 == 0:
                print(f'  {i}/{len(pairs)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'by_date.csv', index=False)

    measures = ['gmv_vol', 'gmv_pred', 'gmv_turnover', 'gmv_gross', 'gmv_max_w',
                'minvar_vol', 'minvar_pred', 'minvar_turnover', 'minvar_max_w',
                'active_te', 'active_te_pred', 'active_turnover', 'active_vol', 'condition']
    wide = df.pivot(index='date', columns='estimator')
    summary = {}
    for m in measures:
        summary[m] = {}
        for est in ESTIMATORS:
            entry = {'mean': float(wide[m][est].mean()), 'median': float(wide[m][est].median())}
            if est != 'lw':
                diff = wide[m][est] - wide[m]['lw']
                entry['diff_vs_lw'] = float(diff.mean())
                entry['t_vs_lw'] = float(newey_west_t(diff))
            summary[m][est] = entry
    for p in ('gmv', 'minvar'):
        for est in ESTIMATORS:
            summary[f'{p}_bias_ratio'] = summary.get(f'{p}_bias_ratio', {})
            summary[f'{p}_bias_ratio'][est] = float((wide[f'{p}_vol'][est] / wide[f'{p}_pred'][est]).mean())
    summary['active_te_ratio'] = {est: float((wide['active_te'][est] / wide['active_te_pred'][est]).mean()) for est in ESTIMATORS}
    summary['mp_n_signal'] = {'mean': float(wide['n_signal']['mp'].mean()), 'min': int(wide['n_signal']['mp'].min()),
                              'max': int(wide['n_signal']['mp'].max())}
    summary['n_dates'] = len(wide)
    summary['n_stocks_mean'] = float(wide['n_stocks']['lw'].mean())
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
