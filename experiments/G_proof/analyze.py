"""
G1, G2 and G5 (first half) of the proof plan (plan.md), from the daily series written by run.py.

    python experiments/G_proof/analyze.py        -> results/summary.json
"""
import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r1_run', REPO / 'experiments' / 'R1_replication' / 'run.py')
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)

EVAL = '2017-01-01'
COSTS = (0.0, 0.003, 0.005, 0.010)
MAIN_COST = 0.003
NW_LAG = 21
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
EXPLORE = {'orig': 0.030379, 'rep': 0.028596}      # R1 exploration, main threshold, B_cal, before costs (rounded)


def net(df, c):
    return df.active - c * (df.traded - df.traded_band)


def factors():
    c = sqlite3.connect(f'file:{REPO / "data" / "blm_2015.sqlite"}?mode=ro', uri=True)
    f = pd.read_sql("SELECT date, factor, value FROM factor_returns WHERE freq = 'daily' AND dataset IN "
                    "('F-F_Research_Data_5_Factors_2x3_daily', 'F-F_Momentum_Factor_daily')", c)
    return f.pivot(index='date', columns='factor', values='value')[FACTORS]


def ols_nw(y, X, lag=NW_LAG):
    """OLS with a constant; Newey-West (Bartlett) standard errors."""
    X = np.column_stack([np.ones(len(y)), X])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    xe = X * e[:, None]
    s = xe.T @ xe
    for k in range(1, lag + 1):
        g = xe[k:].T @ xe[:-k]
        s += (1 - k / (lag + 1)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2


def one_sided_p(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def g2(y, fac):
    d = pd.concat([y.rename('y'), fac], axis=1, join='inner').dropna()
    beta, se, r2 = ols_nw(d.y.to_numpy(), d[FACTORS].to_numpy())
    t = beta / se
    return {'days': int(len(d)), 'first': d.index.min(), 'last': d.index.max(),
            'mean_annual_same_days': float(d.y.mean() * 252),
            'alpha_annual': float(beta[0] * 252), 'alpha_t': float(t[0]), 'alpha_p_one_sided': one_sided_p(t[0]),
            'r2': float(r2), 'loadings': {k: {'beta': float(beta[i + 1]), 't': float(t[i + 1])} for i, k in enumerate(FACTORS)}}


def sample(name, fac):
    d = HERE / 'results' / name
    real = pd.read_csv(d / 'real.csv')
    real = real[real.date >= EVAL].reset_index(drop=True)
    sh = pd.concat([pd.read_csv(f) for f in sorted(d.glob('shuffle_*.csv'))])
    sh = sh[sh.date >= EVAL]
    seeds = sorted(sh.seed.unique())
    out = {'days': int(len(real)), 'n_shuffles': len(seeds),
           'check_explore_before_costs': {'here': float(real.active.mean() * 252), 'R1_exploration': EXPLORE[name]}}
    # G5: costs
    reb = real[real.traded > 0]
    out['G5_costs'] = {'mean_traded_blm': float(reb.traded.iloc[1:].mean()), 'mean_traded_band': float(reb.traded_band.iloc[1:].mean()),
                       'rebalances': int(len(reb))}
    for c in COSTS:
        out['G5_costs'][f'cost_{c}'] = r1.mean_test(net(real, c).to_numpy())
    # G1: shuffled views
    g1 = {}
    for c in (0.0, MAIN_COST):
        real_ann = float(net(real, c).mean() * 252)
        fake = sh.assign(n=net(sh, c)).groupby('seed').n.mean() * 252
        n_ge = int((fake >= real_ann).sum())
        g1[f'cost_{c}'] = {'real_annual': real_ann, 'fake_mean': float(fake.mean()), 'fake_sd': float(fake.std()),
                           'fake_q05': float(fake.quantile(0.05)), 'fake_q50': float(fake.quantile(0.5)),
                           'fake_q95': float(fake.quantile(0.95)), 'fake_ge_real': n_ge,
                           'p': (1 + n_ge) / (1 + len(fake)), 'real_minus_fake_mean': float(real_ann - fake.mean()),
                           'fake_ir_mean': float((sh.assign(n=net(sh, c)).groupby('seed').n.agg(
                               lambda s: s.mean() / s.std()) * np.sqrt(252)).mean())}
    g1['real_ir_cost_0.003'] = float(net(real, MAIN_COST).mean() / net(real, MAIN_COST).std() * np.sqrt(252))
    out['G1_shuffled_views'] = g1
    # G2: factors
    y = net(real, MAIN_COST).set_axis(real.date)
    fake_mean = sh.assign(n=net(sh, MAIN_COST)).groupby('date').n.mean()
    out['G2_factors'] = {'real': g2(y, fac), 'real_minus_fake_mean': g2(y - fake_mean.reindex(y.index), fac),
                         'fake_mean': g2(fake_mean, fac)}
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    fac = factors()
    res = {'note': 'exploration (plan.md): G1 shuffled views, G2 factor regression, G5 costs', 'main_cost': MAIN_COST,
           'nw_lag': NW_LAG}
    for name in ('orig', 'rep'):
        if (HERE / 'results' / name / 'real.csv').exists():
            res[name] = sample(name, fac)
    (HERE / 'results' / 'summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                     encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
