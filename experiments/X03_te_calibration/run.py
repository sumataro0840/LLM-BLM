"""
X03: can the BLM keep its tracking-error promise if the ex-ante limit is scaled by the past ratio of realized to
ex-ante TE? X01/X02: the realized TE is 1.4-1.5 times the 3 % limit whatever the covariance window.
(No LLM; S&P500 members; the same dates, universe and optimizer inputs as X02.)

Fixed before the first run:
- Estimators: lw252 (now) and ewma126 (X02's best realized TE).
- Rule on each date t (from X02's 37th date, 2020-01 .. 2026-07, 79 months): ratio_t = mean of realized TE / ex-ante TE
  of the uncalibrated optimizer (X02's results/by_date.csv, same estimator) over the previous 36 dates, all of whose
  holding months ended before t; ex-ante limit = 3 % / ratio_t. Nothing after t is used.
- Compared on the same months: uncalibrated (3 %, X02) and calibrated, for each estimator.
- Measures per date: realized annualized TE, realized active return over the holding month.
- Primary: mean squared miss (realized TE - 3 %)^2, calibrated against uncalibrated, Newey-West (lag 3) t.
  Also reported: mean realized TE, share of months above 3 %, realized active return and information ratio
  (the cost of the smaller ex-ante risk; not used for the verdict).

    python experiments/X03_te_calibration/run.py      # needs experiments/X02_covariance_window/results/by_date.csv
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from blm.optimizer import optimize_with_tracking_error
from experiments.common import data as D

OUT = Path(__file__).resolve().parent / 'results'
X02_DIR = REPO / 'experiments' / 'X02_covariance_window'
_spec = importlib.util.spec_from_file_location('x02_run', X02_DIR / 'run.py')
x02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x02)

ESTIMATORS = ['lw252', 'ewma126']
TARGET = 0.03
LOOKBACK = 36


def limits():
    """Calibrated annual ex-ante limit per (date, estimator) from the uncalibrated months that ended before it."""
    df = pd.read_csv(X02_DIR / 'results' / 'by_date.csv')
    df['ratio'] = df.active_te / df.active_te_pred
    ratio = df.pivot(index='date', columns='estimator', values='ratio')[ESTIMATORS].sort_index()
    past = ratio.rolling(LOOKBACK).mean().shift(1)           # row t: mean over rows t-36 .. t-1
    base = df.pivot(index='date', columns='estimator', values='active_te')[ESTIMATORS].sort_index()
    return (TARGET / past).dropna(), past.dropna(), base


def one_date(args):
    date, next_end, lim = args
    conn = D.connect()
    returns, mom, future, w_mkt = x02.inputs(conn, date, next_end)
    alpha = x02.IC * returns.iloc[-252:].std() * x02.normal_scores(mom)
    out = {}
    for name in ESTIMATORS:
        cov = x02.estimate(name, returns)
        active, status = optimize_with_tracking_error(x02.DELTA * cov @ w_mkt + alpha, cov, x02.DELTA, w_mkt,
                                                      lim[name] / np.sqrt(252))
        act = active - w_mkt
        daily = future[act.index] @ act
        out[name] = {'te': float(np.sqrt(252) * daily.std()), 'active_return': float((1 + daily).prod() - 1),
                     'limit': float(lim[name]), 'status': status}
    # the uncalibrated active return on the same month (X02 stored only its TE)
    for name in ESTIMATORS:
        cov = x02.estimate(name, returns)
        active, _ = optimize_with_tracking_error(x02.DELTA * cov @ w_mkt + alpha, cov, x02.DELTA, w_mkt, x02.TE)
        act = active - w_mkt
        out[name]['active_return_uncal'] = float((1 + future[act.index] @ act).prod() - 1)
    return date, out


def main():
    OUT.mkdir(exist_ok=True)
    lim, past, base = limits()
    conn = D.connect()
    dates = x02.decision_dates(conn)
    nxt = dict(zip(dates[:-1], dates[1:]))
    jobs = [(d, nxt[d], lim.loc[d]) for d in lim.index]
    print(f'{len(jobs)} dates from {jobs[0][0]}', flush=True)
    rows = []
    with ProcessPoolExecutor(5) as pool:
        for date, out in pool.map(one_date, jobs):
            for name, r in out.items():
                rows.append({'date': date, 'estimator': name, 'te_uncal': float(base.loc[date, name]),
                             'ratio_past': float(past.loc[date, name]), **r})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'by_date.csv', index=False)
    summary = {'n_dates': int(df.date.nunique()), 'first_date': df.date.min(), 'target': TARGET}
    for name, g in df.groupby('estimator'):
        miss_c, miss_u = (g.te - TARGET) ** 2, (g.te_uncal - TARGET) ** 2
        d = miss_c - miss_u
        ir = lambda r: float(r.mean() * 12 / (r.std() * np.sqrt(12)))
        summary[name] = {
            'mean_te': {'calibrated': float(g.te.mean()), 'uncalibrated': float(g.te_uncal.mean())},
            'rmse_miss': {'calibrated': float(np.sqrt(miss_c.mean())), 'uncalibrated': float(np.sqrt(miss_u.mean()))},
            'miss_sq_diff': float(d.mean()), 'miss_sq_t': float(x02.newey_west_t(d)),
            'share_above_target': {'calibrated': float((g.te > TARGET).mean()), 'uncalibrated': float((g.te_uncal > TARGET).mean())},
            'limit': {'mean': float(g.limit.mean()), 'min': float(g.limit.min()), 'max': float(g.limit.max())},
            'active_return_annual': {'calibrated': float(g.active_return.mean() * 12),
                                     'uncalibrated': float(g.active_return_uncal.mean() * 12)},
            'information_ratio': {'calibrated': ir(g.active_return), 'uncalibrated': ir(g.active_return_uncal)},
            'status': g.status.value_counts().to_dict()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
