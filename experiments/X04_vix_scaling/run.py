"""
X04: does the option market's forecast of next month's risk (VIX) help the BLM keep its tracking-error promise?
X03 brought the realized TE to about 3 % by scaling the limit with the past realized / ex-ante ratio, but months
in which volatility jumps (2020-02 .. 04, 2020-10) still missed by far. The VIX is the market's forecast of the
S&P500's volatility over the next 30 days, read from option prices.
(No LLM; S&P500 members; the same dates, universe and optimizer inputs as X02 and X03.)

Fixed before the first run:
- Estimators: lw252 and ewma126 (as X03).
- VIX scaling on each date t: the whole covariance S is multiplied by k_t = (VIX_t / 100)^2 / (252 w_mkt' S w_mkt),
  so that the market-cap portfolio's predicted annual volatility equals the VIX at the close of t.
  (The VIX is usually above the realized volatility, the variance risk premium; the calibration below absorbs
  the average level, the scaling supplies the changes.)
- Variants, each on the BLM optimizer with the scaled covariance:
    vix        limit 3 %, all 115 dates (2017-01 .. 2026-07)
    vix_cal    limit 3 % / (mean realized / ex-ante TE of 'vix' over the previous 36 dates), X03's rule;
               from the 37th date (2020-01 .. 2026-07, 79 months)
- Primary: on the 79 months, mean squared miss (realized TE - 3 %)^2 of vix_cal against X03's calibrated run
  (same estimator), Newey-West (lag 3) t. Also: vix against X03's uncalibrated run on the same months; RMSE of the
  miss, share of months above 3 %, the four worst months of X03, realized active return and information ratio.

    python experiments/X04_vix_scaling/run.py     # needs X02's and X03's results/by_date.csv
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
from marketdb.db.market import load_prices

OUT = Path(__file__).resolve().parent / 'results'
X02_DIR = REPO / 'experiments' / 'X02_covariance_window'
X03_DIR = REPO / 'experiments' / 'X03_te_calibration'
_spec = importlib.util.spec_from_file_location('x02_run', X02_DIR / 'run.py')
x02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x02)

ESTIMATORS = ['lw252', 'ewma126']
TARGET = 0.03
LOOKBACK = 36


def vix_on(conn, date):
    p = load_prices(conn, ['^VIX'], date, date, column='close')
    return float(p['^VIX'].iloc[0])


def scale(cov, w_mkt, vix):
    """Covariance multiplied so that the market-cap portfolio's predicted annual volatility equals the VIX."""
    var = float(w_mkt.values @ cov.values @ w_mkt.values) * 252
    k = (vix / 100) ** 2 / var
    return cov * k, k


def one_date(args):
    date, next_end, limits = args               # limits: {estimator: annual ex-ante limit}
    conn = D.connect()
    returns, mom, future, w_mkt = x02.inputs(conn, date, next_end)
    alpha = x02.IC * returns.iloc[-252:].std() * x02.normal_scores(mom)
    vix = vix_on(conn, date)
    out = {}
    for name in ESTIMATORS:
        cov, k = scale(x02.estimate(name, returns), w_mkt, vix)
        active, status = optimize_with_tracking_error(x02.DELTA * cov @ w_mkt + alpha, cov, x02.DELTA, w_mkt,
                                                      limits[name] / np.sqrt(252))
        act = active - w_mkt
        daily = future[act.index] @ act
        out[name] = {'te': float(np.sqrt(252) * daily.std()),
                     'te_pred': float(np.sqrt(252 * act.values @ cov.values @ act.values)),
                     'active_return': float((1 + daily).prod() - 1), 'limit': float(limits[name]),
                     'vix': vix, 'k': float(k), 'status': status}
    return date, out


def run(jobs):
    rows = []
    with ProcessPoolExecutor(5) as pool:          # leave cores for the ChronoGPT job running on the GPU
        for date, out in pool.map(one_date, jobs):
            rows += [{'date': date, 'estimator': n, **r} for n, r in out.items()]
    return pd.DataFrame(rows)


def ir(r):
    return float(r.mean() * 12 / (r.std() * np.sqrt(12)))


def main():
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    dates = x02.decision_dates(conn)
    pairs = list(zip(dates[:-1], dates[1:]))
    print(f'pass 1 (vix, 3 %): {len(pairs)} dates', flush=True)
    vix = run([(d, n, {e: TARGET for e in ESTIMATORS}) for d, n in pairs])
    vix.to_csv(OUT / 'by_date_vix.csv', index=False)

    ratio = (vix.assign(r=vix.te / vix.te_pred).pivot(index='date', columns='estimator', values='r')[ESTIMATORS]
             .sort_index())
    past = ratio.rolling(LOOKBACK).mean().shift(1).dropna()
    nxt = dict(pairs)
    print(f'pass 2 (vix_cal): {len(past)} dates from {past.index[0]}', flush=True)
    cal = run([(d, nxt[d], {e: TARGET / past.loc[d, e] for e in ESTIMATORS}) for d in past.index])
    cal.to_csv(OUT / 'by_date_vix_cal.csv', index=False)

    x03 = pd.read_csv(X03_DIR / 'results' / 'by_date.csv')
    summary = {'n_dates': int(cal.date.nunique()), 'first_date': cal.date.min(), 'target': TARGET}
    for name in ESTIMATORS:
        a = x03[x03.estimator == name].set_index('date')
        c = cal[cal.estimator == name].set_index('date')
        v = vix[vix.estimator == name].set_index('date').loc[c.index]
        a = a.loc[c.index]
        runs = {'x03_uncal': (a.te_uncal, a.active_return_uncal), 'x03_cal': (a.te, a.active_return),
                'vix': (v.te, v.active_return), 'vix_cal': (c.te, c.active_return)}
        miss = {k: (te - TARGET) ** 2 for k, (te, _) in runs.items()}
        d_main = miss['vix_cal'] - miss['x03_cal']
        d_raw = miss['vix'] - miss['x03_uncal']
        worst = a.te.sort_values(ascending=False).index[:4]
        summary[name] = {
            'mean_te': {k: float(te.mean()) for k, (te, _) in runs.items()},
            'median_te': {k: float(te.median()) for k, (te, _) in runs.items()},
            'rmse_miss': {k: float(np.sqrt(m.mean())) for k, m in miss.items()},
            'share_above_target': {k: float((te > TARGET).mean()) for k, (te, _) in runs.items()},
            'primary_vix_cal_vs_x03_cal': {'diff': float(d_main.mean()), 't': float(x02.newey_west_t(d_main)),
                                           'share_better': float((d_main < 0).mean())},
            'vix_vs_x03_uncal': {'diff': float(d_raw.mean()), 't': float(x02.newey_west_t(d_raw))},
            'worst_x03_months': {d: {k: float(te.loc[d]) for k, (te, _) in runs.items()} for d in worst},
            'active_return_annual': {k: float(r.mean() * 12) for k, (_, r) in runs.items()},
            'information_ratio': {k: ir(r) for k, (_, r) in runs.items()},
            'k': {'mean': float(v.k.mean()), 'min': float(v.k.min()), 'max': float(v.k.max())},
            'vix_cal_limit': {'mean': float(c.limit.mean()), 'min': float(c.limit.min()), 'max': float(c.limit.max())},
            'status': {k: df[df.estimator == name].status.value_counts().to_dict()
                       for k, df in (('vix', vix), ('vix_cal', cal))}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
