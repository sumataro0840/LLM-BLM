"""
H04: Omega - the answers' variance, He-Litterman, calibrated (diagonal / proportional / shrunk),
and an uninformative control (the forecasts shuffled across stocks, 200 times).

    python experiments/H04_omega_calibration/run.py [number of shuffles]

No API calls. Writes results/summary.json, results/performance.csv, results/monthly.csv, results/placebo.csv.
Decision rules: 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf, chapter H04.
"""
import os

# one thread per process: the shuffles run in 12 processes, and numpy / the solver would otherwise each start
# as many threads as there are cores (hundreds of threads fighting over 16 cores)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common.backtest import market_cap_strategy, run
from experiments.common.metrics import performance, t_summary
from experiments.common.strategies import bl_strategy

OUT = Path(__file__).resolve().parent / 'results'
N_SHUFFLES = 200
VARIANTS = {   # name -> (omega, shrink)
    'answer_variance': ('answer', False),
    'he_litterman': ('he_litterman', False),
    'calibrated_diagonal': ('calibrated_diagonal', False),
    'calibrated_proportional': ('proportional', False),
    'calibrated_proportional_shrunk': ('proportional', True),
}
MAIN = 'calibrated_proportional_shrunk'


def _placebo(seed):
    """One shuffled run (module level so that worker processes can run it)."""
    conn = D.connect()
    all_data = D.load_all(conn)
    panel = D.forecast_panel(all_data)
    bench = run(market_cap_strategy, all_data)
    res = run(bl_strategy(panel, form='relative', omega='proportional', shrink=True, permute_seed=seed), all_data)
    p = performance(res, bench)
    return {'seed': seed, **p}


def partial_rank_corr(x, y, z):
    """Spearman correlation of x and y after the ranks of z are regressed out of both."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    X = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    n_shuffles = int(sys.argv[1]) if len(sys.argv) > 1 else N_SHUFFLES
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    all_data = D.load_all(conn)
    panel = D.forecast_panel(all_data)
    bench = run(market_cap_strategy, all_data)
    bench_lag = run(market_cap_strategy, all_data, lagged=True)

    # (1) does the spread of the 30 answers say which forecasts will miss more, once volatility is known?
    rows = []
    for d, x in all_data.items():
        df = pd.DataFrame({'sd': np.sqrt(x['view_variance']), 'err': (x['view_mean'] - x['next_mean_daily'].reindex(x['universe'])).abs(),
                           'vol': x['vol_daily'], 'n': x['next_days'].reindex(x['universe'])}).dropna()
        df = df[df.n >= 15]
        rows.append({'date': d, 'raw': stats.spearmanr(df.sd, df.err)[0], 'partial': partial_rank_corr(df.sd, df.err, df.vol),
                     'sd_vs_vol': stats.spearmanr(df.sd, df.vol)[0]})
    spread = pd.DataFrame(rows).set_index('date')
    spread.to_csv(OUT / 'answer_spread_vs_error.csv')

    # (2) the Omega variants
    results, perf, monthly = {}, [], []
    for name, (omega, shrink) in VARIANTS.items():
        t0 = time.time()
        res = run(bl_strategy(panel, form='relative', omega=omega, shrink=shrink), all_data)
        results[name] = res
        perf.append({'name': name, **performance(res, bench), 'adoption_rel_median': float(res.adoption_rel.median()),
                     'ex_ante_te_mean': float(res.ex_ante_te.mean()), 'b_last': float(res.b.iloc[-1]),
                     'months_b_zero': int((res.b <= 0).sum())})
        monthly.append(res.drop(columns=['weights']).assign(name=name))
        print(f'{name}: {time.time() - t0:.0f}s', flush=True)
    # sensitivity: trade one day after the decision date
    lag = run(bl_strategy(panel, form='relative', omega='proportional', shrink=True), all_data, lagged=True)
    perf.append({'name': MAIN + '|next_day_trade', **performance(lag, bench_lag)})
    perf = pd.DataFrame(perf).set_index('name')
    perf.loc['market_cap'] = performance(bench, bench)
    perf.to_csv(OUT / 'performance.csv')
    pd.concat(monthly).to_csv(OUT / 'monthly.csv')

    # (3) the uninformative control
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for k, fut in enumerate(as_completed([pool.submit(_placebo, s) for s in range(n_shuffles)]), 1):
            rows.append(fut.result())
            if k % 10 == 0:
                print(f'placebo {k}/{n_shuffles} ({time.time() - t0:.0f}s)', flush=True)
    placebo = pd.DataFrame(rows).sort_values('seed')
    placebo.to_csv(OUT / 'placebo.csv', index=False)
    print(f'placebo: {time.time() - t0:.0f}s', flush=True)

    actual = perf.loc[MAIN, 'active_return']
    p_one = (1 + int((placebo.active_return >= actual).sum())) / (1 + len(placebo))
    main_res = results[MAIN]
    live = main_res[main_res.b > 0]
    behaviour = {
        'adoption_minus_b_max_abs': float((live.adoption_rel - live.b).abs().max()) if len(live) else None,
        'effective_n_mean': float(main_res.effective_n.mean()),
        'realized_te': float(perf.loc[MAIN, 'tracking_error']),
    }
    behaviour['supported'] = bool(behaviour['adoption_minus_b_max_abs'] is not None and behaviour['adoption_minus_b_max_abs'] <= 0.01
                                  and behaviour['effective_n_mean'] >= 30 and behaviour['realized_te'] <= 0.06)
    part = t_summary(spread.partial)
    summary = {
        'primary_placebo': {'actual_active_return': float(actual), 'placebo_mean': float(placebo.active_return.mean()),
                            'placebo_p95': float(placebo.active_return.quantile(0.95)), 'n_shuffles': len(placebo),
                            'p_one_sided': p_one, 'p_two_sided_for_holm': min(1.0, 2 * p_one),
                            'verdict': 'supported (no information of the views detected)' if p_one >= 0.05 else 'rejected (better than shuffled views)'},
        'answer_spread': {'raw': t_summary(spread.raw), 'partial_on_volatility': part, 'sd_vs_vol': t_summary(spread.sd_vs_vol),
                          'verdict': 'supported' if not (part['p'] < 0.05 and part['mean'] > 0) else 'rejected'},
        'behaviour_of_main': behaviour,
        'performance': perf.reset_index().to_dict(orient='records'),
    }
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: summary[k] for k in ('primary_placebo', 'answer_spread', 'behaviour_of_main')}, indent=2, default=float))
    print(perf[['annual_return', 'active_return', 'tracking_error', 'information_ratio', 't_active', 'effective_n',
                'turnover', 'adoption_rel_median']].round(3).to_string())


if __name__ == '__main__':
    main()
