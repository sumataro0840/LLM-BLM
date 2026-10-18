"""
Checks added after the results were shared with two other models (gpt-5.3-codex, gemini-3.1-pro) for review.
These are exploratory (not pre-registered).

    python experiments/robustness/run.py [nw|temperature|feedback|placebo]

nw           Newey-West (lag 3) t-values of the primary monthly series of H02, H06, H07
temperature  H05: are the digit probabilities the same at temperature 0 and 1? (30 requests each)
feedback     H07: did F2 follow its own average written in the summary, or the market?
placebo      H04: shuffles within sectors (a stricter uninformative control), 100 runs
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common import llm
from experiments.common.backtest import market_cap_strategy, run
from experiments.common.metrics import performance

E = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'results'


def newey_west_t(x, lags=3):
    x = np.asarray(pd.Series(x).dropna(), float)
    n, m = len(x), x.mean()
    e = x - m
    s = (e @ e) / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    t = m / math.sqrt(s / n)
    return {'mean': m, 't_nw': t, 'p_nw': float(2 * stats.t.sf(abs(t), n - 1)), 'n': n, 'lags': lags,
            'autocorr_1': float(pd.Series(x).autocorr(1))}


def nw():
    h02 = pd.read_csv(E / 'H02_statistical_baseline/results/monthly_ic.csv')
    h06 = pd.read_csv(E / 'H06_earnings_text/results/fama_macbeth.csv')
    h06m = pd.read_csv(E / 'H06_earnings_text/results/monthly.csv')
    h07 = pd.read_csv(E / 'H07_feedback_llm/results/monthly.csv')
    ic = h07.pivot(index='date', columns='condition', values='ic').iloc[1:]
    out = {'H02_ic_llm_residual': newey_west_t(h02.ic_llm_residual), 'H02_ic_llm': newey_west_t(h02.ic_llm),
           'H06_fm_composite': newey_west_t(h06.composite), 'H06_ic': newey_west_t(h06m.ic),
           'H06_ic_fresh': newey_west_t(h06m.ic_fresh), 'H07_ic_F2_minus_F0': newey_west_t(ic.F2 - ic.F0)}
    return out


def temperature():
    from experiments.H05_text_vs_numbers.run import SYSTEM, build
    g = pd.read_pickle(D.REPO / '過去のコードと記録' / '総合報告1の実験' / 'premium_results.pkl')
    sample = g.sample(30, random_state=0)
    api = llm.client()
    log = OUT / 'temperature.jsonl'
    rows = []
    for r in sample.itertuples():
        probs = {}
        for temp in (0.0, 1.0):
            rec = llm.chat(api, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': build(r, 'T')}], log,
                           {'date': r.date, 'name': r.name, 'temperature': temp},
                           max_tokens=1, temperature=temp, logprobs=True, top_logprobs=20)
            top = rec['top_logprobs'][0]
            p = np.array([math.exp(top[d]) if d in top else 0.0 for d in '12345'])
            probs[temp] = p / p.sum()
        rows.append({'max_abs_prob_diff': float(np.abs(probs[0.0] - probs[1.0]).max()),
                     'score_diff': float((probs[0.0] - probs[1.0]) @ np.arange(1, 6))})
    df = pd.DataFrame(rows)
    return {'requests': 60, 'max_abs_prob_diff_max': float(df.max_abs_prob_diff.max()),
            'max_abs_prob_diff_median': float(df.max_abs_prob_diff.median()),
            'score_diff_abs_max': float(df.score_diff.abs().max()), 'spending': llm.spending(log)}


def feedback():
    """F2's monthly mean forecast against the average forecast written in its prompt and against F0 of the same month."""
    rows = []
    for line in open(E / 'H07_feedback_llm/results/answers_F2.jsonl', encoding='utf-8'):
        r = json.loads(line)
        m = re.search(r'average forecast: ([-\d.]+) per day; average actual: ([-\d.]+)', r['messages'][1]['content'])
        if m:
            rows.append({'date': r['date'], 'stated_forecast': float(m.group(1)), 'stated_actual': float(m.group(2))})
    stated = pd.DataFrame(rows).groupby('date').first()
    f = pd.read_csv(E / 'H07_feedback_llm/results/forecasts.csv')
    mean = f.pivot_table(index='date', columns='condition', values='forecast', aggfunc='mean')
    df = stated.join(mean[['F0', 'F0p', 'F2']], how='inner')
    df['gap_F2_F0'] = df.F2 - df.F0
    # market level of the month (cap-weighted input mean) as the alternative explanation
    all_data = D.load_all(D.connect())
    df['market_input'] = [float((all_data[d]['input_mean'].reindex(all_data[d]['market_weights'].index) * all_data[d]['market_weights']).sum()) for d in df.index]
    X = np.column_stack([np.ones(len(df)), df.stated_forecast, df.F0])
    beta = np.linalg.lstsq(X, df.F2.values, rcond=None)[0]
    return {'months': len(df),
            'corr_F2_vs_stated_forecast': float(np.corrcoef(df.F2, df.stated_forecast)[0, 1]),
            'corr_F2_vs_F0_same_month': float(np.corrcoef(df.F2, df.F0)[0, 1]),
            'corr_gap_F2_F0_vs_stated_forecast': float(np.corrcoef(df.gap_F2_F0, df.stated_forecast)[0, 1]),
            'corr_gap_F2_F0_vs_market_input': float(np.corrcoef(df.gap_F2_F0, df.market_input)[0, 1]),
            'regression_F2_on_stated_and_F0': {'const': beta[0], 'stated_forecast': beta[1], 'F0': beta[2]},
            'mean_F2_minus_stated': float((df.F2 - df.stated_forecast).mean()),
            'table': df.round(5).reset_index().to_dict(orient='records')}


def _sector_placebo(seed):
    from experiments.common.strategies import bl_strategy
    conn = D.connect()
    all_data = D.load_all(conn)
    panel = D.forecast_panel(all_data)
    forecasts = {}
    for d, x in all_data.items():
        q = x['view_mean'].reindex(x['universe'])
        rng = np.random.default_rng([seed, int(d.replace('-', ''))])
        out = q.copy()
        for _, idx in x['sector'].reindex(q.index).groupby(x['sector'].reindex(q.index)).groups.items():
            out[idx] = rng.permutation(q[idx].values)
        forecasts[d] = out
    bench = run(market_cap_strategy, all_data)
    # the calibration slope stays the one of the real forecasts (the panel holds the real Q)
    res = run(bl_strategy(panel, form='relative', omega='proportional', shrink=True, forecasts=forecasts), all_data)
    return {'seed': seed, **performance(res, bench)}


def placebo(n=100):
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=10) as pool:
        for k, fut in enumerate(as_completed([pool.submit(_sector_placebo, s) for s in range(n)]), 1):
            rows.append(fut.result())
            if k % 10 == 0:
                print(f'sector placebo {k}/{n} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows).sort_values('seed')
    df.to_csv(OUT / 'placebo_within_sector.csv', index=False)
    actual = json.loads((E / 'H04_omega_calibration/results/summary.json').read_text(encoding='utf-8'))['primary_placebo']['actual_active_return']
    p = (1 + int((df.active_return >= actual).sum())) / (1 + len(df))
    return {'actual': actual, 'mean': float(df.active_return.mean()), 'p95': float(df.active_return.quantile(0.95)),
            'n': len(df), 'p_one_sided': p, 'mean_turnover': float(df.turnover.mean()), 'mean_te': float(df.tracking_error.mean())}


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    what = sys.argv[1]
    result = {'nw': nw, 'temperature': temperature, 'feedback': feedback, 'placebo': placebo}[what]()
    (OUT / f'{what}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'table'}, indent=2, ensure_ascii=False, default=float))
