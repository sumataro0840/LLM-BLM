"""
E9: E7's monthly BLM over about ten years, with the look-ahead-free ChronoGPT predictions of E5 as the views.
Everything follows preregistration.md. Run once, after the pre-registration commit:

    python experiments/E9_monthly_blm_long/run.py

Decision dates: month ends 2016-03 .. 2026-07. A release filed up to 2023-10-31 uses E5's block-B prediction
(vintage 2015), a later one its block-A prediction (vintage 2022); block A's predictions of earlier releases are not
used (vintage 2022 had read those years).
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
from experiments.common import data as D
from marketdb.db.market import load_trading_days


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e7 = _load('e7_run', 'experiments/E7_monthly_blm/run.py')
x02 = e7.x02

OUT = Path(__file__).resolve().parent / 'results' / (sys.argv[1] if len(sys.argv) > 1 else '')
PANEL = REPO / 'experiments' / 'E2_event_reaction' / 'results' / 'panel.csv'
PRED = REPO / 'experiments' / 'E5_chrono_embeddings' / 'results' / 'predictions.csv'
FIRST, LAST = '2016-03', '2026-07'
CUTOFF = '2023-10-31'          # filing date that separates E5's blocks
TARGET = 0.03
MIN_RATIO_MONTHS = 12          # the TE limit is calibrated once 12 past months of this strategy exist
LOOKBACK = 36
SPLIT = '2023-09'              # last decision month whose views are all from block B (secondary S2)


def decision_dates(conn):
    """Month ends from FIRST to the month after LAST (X02's own list starts only in 2017-01; fixed after the first run,
    see results/findings.md)."""
    days = pd.Series(pd.to_datetime(load_trading_days(conn, '2016-01-01', '2026-09-30')))
    ends = days.groupby(days.dt.to_period('M')).max()
    return [d.strftime('%Y-%m-%d') for d in ends if FIRST <= d.strftime('%Y-%m') <= '2026-08']


def views_frame():
    """One row per release: symbol, event date, accepted time and the E5 prediction of its own block."""
    p = pd.read_csv(PANEL, usecols=['accession', 'symbol', 'event_date', 'accepted_at', 'filing_date'])
    pred = pd.read_csv(PRED)
    b = pred[pred.block == 'B'].set_index('accession').pred
    a = pred[pred.block == 'A'].set_index('accession').pred
    p['llm_composite'] = np.where(p.filing_date <= CUTOFF, p.accession.map(b), p.accession.map(a))
    return p[p.llm_composite.notna()]


def inputs(args):
    """Covariance, sigma, cap weights, the next month's returns and the views' z of one decision date."""
    date, next_end, frame = args
    conn = D.connect()
    returns, _, future, w_mkt = x02.inputs(conn, date, next_end)
    r252 = returns.iloc[-252:]
    cov = x02.lw(r252)
    scores = e7.latest_scores(frame, date, set(cov.index))
    z = pd.Series(e7.blom(scores.values), index=scores.index)
    return date, {'cov': cov, 'sigma': r252.std(), 'w_mkt': w_mkt, 'future': future, 'z': z,
                  'month': (1 + future).prod() - 1}


def limit_for(history):
    """3 % / mean realized-to-ex-ante TE ratio of this strategy's previous months (at most 36, at least 12)."""
    past = history[-LOOKBACK:]
    if len(past) < MIN_RATIO_MONTHS:
        return TARGET
    return TARGET / float(np.mean([h['te_realized'] / h['te_pred'] for h in past]))


def realized(x, w):
    act = w - x['w_mkt']
    daily = x['future'][w.index] @ act
    return float(np.sqrt(252) * daily.std()), float(np.sqrt(252 * act.values @ x['cov'].values @ act.values))


def run_llm(data):
    """Sequential, because each month's TE limit depends on the earlier months' realized TE."""
    out, history = [], []
    for date, x in data:
        lim = limit_for(history)
        w, status = e7.blm_weights(x['cov'], x['w_mkt'], x['sigma'], x['z'], lim)
        te_r, te_p = realized(x, w)
        history.append({'te_realized': te_r, 'te_pred': te_p})
        out.append((date, w, status, lim))
    return out


def run_perm(args):
    date, x, lim = args
    res = []
    for s in range(e7.SEEDS):
        rng = np.random.default_rng([s, int(date.replace('-', ''))])
        zp = pd.Series(rng.permutation(x['z'].values), index=x['z'].index)
        res.append(e7.blm_weights(x['cov'], x['w_mkt'], x['sigma'], zp, lim))
    return date, res


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    frame = views_frame()
    dates = decision_dates(conn)
    jobs = [(d, n, frame) for d, n in zip(dates[:-1], dates[1:]) if FIRST <= d[:7] <= LAST]
    print(f'{len(jobs)} decision dates {jobs[0][0]} .. {jobs[-1][0]}', flush=True)
    with ProcessPoolExecutor(5) as pool:
        data = list(pool.map(inputs, jobs))
    llm = run_llm(data)
    limits = {d: lim for d, _, _, lim in llm}
    with ProcessPoolExecutor(5) as pool:
        perms = dict(pool.map(run_perm, [(d, x, limits[d]) for d, x in data]))
    results = []
    for (date, x), (_, w, status, lim) in zip(data, llm):
        r = {'llm': (w, status), 'month': x['month'], 'w_mkt': x['w_mkt'], 'daily': x['future']}
        for s, p in enumerate(perms[date]):
            r[f'perm{s}'] = p
        results.append((date, r))
    df = e7.backtest(results, 'llm')
    df['limit'] = [limits[d] for d in df.index]
    df['n_views'] = [len(x['z']) for _, x in data]
    df.to_csv(OUT / 'by_month.csv')
    perm_dfs = [e7.backtest(results, f'perm{s}') for s in range(e7.SEEDS)]

    main_result = e7.summarize(df)
    main_result['verdict'] = '支持' if main_result['p_one_sided'] < 0.05 else '支持されない'
    perm_gross = pd.concat([p.active_gross for p in perm_dfs], axis=1).mean(axis=1)
    diff = df.active_gross - perm_gross
    early = df[df.index.str[:7] <= SPLIT]
    late = df[df.index.str[:7] > SPLIT]
    summary = {
        'code_commit': commit, 'n_months': len(df), 'first': df.index[0], 'last': df.index[-1],
        'views_per_month': {'mean': float(df.n_views.mean()), 'min': int(df.n_views.min())},
        'te_limit': {'mean': float(df.limit.mean()), 'min': float(df.limit.min()), 'max': float(df.limit.max())},
        'primary_vs_market': main_result,
        'S1_vs_permuted': {'mean_diff_gross_monthly': float(diff.mean()), 'nw_t': float(x02.newey_west_t(diff)),
                           'permuted_mean_active_net': float(np.mean([p.active_net.mean() for p in perm_dfs]))},
        'S2_block_B_months': e7.summarize(early) | {'months': len(early)},
        'S3_block_A_months': e7.summarize(late) | {'months': len(late)},
        'S4_te': {'realized_mean': main_result['realized_te_mean'], 'target': TARGET},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
