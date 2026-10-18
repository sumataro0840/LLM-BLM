"""
H07: what a feedback LLM changes. Four new conditions on 100 stocks, against run 1 (F0):

  F0p  the run-1 prompt asked again (noise floor of a re-ask)
  F1   + the stock's own forecast of last month and what actually happened (its own forecasts, month by month)
  F2   + a summary of its own past forecasts over all sampled stocks (level, rank correlation, input tracking)
  F3   a separate reviewer: gets the run-1 forecast, the input and the same summary computed on run 1, and adjusts it

    python experiments/H07_feedback_llm/run.py ask [test]
    python experiments/H07_feedback_llm/run.py evaluate

Same model, temperature, number of answers and structured output as run 1. Decision rules:
資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf, chapter H07.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common import llm
from experiments.common.metrics import t_summary
from llm.context import build_context
from llm.prompts import SYSTEM_PROMPT, make_user_prompt
from llm.returns import to_daily_returns
from marketdb.db.market import last_trading_day_before, load_prices, load_ticker_table

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
N_STOCKS, SEED = 100, 0
MIN_EFFECT = 0.03
SCHEMA = {'name': 'expected_return', 'strict': True,
          'schema': {'type': 'object', 'additionalProperties': False, 'required': ['expected_return'],
                     'properties': {'expected_return': {'type': 'number'}}}}
PARAMS = {'temperature': 1.0, 'n': 30, 'response_format': {'type': 'json_schema', 'json_schema': SCHEMA}}
REVIEWER = (
    "You review a stock return forecast made by another model. You get the information the forecaster saw, "
    "its forecast, and a record of how its past forecasts turned out. Give your own best estimate of the average daily "
    "return for the next month. Every return is a decimal fraction, not a percent: 0.01 means +1%. "
    "Answer the expected average daily return as a decimal fraction (answer 0.001 for +0.1% per day, never 0.1).")


def sample_stocks(all_data):
    first = next(iter(all_data.values()))
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(sorted(first['universe']), N_STOCKS, replace=False))


def contexts(conn, d, symbols):
    """The run-1 user prompt of each stock on date d (same price window and rounding as run_llm.py)."""
    month_start = pd.Timestamp(d).replace(day=1).strftime('%Y-%m-%d')
    prices = load_prices(conn, symbols, last_trading_day_before(conn, month_start), d)
    returns = to_daily_returns(prices)
    table = load_ticker_table(conn)
    out = {}
    for s in symbols:
        c = build_context(s, table, returns)
        c['pct_change'] = [r for r in c['pct_change'] if r == r]
        out[s] = (make_user_prompt(c), float(np.mean(c['pct_change'])) if c['pct_change'] else np.nan)
    return out


def parse(record):
    vals = []
    for a in record['answers']:
        try:
            vals.append(float(json.loads(a)['expected_return']))
        except Exception:
            pass
    return np.mean(vals) if vals else np.nan, np.var(vals) if vals else np.nan, len(vals)


def summary_text(hist):
    """Feedback on past forecasts (hist: rows of date, forecast, actual, input over the sampled stocks)."""
    if hist.empty:
        return None
    ics = hist.groupby('date').apply(lambda g: stats.spearmanr(g.forecast, g.actual)[0] if len(g) > 5 else np.nan).dropna()
    tr = hist.groupby('date').apply(lambda g: np.corrcoef(g.forecast, g.input)[0, 1] if len(g) > 5 else np.nan).dropna()
    return (f"Record of the past forecasts ({hist.date.nunique()} months, {len(hist)} stock-months):\n"
            f"- average forecast: {hist.forecast.mean():.4f} per day; average actual: {hist.actual.mean():.4f} per day\n"
            f"- rank correlation between forecasts and actual returns (average over months): {ics.mean():+.3f}\n"
            f"- correlation between forecasts and the average of the daily returns given as input: {tr.mean():+.2f}")


def ask(test=False):
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    all_data = D.load_all(conn)
    dates = list(all_data)[:2] if test else list(all_data)
    stocks = sample_stocks(all_data)
    (OUT / 'stocks.json').write_text(json.dumps(stocks))
    api = llm.client()
    run1 = D.forecast_panel(all_data).set_index(['date', 'symbol'])
    logs = {c: OUT / f'answers_{c}.jsonl' for c in ('F0p', 'F1', 'F2', 'F3')}
    done = {c: llm.done_keys(p, ['date', 'symbol']) for c, p in logs.items()}
    own = {c: {} for c in logs}                      # (date, symbol) -> mean forecast of the condition
    for c, p in logs.items():
        if p.exists():
            for r in llm.load_log(p):
                own[c][(r['date'], r['symbol'])] = parse(r)[0]

    for i, d in enumerate(dates):
        x = all_data[d]
        syms = [s for s in stocks if s in x['universe']]
        ctx = contexts(conn, d, syms)
        prev = dates[i - 1] if i > 0 else None
        # F2: summary of its own past forecasts (outcomes known by date d); the same for every stock of the month
        hist = pd.DataFrame([{'date': dd, 'forecast': own['F2'].get((dd, ss)), 'actual': run1.r_next.get((dd, ss)),
                              'input': run1.r_in.get((dd, ss))} for dd in dates[:i] for ss in stocks
                             if (dd, ss) in own['F2']]).dropna() if i > 0 else pd.DataFrame()
        text = summary_text(hist)
        # F3: the same summary computed on run 1 (the forecaster that is reviewed)
        hist1 = pd.DataFrame([{'date': dd, 'forecast': run1.Q.get((dd, ss)), 'actual': run1.r_next.get((dd, ss)),
                               'input': run1.r_in.get((dd, ss))} for dd in dates[:i] for ss in stocks
                              if (dd, ss) in run1.index]).dropna() if i > 0 else pd.DataFrame()
        text1 = summary_text(hist1)
        jobs = []
        for s in syms:
            user, r_in = ctx[s]
            base = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
            jobs.append(('F0p', s, base))
            # F1: its own forecast of last month for this stock and what happened (= this month's input average)
            f1 = user
            if prev is not None and (prev, s) in own['F1'] and np.isfinite(own['F1'][(prev, s)]):
                f1 += (f"\n\nFeedback on your previous forecast for this stock: last month you predicted an average daily return "
                       f"of {own['F1'][(prev, s)]:.4f}; the actual average daily return was {r_in:.4f}.")
            jobs.append(('F1', s, [base[0], {'role': 'user', 'content': f1}]))
            jobs.append(('F2', s, [base[0], {'role': 'user', 'content': user + (f'\n\n{text}' if text else '')}]))
            # F3: reviewer of the run-1 forecast
            q0 = run1.Q.get((d, s))
            review = (f"{user}\n\nThe forecaster's answer (average of 30 answers): {q0:.4f}"
                      + (f"\n\n{text1}" if text1 else "\n\n(No record yet: this is the first month.)"))
            jobs.append(('F3', s, [{'role': 'system', 'content': REVIEWER}, {'role': 'user', 'content': review}]))
        jobs = [j for j in jobs if (d, j[1]) not in done[j[0]]]

        def one(job):
            c, s, messages = job
            try:
                rec = llm.chat(api, messages, logs[c], {'condition': c, 'date': d, 'symbol': s}, **PARAMS)
                own[c][(d, s)] = parse(rec)[0]
            except Exception as e:
                return f'{c} {d} {s}: {str(e)[:120]}'
        with ThreadPoolExecutor(8) as pool:
            errors = [e for e in pool.map(one, jobs) if e]
        print(d, len(jobs), 'requests, failed', len(errors), errors[:2], flush=True)
        if test and i == 1:
            print(jobs[1][2][1]['content'][-400:] if jobs else '')
    for c, p in logs.items():
        print(c, llm.spending(p))


def evaluate():
    conn = D.connect()
    all_data = D.load_all(conn)
    run1 = D.forecast_panel(all_data)
    rows = []
    for c in ('F0p', 'F1', 'F2', 'F3'):
        for r in llm.load_log(OUT / f'answers_{c}.jsonl'):
            m, v, n = parse(r)
            rows.append({'condition': c, 'date': r['date'], 'symbol': r['symbol'], 'forecast': m, 'variance': v, 'n_parsed': n})
    t = pd.DataFrame(rows)
    stocks = json.loads((OUT / 'stocks.json').read_text())
    f0 = run1[run1.symbol.isin(stocks)][['date', 'symbol', 'Q', 'variance']].rename(columns={'Q': 'forecast'}).assign(condition='F0')
    t = pd.concat([t, f0], ignore_index=True).merge(run1[['date', 'symbol', 'Q', 'r_in', 'r_next', 'n_next', 'R_next']], on=['date', 'symbol'])
    t = t[t.n_next >= 15]
    t.to_csv(OUT / 'forecasts.csv', index=False)
    per = []
    for (c, d), g in t.groupby(['condition', 'date']):
        per.append({'condition': c, 'date': d, 'ic': stats.spearmanr(g.forecast, g.R_next)[0],
                    'bias': g.forecast.mean() - g.r_next.mean(), 'input_corr': np.corrcoef(g.forecast, g.r_in)[0, 1],
                    'slope': float(((g.forecast - g.forecast.mean()) * (g.r_next - g.r_next.mean())).sum() / ((g.forecast - g.forecast.mean()) ** 2).sum()),
                    'change_vs_f0': (g.forecast - g.Q).abs().mean(), 'answer_sd': np.sqrt(g.variance).mean(),
                    'positive_share': (g.forecast > 0).mean()})
    per = pd.DataFrame(per)
    per.to_csv(OUT / 'monthly.csv', index=False)
    first = sorted(t.date.unique())[0]
    live = per[per.date != first]                  # feedback exists from the second month
    piv = lambda col: live.pivot(index='date', columns='condition', values=col)
    ic = piv('ic')
    diff = t_summary(ic['F2'] - ic['F0'])
    if diff['p'] < 0.05 and diff['mean'] > 0:
        verdict = 'rejected (feedback raises the IC)'
    elif diff['ci_high'] < MIN_EFFECT:
        verdict = 'supported (no IC gain of 0.03 or more)'
    else:
        verdict = 'insufficient evidence'
    noise = piv('change_vs_f0')['F0p'].mean()
    summary = {
        'primary_ic_F2_minus_F0': diff, 'verdict': verdict,
        'by_condition': {c: {'ic': t_summary(ic[c]), 'abs_bias': float(piv('bias')[c].abs().mean()), 'bias': float(piv('bias')[c].mean()),
                             'input_corr': float(piv('input_corr')[c].mean()), 'slope': float(piv('slope')[c].mean()),
                             'change_vs_f0': float(piv('change_vs_f0')[c].mean()),
                             'change_over_noise_floor': float(piv('change_vs_f0')[c].mean() / noise),
                             'answer_sd': float(piv('answer_sd')[c].mean()), 'positive_share': float(piv('positive_share')[c].mean())}
                         for c in ['F0', 'F0p', 'F1', 'F2', 'F3']},
        'ic_differences_vs_F0': {c: t_summary(ic[c] - ic['F0']) for c in ['F0p', 'F1', 'F3']},
        'spending': {c: llm.spending(OUT / f'answers_{c}.jsonl') for c in ('F0p', 'F1', 'F2', 'F3')},
        'stocks': len(stocks), 'months_with_feedback': int(live.date.nunique()),
    }
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == '__main__':
    if sys.argv[1] == 'ask':
        ask(test=len(sys.argv) > 2 and sys.argv[2] == 'test')
    else:
        evaluate()
