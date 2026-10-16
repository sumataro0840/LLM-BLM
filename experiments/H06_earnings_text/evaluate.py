"""
H06 step 3: do the scores read from earnings releases predict the next month, beyond the numbers?

    python experiments/H06_earnings_text/evaluate.py

Composite score = tone + outlook + guidance_change (null -> 0), fixed before the run, not fitted.
Writes results/summary.json, results/monthly.csv, results/fama_macbeth.csv, results/performance.csv.
Decision rules: 資料集/LLM-BLM_実験仮説.pdf, chapter H06.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common import llm
from experiments.common.backtest import market_cap_strategy, run
from experiments.common.metrics import fama_macbeth, performance, t_summary
from experiments.common.strategies import bl_strategy
from experiments.H02_statistical_baseline.run import load_metric_panel, rank_score, statistical_scores

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
FRESH_DAYS = 35
CONTROLS = ['revenue_growth_yoy', 'net_income_growth_yoy', 'return_1m', 'momentum_12m']
SCALE = 1e-3     # score points -> a number in the unit of daily returns (the calibration slope absorbs the scale)


def load_scores():
    rows = []
    for r in llm.load_log(OUT / 'scores.jsonl'):
        try:
            a = json.loads(r['answers'][0])
        except Exception:
            continue
        g = a.get('guidance_change')
        rows.append({'cik': r['cik'], 'accession': r['accession'], 'tone': a['tone'], 'outlook': a['outlook'],
                     'guidance': g, 'n_risks': len(a.get('risk_flags', [])),
                     'composite': a['tone'] + a['outlook'] + (g if g is not None else 0)})
    return pd.DataFrame(rows)


def forecasts_from(signal_by_date, all_data):
    """{date: Series over the universe} in daily-return units; stocks without a signal get the cap-weighted
    average of the others (a neutral view)."""
    out = {}
    for d, x in all_data.items():
        s = signal_by_date[d].reindex(x['universe'])
        w = x['market_weights']
        known = s.dropna()
        neutral = float((known * w.reindex(known.index)).sum() / w.reindex(known.index).sum()) if len(known) else 0.0
        out[d] = s.fillna(neutral) * SCALE
    return out


def panel_of(forecasts, all_data):
    rows = []
    for d, x in all_data.items():
        rows.append(pd.DataFrame({'date': d, 'Q': forecasts[d].values, 'r_next': x['next_mean_daily'].reindex(x['universe']).values,
                                  'n_next': x['next_days'].reindex(x['universe']).values}))
    return pd.concat(rows, ignore_index=True)


def main():
    conn = D.connect()
    all_data = D.load_all(conn)
    used = pd.read_csv(OUT / 'releases_used.csv', dtype={'cik': int})
    scores = load_scores()
    used = used.merge(scores, on=['cik', 'accession'], how='left')
    metrics = pd.read_sql("SELECT as_of_date, symbol, metric, value FROM stock_metrics WHERE metric IN (%s)"
                          % ','.join(f"'{c}'" for c in CONTROLS), conn).pivot_table(index=['as_of_date', 'symbol'], columns='metric', values='value')

    frames, monthly, signal = {}, [], {}
    for d, x in all_data.items():
        u = x['universe']
        g = used[used.date == d].drop_duplicates('symbol').set_index('symbol')
        df = pd.DataFrame({'composite': g.composite.reindex(u), 'tone': g.tone.reindex(u), 'n_risks': g.n_risks.reindex(u),
                           'age': (pd.Timestamp(d) - pd.to_datetime(g.filing_date)).dt.days.reindex(u),
                           'R_next': x['next_return'].reindex(u), 'r_in': x['input_mean'].reindex(u)})
        df = df.join(metrics.loc[d].reindex(u) if d in metrics.index.get_level_values(0) else pd.DataFrame(index=u, columns=CONTROLS))
        signal[d] = df.composite
        s = df.dropna(subset=['composite', 'R_next'])
        fresh = s[s.age <= FRESH_DAYS]
        rk = pd.DataFrame({c: rank_score(s[c]) for c in ['composite', 'R_next', *CONTROLS]}).fillna(0.0)
        frames[d] = rk
        monthly.append({'date': d, 'n_scored': len(s), 'n_fresh': len(fresh),
                        'ic': stats.spearmanr(s.composite, s.R_next)[0],
                        'ic_fresh': stats.spearmanr(fresh.composite, fresh.R_next)[0] if len(fresh) >= 20 else np.nan,
                        'ic_tone': stats.spearmanr(s.tone, s.R_next)[0],
                        'ic_risks': stats.spearmanr(-s.n_risks, s.R_next)[0],
                        'corr_return_1m': stats.spearmanr(s.composite, s.return_1m, nan_policy='omit')[0],
                        'corr_input_mean': stats.spearmanr(s.composite, s.r_in, nan_policy='omit')[0],
                        'corr_revenue_growth': stats.spearmanr(s.composite, s.revenue_growth_yoy, nan_policy='omit')[0],
                        'corr_income_growth': stats.spearmanr(s.composite, s.net_income_growth_yoy, nan_policy='omit')[0]})
    mon = pd.DataFrame(monthly).set_index('date')
    mon.to_csv(OUT / 'monthly.csv')
    fm_table, fm = fama_macbeth(frames, 'R_next', ['composite', *CONTROLS])
    fm_table.to_csv(OUT / 'fama_macbeth.csv')
    primary = fm['composite']
    verdict = 'supported' if (primary['mean'] > 0 and primary['p'] < 0.05) else 'not supported'

    # portfolios with the same construction (relative views, Sigma-proportional Omega, shrunk slope, TE 3%, costs)
    bench = run(market_cap_strategy, all_data)
    f_text = forecasts_from(signal, all_data)
    panel_metric, metric_dates = load_metric_panel(conn)
    nxt = {m: metric_dates[i + 1] for i, m in enumerate(metric_dates[:-1])}
    stat, _ = statistical_scores(panel_metric, list(all_data), lambda m, d: m in nxt and nxt[m] <= d)
    f_stat = forecasts_from({d: stat[d] * 10 for d in all_data}, all_data)    # ridge scores are about +-0.1
    f_llm = {d: x['view_mean'] for d, x in all_data.items()}
    llm_panel = D.forecast_panel(all_data)
    perf, monthly_runs = [], []
    for name, fc, panel in [('text_scores', f_text, panel_of(f_text, all_data)),
                            ('statistical_model', f_stat, panel_of(f_stat, all_data)),
                            ('llm_numeric_run1', None, llm_panel)]:
        res = run(bl_strategy(panel, form='relative', omega='proportional', shrink=True, forecasts=fc), all_data)
        perf.append({'name': name, **performance(res, bench), 'months_b_positive': int((res.b > 0).sum()),
                     'b_last': float(res.b.iloc[-1])})
        monthly_runs.append(res.drop(columns=['weights']).assign(name=name))
    perf = pd.DataFrame(perf).set_index('name')
    perf.loc['market_cap'] = performance(bench, bench)
    perf.to_csv(OUT / 'performance.csv')
    pd.concat(monthly_runs).to_csv(OUT / 'portfolio_monthly.csv')

    dist = scores[['tone', 'outlook', 'guidance']].apply(lambda c: c.value_counts(dropna=False, normalize=True).round(3).to_dict())
    summary = {
        'releases_scored': int(len(scores)), 'pairs_used': int(used.composite.notna().sum()),
        'scored_per_month': float(mon.n_scored.mean()), 'fresh_per_month': float(mon.n_fresh.mean()),
        'primary_fama_macbeth_composite': primary, 'verdict': verdict,
        'fama_macbeth_controls': {c: fm[c] for c in CONTROLS},
        'ic': t_summary(mon.ic), 'ic_fresh': t_summary(mon.ic_fresh), 'ic_tone': t_summary(mon.ic_tone),
        'ic_fewer_risks': t_summary(mon.ic_risks),
        'tracing': {c: float(mon[c].mean()) for c in ['corr_return_1m', 'corr_input_mean', 'corr_revenue_growth', 'corr_income_growth']},
        'score_distribution': dist.to_dict(),
        'spending': llm.spending(OUT / 'scores.jsonl'),
        'performance': perf.reset_index().to_dict(orient='records'),
    }
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'performance'}, indent=2, ensure_ascii=False, default=float))
    print(perf[['annual_return', 'active_return', 'tracking_error', 'information_ratio', 't_active', 'effective_n', 'turnover']].round(3).to_string())


if __name__ == '__main__':
    main()
