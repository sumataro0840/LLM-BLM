"""
H03: view forms (absolute / relative / rank) under a diagonal and a Sigma-proportional Omega.

    python experiments/H03_view_shape/run.py

No API calls. Writes results/summary.json, results/performance.csv, results/monthly.csv.
Decision rules: 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf, chapter H03.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common.backtest import market_cap_strategy, run
from experiments.common.metrics import performance
from experiments.common.strategies import bl_strategy

OUT = Path(__file__).resolve().parent / 'results'
FORMS = ['absolute', 'relative', 'rank']
OMEGAS = ['calibrated_diagonal', 'proportional']


def corr_test(a, b):
    """Pearson correlation over the months and its two-sided p-value (t-test of a zero correlation)."""
    r, p = stats.pearsonr(a, b)
    return {'r': float(r), 'p': float(p), 'n': len(a)}


def main():
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    all_data = D.load_all(conn)
    panel = D.forecast_panel(all_data)
    bench = run(market_cap_strategy, all_data)
    results, perf, monthly = {}, [], []
    for te in (0.03, None):
        for omega in OMEGAS:
            for form in FORMS:
                name = f'{form}|{omega}|{"te3" if te else "no_te"}'
                res = run(bl_strategy(panel, form=form, omega=omega, te=te), all_data)
                results[name] = res
                perf.append({'name': name, 'form': form, 'omega': omega, 'te_limit': te, **performance(res, bench)})
                m = res.drop(columns=['weights']).assign(name=name)
                monthly.append(m)
                print(name, f"beta {res.active_beta.abs().mean():.3f}  eff {res.effective_n.mean():.1f}", flush=True)
    perf = pd.DataFrame(perf).set_index('name')
    perf.loc['market_cap'] = performance(bench, bench)
    perf.to_csv(OUT / 'performance.csv')
    pd.concat(monthly).to_csv(OUT / 'monthly.csv')

    # primary test: with a diagonal Omega (and the TE limit), the forecast level shows up as an active beta
    tests = {}
    for te in ('te3', 'no_te'):
        for omega in OMEGAS:
            for form in FORMS:
                r = results[f'{form}|{omega}|{te}']
                live = r[r.b > 0]
                tests[f'{form}|{omega}|{te}'] = {
                    'level_vs_active_beta_all_months': corr_test(r.forecast_level, r.active_beta),
                    'level_vs_active_beta_b_positive': corr_test(live.forecast_level, live.active_beta) if len(live) > 3 else None,
                    'mean_abs_active_beta': float(r.active_beta.abs().mean()),
                    'mean_max_active_weight': float(r.max_active_weight.mean()),
                    'months_b_positive': int((r.b > 0).sum()),
                }
    a, rel = tests['absolute|calibrated_diagonal|te3'], tests['relative|calibrated_diagonal|te3']
    c = a['level_vs_active_beta_all_months']
    ratio = rel['mean_abs_active_beta'] / a['mean_abs_active_beta'] if a['mean_abs_active_beta'] > 0 else np.nan
    if c['r'] > 0 and c['p'] < 0.05 and ratio <= 0.5:
        verdict = 'supported'
    elif not (c['r'] > 0 and c['p'] < 0.05):
        verdict = 'not supported (the level of the forecasts does not show up as an active beta)'
    else:
        verdict = 'not supported (relative views do not halve the active beta)'
    summary = {'primary': {'absolute_diag_te3_corr': c, 'relative_over_absolute_abs_beta': ratio, 'verdict': verdict},
               'secondary_rank_max_active_weight_vs_absolute': {
                   'diag_te3': (tests['rank|calibrated_diagonal|te3']['mean_max_active_weight'],
                                a['mean_max_active_weight']),
                   'prop_te3': (tests['rank|proportional|te3']['mean_max_active_weight'],
                                tests['absolute|proportional|te3']['mean_max_active_weight'])},
               'tests': tests,
               'performance': perf.reset_index().to_dict(orient='records')}
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(summary['primary'], indent=2, default=float))
    print(perf[['annual_return', 'active_return', 'tracking_error', 'information_ratio', 't_active', 'effective_n',
                'abs_active_beta', 'turnover']].round(3).to_string())


if __name__ == '__main__':
    main()
