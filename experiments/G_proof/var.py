"""
VaR and CVaR of plan.md's real portfolio (S4's main BLM on cleaned returns, one-way cost 0.3%) and of its band, from
run.py's daily series (2017-). Description only, no test.

  historical  VaR = minus the alpha-quantile of the daily returns, CVaR = minus the mean of the returns at or below it
  parametric  VaR = z * sd - mean (normal), CVaR = sd * phi(z) / (1 - alpha') - mean
for the BLM's return, the band's return and the active return (BLM - band), at 95% and 99%, one day.

    python experiments/G_proof/var.py      -> results/var.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL, COST = '2017-01-01', 0.003
Z = {0.95: 1.6448536269514722, 0.99: 2.3263478740408408}


def measures(x):
    x = np.asarray(x, float)
    out = {}
    for level, z in Z.items():
        q = np.quantile(x, 1 - level)
        mu, sd = x.mean(), x.std(ddof=1)
        out[f'{level:.0%}'] = {'hist_var': float(-q), 'hist_cvar': float(-x[x <= q].mean()),
                               'param_var': float(z * sd - mu),
                               'param_cvar': float(sd * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / (1 - level) - mu)}
    out['worst_day'] = float(x.min())
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    res = {'note': 'daily, 2017-, cleaned returns, one-way cost 0.3% on the BLM and the band; description only'}
    for name in ('orig', 'rep'):
        d = pd.read_csv(HERE / 'results' / name / 'real.csv')
        d = d[d.date >= EVAL]
        blm = d.port - COST * d.traded
        band = d.bench - COST * d.traded_band
        res[name] = {'days': int(len(d)), 'blm': measures(blm), 'band': measures(band), 'active': measures(blm - band),
                     'worst_dates': {'blm': d.date[blm.idxmin()], 'band': d.date[band.idxmin()],
                                     'active': d.date[(blm - band).idxmin()]}}
    (HERE / 'results' / 'var.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for name in ('orig', 'rep'):
        for k in ('blm', 'band', 'active'):
            m = res[name][k]
            print(name, k, {lv: {a: round(b * 100, 2) for a, b in m[lv].items()} for lv in ('95%', '99%')},
                  'worst', round(m['worst_day'] * 100, 2), res[name]['worst_dates'][k])


if __name__ == '__main__':
    main()
