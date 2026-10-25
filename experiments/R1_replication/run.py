"""
R1: the replication of S1-S4 on 1,000 new small caps, and the new primary test R-BLM (does S4's calibrated BLM beat the
float-weighted band?). Run after S1-S4 have been run once on the new sample (SMALLCAP_RESULTS=results_rep, see
preregistration.md); reads their results_rep/ (and the original results/ for the side-by-side table).

    python experiments/R1_replication/run.py [--dry]     # --dry reads the dry runs of the new sample
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP = REPO / 'experiments'
DIRS = {'S1': 'S1_smallcap_drift', 'S2': 'S2_smallcap_risk', 'S3': 'S3_smallcap_risk_blm', 'S4': 'S4_te_calibration'}
TE_DAILY = 0.03 / np.sqrt(252)
BLOCK, N_BOOT = 63, 2000
EVAL = {'2015': '2017-01-01', '2023': '2024-07-01'}
PRIMARY = '2015/quarter/chrono_ridge/B_cal'
SECONDARY = ['2015/half/chrono_ridge/B_cal', '2015/year/chrono_ridge/B_cal', '2023/quarter/chrono_ridge/B_cal',
             '2023/half/chrono_ridge/B_cal', '2023/year/chrono_ridge/B_cal', '2023/quarter/gpt_ridge/B_cal']


def mean_test(a, seed=0):
    """Circular block bootstrap of the mean daily active return; one-sided p = share of bootstrap means <= 0."""
    a = np.asarray(a, float)
    n, nb = len(a), int(np.ceil(len(a) / BLOCK))
    rng = np.random.default_rng(seed)
    means = np.array([a[(rng.integers(0, n, nb)[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n].mean()
                      for _ in range(N_BOOT)])
    te = a.std() * np.sqrt(252)
    return {'days': int(n), 'annual_excess': float(a.mean() * 252), 'te_realized': float(te),
            'ir': float(a.mean() * 252 / te), 'ci95_annual': [float(np.quantile(means, 0.025) * 252),
                                                              float(np.quantile(means, 0.975) * 252)],
            'p_one_sided': float(np.mean(means <= 0))}


def blm_tests(res):
    u = pd.read_csv(res / 'u.csv')
    out = {}
    for key in [PRIMARY] + SECONDARY:
        s = u[(u.key == key) & (u.date >= EVAL[key.split('/')[0]])]
        if len(s) > 100:
            out[key] = mean_test(s.u.to_numpy() * TE_DAILY)
    return out


def load(name, sub):
    p = EXP / DIRS[name] / sub / 'summary.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def verdicts(rep):
    s1, s2, s3, s4 = (rep[k] for k in ('S1', 'S2', 'S3', 'S4'))
    return {'R-S1': s1['primary']['verdict'] == 'supported',
            'R-S2': s2['primary']['verdict'] == 'supported',
            'R-S3': s3['tests']['primary']['verdict'] == 'not supported',
            'R-S4': s4['tests']['primary']['verdict'] == 'supported'}


def side_by_side(orig, rep):
    rows = []
    for label, get in [
        ('S1 主：係数 bp/SD', lambda s: s['S1']['primary']['coef_bp_per_sd']),
        ('S1 主：t', lambda s: s['S1']['primary']['t']),
        ('S1 主：置換の p', lambda s: s['S1']['primary']['permutation_p_one_sided']),
        ('S1 gpt-4o-mini：係数 bp/SD', lambda s: s['S1']['secondary_drift_gpt_period']['gpt'].get('coef_bp_per_sd')),
        ('S1 月次の対決：費用 0.3% の後の月次の差', lambda s: s['S1']['portfolio']['excess_monthly_cost_0.003']),
        ('S2 主：係数 %', lambda s: s['S2']['primary']['coef']),
        ('S2 主：t', lambda s: s['S2']['primary']['t']),
        ('S3 主：b（B）', lambda s: s['S3']['tests']['primary']['b_ref']),
        ('S3 主：b（C）', lambda s: s['S3']['tests']['primary']['b_new']),
        ('S3 主：p', lambda s: s['S3']['tests']['primary']['p_one_sided']),
        ('S4 主：g（B）', lambda s: s['S4']['tests']['primary']['b_ref']),
        ('S4 主：g（B＋校正）', lambda s: s['S4']['tests']['primary']['b_new']),
        ('S4 主：p', lambda s: s['S4']['tests']['primary']['p_one_sided']),
        ('BLM（2015 年末、四半期、B＋校正）の最終の値', lambda s: s['S4']['runs'][PRIMARY]['final_value']),
        ('比べる相手の最終の値', lambda s: s['S4']['runs']['2015/quarter/band']['final_value']),
    ]:
        row = {'item': label}
        for name, s in (('original', orig), ('replication', rep)):
            try:
                row[name] = get(s)
            except (KeyError, TypeError):
                row[name] = None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    args = a.parse_args()
    sub = 'results_rep/dry' if args.dry else 'results_rep'
    out_dir = HERE / ('results/dry' if args.dry else 'results')
    if not args.dry and (out_dir / 'summary.json').exists():
        raise SystemExit('results/summary.json があります。R1 は実行済みです（事前登録どおり 1 回だけ）')
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {k: load(k, sub) for k in DIRS}
    orig = {k: load(k, 'results') for k in DIRS}
    blm = blm_tests(EXP / DIRS['S4'] / sub)
    blm_orig = blm_tests(EXP / DIRS['S4'] / 'results')
    primary = blm[PRIMARY]
    primary['verdict'] = 'supported' if primary['annual_excess'] > 0 and primary['p_one_sided'] < 0.05 else 'not supported'
    summary = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
               'dry': args.dry, 'replicated': verdicts(rep), 'R-BLM': primary,
               'blm_secondary': {k: v for k, v in blm.items() if k != PRIMARY},
               'blm_original_exploratory': blm_orig,
               'commits_of_the_runs': {k: (v or {}).get('commit') for k, v in rep.items()}}
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    side_by_side(orig, rep).to_csv(out_dir / 'side_by_side.csv', index=False)
    print(json.dumps({k: summary[k] for k in ('replicated', 'R-BLM')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
