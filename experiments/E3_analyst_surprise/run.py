"""
E3: does the gpt-4o-mini score still explain the reaction once the surprise against the analyst consensus is controlled?
Everything follows preregistration.md. Run once, after the pre-registration commit:

    python experiments/E3_analyst_surprise/run.py

Uses E2's panel (experiments/E2_event_reaction/results/panel.csv, rebuilt by E2's run.py) and the analyst
estimates in the database (run_events.py estimates). The estimates themselves are not written to results/
(Alpha Vantage's license is personal and non-sublicensable); only the statistics are.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load_e2():
    """E2's run.py under its own name (both experiments call their script run.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('e2_run', REPO / 'experiments' / 'E2_event_reaction' / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2 = _load_e2()
from events import config, estimates
from events.evaluate import holm, permutation_test, regress
from marketdb.db import events as db
from marketdb.db.connection import connect

OUT = Path(__file__).resolve().parent / 'results'
PANEL = REPO / 'experiments' / 'E2_event_reaction' / 'results' / 'panel.csv'
N_PERMUTATIONS = 1000
BASE = list(e2.CONTROLS)
M1 = BASE + ['analyst_surprise', 'analyst_missing']


def sample_a(conn):
    """E2's sample A with the matched analyst estimate of each release (section 3)."""
    if not PANEL.exists():
        raise SystemExit(f'{PANEL} がありません。experiments/E2_event_reaction/run.py で作り直してください')
    p = pd.read_csv(PANEL)
    A = p[p.abn_r_0_1.notna() & p.filing_date.between(*e2.SAMPLE_A) & p.llm_composite.notna()].copy()
    A = A.merge(db.load_releases(conn)[['accession', 'accepted_et']], on='accession', how='left')
    est = db.load_estimates(conn, sorted(A.symbol.unique()))
    m = estimates.match_releases(A, est)
    A = A.merge(m[['accession', 'matched', 'fiscal_date_ending', 'report_time']], on='accession', how='left')
    A = A.merge(est[['symbol', 'fiscal_date_ending', 'surprise', 'surprise_pct']], on=['symbol', 'fiscal_date_ending'],
                how='left')
    return A.reset_index(drop=True)


def prepare(A, focal, y, an_cut=e2.CUT, rank=False):
    """E2's preparation, plus the analyst variables (section 4)."""
    d = e2.prepare(A.assign(y=A[y]), focal, rank=rank)
    s = A['surprise_pct']
    d['analyst_missing'] = s.isna().astype(float)
    if rank:
        d['analyst_surprise'] = s.groupby(A.year_quarter).rank(pct=True).fillna(0.5)
    else:
        v = e2.winsorize(s, an_cut) if an_cut else s
        d['analyst_surprise'] = e2.standardize(v).fillna(0.0)
    d['beat'] = (A['surprise'] > 0).astype(float)
    d['miss'] = (A['surprise'] < 0).astype(float)
    return d


def fit(d, focal, controls, permute=False):
    d = d.dropna(subset=['y', focal])
    controls = [c for c in controls if d[c].std() > 0]
    full = regress(d, 'y', [focal, *controls], e2.FIXED)
    row = full.loc[focal]
    out = {'coef_bp_per_sd': float(row.coef * 1e4), 'se_bp': float(row.se * 1e4), 't': float(row.t),
           'p_one_sided': float(e2.one_sided(row)),
           'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4), float((row.coef + 1.96 * row.se) * 1e4)],
           'n': int(row.n), 'event_dates': int(row.clusters),
           'r2_gain': float(e2.r_squared(d, [focal, *controls]) - e2.r_squared(d, controls)),
           'others': {c: {'coef_bp': float(full.loc[c].coef * 1e4), 't': float(full.loc[c].t)}
                      for c in ('analyst_surprise', 'beat', 'miss', 'sue') if c in full.index}}
    if permute:
        perm = permutation_test(d, 'y', focal, controls, e2.FIXED, n=N_PERMUTATIONS, seed=0)
        out['permutation_p_one_sided'] = float(perm['p_one_sided'])
    return out


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    conn = connect(config.DB_PATH)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    A = sample_a(conn)
    counts = {'sample_A': int(len(A)), 'matched': int(A.matched.sum()), 'with_surprise': int(A.surprise_pct.notna().sum()),
              'beat': int((A.surprise > 0).sum()), 'meet': int((A.surprise == 0).sum()), 'miss': int((A.surprise < 0).sum())}
    print(counts, flush=True)

    d = prepare(A, 'llm_composite', 'abn_r_0_1')
    m0 = fit(d, 'llm_composite', BASE)
    m1 = fit(d, 'llm_composite', M1, permute=True)
    m1['verdict'] = '支持' if m1['p_one_sided'] < 0.05 and m1['permutation_p_one_sided'] < 0.05 else '支持されない'
    m1['ratio_to_m0'] = m1['coef_bp_per_sd'] / m0['coef_bp_per_sd']
    print('M0', m0['coef_bp_per_sd'], 'M1', m1, flush=True)

    # S1: the analyst surprise itself (from M1); S2: drift; S3: does the LLM score track the analyst surprise?
    s1_row = m1['others']['analyst_surprise']
    s1 = {'coef_bp_per_sd': s1_row['coef_bp'], 't': s1_row['t']}
    from scipy import stats
    s1['p_one_sided'] = float(stats.t.sf(s1_row['t'], df=m1['event_dates'] - 1))
    s2 = fit(prepare(A, 'llm_composite', 'abn_drift_2_21'), 'llm_composite', M1)
    d3 = prepare(A, 'llm_composite', 'surprise_pct')
    d3 = d3[d3.analyst_missing == 0]
    s3_full = regress(d3.dropna(subset=['y']), 'y', ['llm_composite', *[c for c in BASE if d3[c].std() > 0]], e2.FIXED)
    r3 = s3_full.loc['llm_composite']
    s3 = {'coef_pct_points_per_sd': float(r3.coef), 't': float(r3.t), 'p_one_sided': float(e2.one_sided(r3)), 'n': int(r3.n)}
    secondary = {'S1_analyst_surprise': s1, 'S2_drift': s2, 'S3_llm_tracks_surprise': s3}
    for (k, v), p in zip(secondary.items(), holm([v['p_one_sided'] for v in secondary.values()])):
        v['p_holm'] = float(p)

    clean = A.matched.fillna(False).astype(bool) & ~((A.session == 'open') & (A.report_time == 'post-market'))
    robust = {
        'R1_beat_miss': fit(d, 'llm_composite', BASE + ['beat', 'miss', 'analyst_missing']),
        'R2_ranks': fit(prepare(A, 'llm_composite', 'abn_r_0_1', rank=True), 'llm_composite', M1),
        'R3_matched_and_timing': fit(prepare(A[clean].reset_index(drop=True), 'llm_composite', 'abn_r_0_1'),
                                     'llm_composite', M1),
        'R4_no_winsorizing_surprise': fit(prepare(A, 'llm_composite', 'abn_r_0_1', an_cut=0), 'llm_composite', M1),
    }
    summary = {'code_commit': commit, 'counts': counts, 'M0_e2_model': m0, 'M1_main': m1, 'secondary': secondary,
               'robustness': robust, 'seconds': round(time.time() - t0, 1)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
