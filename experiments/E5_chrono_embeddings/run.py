"""
E5: does ChronoGPT's internal representation, mapped to reactions by a ridge regression trained on earlier releases,
explain the reaction to earnings releases? Everything follows preregistration.md. Run once, after the
pre-registration commit and after embed.py has finished both blocks:

    python experiments/E5_chrono_embeddings/run.py

Block B (vintage 2015) tests sample B 2016-01 .. 2023-10, block A (vintage 2022) sample A 2023-11 .. 2026-08.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config
from events.evaluate import holm, permutation_test, regress
from marketdb.db import events as db
from marketdb.db.connection import connect

OUT = Path(__file__).resolve().parent / 'results'
ALPHAS = np.logspace(0, 5, 21)          # chosen inside each training set by leave-one-out (RidgeCV), never on the test
MIN_TRAIN = 500
N_PERMUTATIONS = 1000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2 = _load('e2_run', 'experiments/E2_event_reaction/run.py')
e3 = _load('e3_run', 'experiments/E3_analyst_surprise/run.py')
emb = _load('e5_embed', 'experiments/E5_chrono_embeddings/embed.py')


def rolling_ridge(frame, X):
    """Out-of-sample predictions for every calendar quarter (by acceptance date): fit on the releases whose window
    closed before the quarter began. frame: accession, accepted_et, window_end, abn_r_0_1 aligned with the rows of X.
    Inside each training set: the target is winsorized at its 1%/99% points and standardized, the features are
    standardized with the training mean and sd, and the penalty is chosen by leave-one-out over ALPHAS."""
    from sklearn.linear_model import RidgeCV
    q = pd.PeriodIndex(frame.accepted_et.str[:10], freq='Q')
    pred = np.full(len(frame), np.nan)
    info = []
    for quarter in sorted(q.unique()):
        start = quarter.start_time.strftime('%Y-%m-%d')
        tr = ((frame.window_end < start) & frame.abn_r_0_1.notna()).to_numpy()
        te = (q == quarter)
        if tr.sum() < MIN_TRAIN:
            continue
        y = frame.abn_r_0_1.to_numpy()[tr]
        y = np.clip(y, *np.quantile(y, [0.01, 0.99]))
        y = (y - y.mean()) / y.std()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        model = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, y)
        pred[te] = model.predict((X[te] - mu) / sd)
        info.append({'quarter': str(quarter), 'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                     'alpha': float(model.alpha_)})
    return pd.Series(pred, index=frame.accession.to_numpy(), name='chrono_emb'), info


def block_predictions(conn, vintage):
    acc, X = emb.load(vintage)
    rel = db.load_releases(conn)[['accession', 'accepted_et']]
    panel = pd.read_csv(e3.PANEL, usecols=['accession', 'window_end', 'abn_r_0_1'])
    frame = pd.DataFrame({'accession': acc}).merge(rel, on='accession').merge(panel, on='accession')
    assert len(frame) == len(acc) and (frame.accession.to_numpy() == acc).all()
    return rolling_ridge(frame, X)


def fit(d, focal, controls, permute=False):
    d = d.dropna(subset=['y', focal])
    controls = [c for c in controls if d[c].std() > 0]
    full = regress(d, 'y', [focal, *controls], e2.FIXED)
    row = full.loc[focal]
    out = {'coef_bp_per_sd': float(row.coef * 1e4), 'se_bp': float(row.se * 1e4), 't': float(row.t),
           'p_one_sided': float(e2.one_sided(row)), 'n': int(row.n), 'event_dates': int(row.clusters),
           'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4), float((row.coef + 1.96 * row.se) * 1e4)],
           'others': {c: {'coef_bp': float(full.loc[c].coef * 1e4), 't': float(full.loc[c].t)}
                      for c in ('chrono_emb', 'llm_composite', 'chrono_composite', 'tfidf_pred')
                      if c in full.index and c != focal}}
    if permute:
        out['permutation_p_one_sided'] = float(permutation_test(d, 'y', focal, controls, e2.FIXED,
                                                                n=N_PERMUTATIONS, seed=0)['p_one_sided'])
    return out


def r2_gains(d, base, a, b):
    dd = d.dropna(subset=['y', a, b])
    base = [c for c in base if dd[c].std() > 0]
    return {f'{a}_only': float(e2.r_squared(dd, base + [a]) - e2.r_squared(dd, base)),
            f'{b}_only': float(e2.r_squared(dd, base + [b]) - e2.r_squared(dd, base)),
            'both': float(e2.r_squared(dd, base + [a, b]) - e2.r_squared(dd, base))}


def p_from_t(t, clusters):
    from scipy import stats
    return float(stats.t.sf(t, df=clusters - 1))


def sample_b():
    p = pd.read_csv(e3.PANEL)
    return p[p.abn_r_0_1.notna() & p.filing_date.between(*e2.SAMPLE_B) & p.chrono_composite.notna()].copy()


def evaluate_b(B, pred):
    """P1 and the block-B secondary analyses (preregistration 4, 5)."""
    B = B.merge(pred.rename('chrono_emb').rename_axis('accession').reset_index(), on='accession', how='left')
    d = e2.prepare(B.assign(y=B.abn_r_0_1), 'chrono_emb')
    d['chrono_composite'] = e2.standardize(d.chrono_composite)
    controls = list(e2.CONTROLS)
    p1 = fit(d, 'chrono_emb', controls, permute=True)
    s3 = fit(d, 'chrono_emb', controls + ['chrono_composite'])
    without_tfidf = fit(d, 'chrono_emb', [c for c in controls if c != 'tfidf_pred'])
    r2 = r2_gains(d, [c for c in controls if c != 'tfidf_pred'], 'chrono_emb', 'tfidf_pred')
    dr = e2.prepare(B.assign(y=B.abn_r_0_1), 'chrono_emb', rank=True)
    rank = fit(dr, 'chrono_emb', controls)
    corr = {c: float(B.chrono_emb.corr(B[c], method='spearman')) for c in ('tfidf_pred', 'chrono_composite')}
    return {'P1': p1, 'S3_with_zero_shot_chrono': s3, 'R1_without_tfidf': without_tfidf, 'R2_ranks': rank,
            'r2_gain_vs_tfidf': r2, 'spearman': corr, 'n_with_prediction': int(B.chrono_emb.notna().sum())}


def evaluate_a(conn, pred):
    """P2 and the block-A secondary analyses."""
    A = e3.sample_a(conn).merge(pred.rename('chrono_emb').rename_axis('accession').reset_index(), on='accession', how='left')
    d = e3.prepare(A, 'llm_composite', 'abn_r_0_1')
    d['chrono_emb'] = e2.standardize(A['chrono_emb'])
    m1 = e3.M1
    p2 = fit(d, 'llm_composite', m1 + ['chrono_emb'], permute=True)
    s1 = fit(d, 'chrono_emb', m1)
    s2 = {k: p2['others']['chrono_emb'][k] for k in ('coef_bp', 't')}
    s2['p_one_sided'] = p_from_t(s2['t'], p2['event_dates'])
    r2 = r2_gains(d, m1, 'llm_composite', 'chrono_emb')
    dr = e3.prepare(A, 'llm_composite', 'abn_r_0_1', rank=True)
    dr['chrono_emb'] = A['chrono_emb'].groupby(A.year_quarter).rank(pct=True)
    rank = fit(dr, 'llm_composite', m1 + ['chrono_emb'])
    corr = float(A.chrono_emb.corr(A.llm_composite, method='spearman'))
    return {'P2': p2, 'S1_emb_alone': s1, 'S2_emb_with_llm': s2, 'R3_ranks': rank, 'r2_gain_over_controls': r2,
            'spearman_emb_llm': corr, 'n_with_prediction': int(A.chrono_emb.notna().sum())}


def verdict(r):
    return '支持' if r['p_holm'] < 0.05 and r['permutation_p_one_sided'] < 0.05 else '支持されない'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    conn = connect(config.DB_PATH)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    pred_b, info_b = block_predictions(conn, 2015)
    pred_a, info_a = block_predictions(conn, 2022)
    res_b = evaluate_b(sample_b(), pred_b)
    res_a = evaluate_a(conn, pred_a)
    primary = [res_b['P1'], res_a['P2']]
    for r, p in zip(primary, holm([r['p_one_sided'] for r in primary])):
        r['p_holm'] = float(p)
        r['verdict'] = verdict(r)
    secondary = {'S1': res_a['S1_emb_alone'], 'S2': res_a['S2_emb_with_llm'], 'S3': res_b['S3_with_zero_shot_chrono']}
    for r, p in zip(secondary.values(), holm([r['p_one_sided'] for r in secondary.values()])):
        r['p_holm'] = float(p)
    OUT.mkdir(exist_ok=True)
    pd.concat([pred_b.rename('pred').to_frame().assign(block='B'), pred_a.rename('pred').to_frame().assign(block='A')]
              ).rename_axis('accession').to_csv(OUT / 'predictions.csv')
    summary = {'code_commit': commit, 'block_B': res_b, 'block_A': res_a,
               'ridge': {'B': info_b, 'A': info_a}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('block_B', 'block_A')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
