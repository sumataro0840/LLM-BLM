"""
E2: does the score gpt-4o-mini reads from an earnings release explain the stock reaction to it?
Everything follows preregistration.md (section numbers in the comments). Run once, after the pre-registration commit:

    python experiments/E2_event_reaction/run.py

Steps: samples (3) -> reactions (4) -> TF-IDF scores trained on earlier reactions (4, a new score run)
-> main test (5) -> secondary tests with Holm (6) -> robustness R1-R7 (6). Results go to results/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from events import config
from events.evaluate import holm, permutation_test, regress
from events.import_h06 import trading_days
from events.returns import OLD_SECTORS, WINDOWS, load_closes, log_market_cap, reactions, sector_benchmarks
from events.scorers import make
from marketdb.db import events as db
from marketdb.db.connection import connect, init_db

OUT = Path(__file__).resolve().parent / 'results'
RUNS = {'llm': 1, 'dictionary': 2, 'finbert': 3, 'chrono': 6}            # 7: fixed score runs
SAMPLE_A = ('2023-11-01', '2026-08-31')
SAMPLE_B = ('2016-01-01', '2023-10-31')
CUT = 0.01                                                              # 4: winsorize at 1% / 99%
N_PERMUTATIONS = 1000
CONTROLS = ['sue', 'revenue_surprise', 'sue_missing', 'revenue_surprise_missing',
            'dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred', 'abn_pre', 'log_mcap']
FIXED = ['sector', 'year_quarter']
COVID = ('2020-02-01', '2020-06-30')


# ---------- samples (3) ----------

def members_on(conn, releases):
    m = pd.read_sql("SELECT symbol, start_date, end_date FROM index_membership", conn)
    ok = []
    by = {s: g for s, g in m.groupby('symbol')}
    for r in releases.itertuples():
        g = by.get(r.symbol)
        ok.append(g is not None and bool(((g.start_date.isna() | (g.start_date <= r.filing_date))
                                          & (g.end_date.isna() | (g.end_date > r.filing_date))).any()))
    return np.array(ok)


def base_sample(conn, releases):
    """3, conditions 1, 2 and 5 (3 needs the reactions)."""
    rel = releases[releases.text_path.notna() & releases.doc_type.fillna('').str.startswith('EX-99')
                   & releases.event_date.notna()].copy()
    rel = rel[members_on(conn, rel)]
    rel = rel.sort_values('accepted_at').drop_duplicates(['symbol', 'event_date'], keep='first')
    return rel


# ---------- variables (4) ----------

def winsorize(x, cut=CUT):
    lo, hi = x.quantile(cut), x.quantile(1 - cut)
    return x.clip(lo, hi)


def standardize(x):
    return (x - x.mean()) / x.std()


def scores(conn):
    llm = db.load_scores(conn, RUNS['llm'])
    llm = llm.reindex(columns=['tone', 'outlook', 'guidance_change'])
    out = pd.DataFrame(index=llm.index)
    out['llm_tone'], out['llm_outlook'] = llm.tone, llm.outlook
    out['llm_guidance'] = llm.guidance_change.fillna(0.0)
    out['llm_composite'] = llm.tone + llm.outlook + llm.guidance_change.fillna(0.0)
    out = out.join(db.load_scores(conn, RUNS['dictionary'], ['net_tone']).rename(columns={'net_tone': 'dictionary_net_tone'}),
                   how='outer')
    out = out.join(db.load_scores(conn, RUNS['finbert'], ['net_tone']).rename(columns={'net_tone': 'finbert_net_tone'}),
                   how='outer')
    ch = db.load_scores(conn, RUNS['chrono'], ['guidance_net', 'tone_net'])
    out = out.join(ch.rename(columns={'guidance_net': 'chrono_guidance_net', 'tone_net': 'chrono_tone_net'}), how='outer')
    return out


def prepare(panel, focal, cut=CUT, rank=False):
    """4: winsorize the outcome-like variables, standardize the text scores, fill missing controls with 0 and add
    missing flags for the surprises. rank=True (R6): every variable becomes its rank (0..1) within the quarter."""
    d = panel.copy()
    numeric = ['y', 'sue', 'revenue_surprise', 'abn_pre', focal, 'dictionary_net_tone', 'finbert_net_tone',
               'tfidf_pred', 'log_mcap']
    for col in ('sue', 'revenue_surprise'):
        d[f'{col}_missing'] = d[col].isna().astype(float)
    if rank:
        for col in numeric:
            d[col] = d.groupby('year_quarter')[col].rank(pct=True)
    else:
        if cut:
            for col in ('y', 'sue', 'revenue_surprise', 'abn_pre'):
                d[col] = winsorize(d[col], cut)
        for col in (focal, 'dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred'):
            d[col] = standardize(d[col])
    for col in CONTROLS:
        if not col.endswith('_missing'):
            d[col] = d[col].fillna(d[col].mean() if rank else 0.0)
    return d


# ---------- tests (5, 6) ----------

def one_sided(row):
    """One-sided p for a positive coefficient, from the two-sided p of the clustered t test."""
    return row.p / 2 if row.coef > 0 else 1 - row.p / 2


def fit(panel, focal, y='y', permute=False, **kw):
    d = prepare(panel.assign(y=panel[y]), focal, **kw).dropna(subset=['y', focal])
    # a control that does not vary in this sample (e.g. no missing SUE at all) cannot be estimated
    controls = [c for c in CONTROLS if d[c].std() > 0]
    full = regress(d, 'y', [focal, *controls], FIXED)
    row = full.loc[focal]
    out ={'coef_bp_per_sd': float(row.coef * 1e4), 'se_bp': float(row.se * 1e4), 't': float(row.t),
           'p_one_sided': float(one_sided(row)), 'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4),
                                                            float((row.coef + 1.96 * row.se) * 1e4)],
           'n': int(row.n), 'event_dates': int(row.clusters),
           'r2_gain': float(r_squared(d, [focal, *controls]) - r_squared(d, controls)),
           'controls_dropped': [c for c in CONTROLS if c not in controls]}
    if permute:
        perm = permutation_test(d, 'y', focal, controls, FIXED, n=N_PERMUTATIONS, seed=0)
        out['permutation_p_one_sided'] = float(perm['p_one_sided'])
    return out


def r_squared(d, xs):
    from events.evaluate import design
    X = design(d, xs, FIXED).to_numpy(float)
    y = d['y'].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


# ---------- main ----------

def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    conn = connect(config.DB_PATH)
    init_db(conn)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[2]).stdout.strip()

    releases = db.load_releases(conn)
    rel = base_sample(conn, releases)
    days = trading_days(conn)
    bench = sector_benchmarks(conn)
    symbols = sorted(set(rel.symbol) | {b for b in bench.values() if b} | {'^GSPC'})
    closes = load_closes(conn, symbols, '2014-01-01', '2026-12-31')
    react = reactions(rel, closes, days, bench).set_index('accession')
    index = {d: n for n, d in enumerate(days)}
    react['window_end'] = [days[index[e] + 1] if e in index and index[e] + 1 < len(days) else None
                           for e in rel.set_index('accession').loc[react.index].event_date]
    react['log_mcap'] = log_market_cap(conn, rel, days)
    print(f'reactions of {len(react)} releases ({time.time() - t0:.0f}s)', flush=True)

    # TF-IDF trained on earlier reactions: a new score run that uses outcomes (4)
    texts_rel = releases[releases.text_path.notna()]
    scorer = make('tfidf', reactions=react, target='abn_r_0_1')
    tfidf_run = db.get_or_create_score_run(conn, 'tfidf', scorer.model, scorer.params(), True,
                                           f'E2, trained on reactions; code {commit[:10]}')
    done = db.scored_accessions(conn, tfidf_run)
    todo = texts_rel[~texts_rel.accession.isin(done)]
    for i, result in enumerate(scorer.score(todo), 1):
        db.save_scores(conn, tfidf_run, result.accession, result.values, result.call)
    print(f'tfidf run {tfidf_run} ({time.time() - t0:.0f}s)', flush=True)

    sc = scores(conn).join(db.load_scores(conn, tfidf_run, ['predicted_reaction']).rename(
        columns={'predicted_reaction': 'tfidf_pred'}))
    surprises = db.load_surprises(conn)[['sue', 'revenue_surprise']]
    sector = pd.read_sql("SELECT symbol, gics_sector FROM tickers", conn).set_index('symbol').gics_sector.replace(OLD_SECTORS)
    panel = (rel.set_index('accession')[['symbol', 'event_date', 'accepted_at', 'session', 'filing_date']]
             .join(react).join(sc).join(surprises))
    panel['sector'] = panel.symbol.map(sector)
    panel['year_quarter'] = pd.PeriodIndex(panel.event_date, freq='Q').astype(str)
    panel = panel.reset_index()
    panel['chrono_composite'] = standardize(panel.chrono_guidance_net) + standardize(panel.chrono_tone_net)

    # condition 3: the main window has prices; condition 4: the sample's score exists
    has_y = panel.abn_r_0_1.notna()
    A = panel[has_y & panel.filing_date.between(*SAMPLE_A) & panel.llm_composite.notna()].copy()
    B = panel[has_y & panel.filing_date.between(*SAMPLE_B) & panel.chrono_composite.notna()].copy()
    counts = {'base_sample': int(len(panel)), 'sample_A': int(len(A)), 'sample_B': int(len(B)),
              'A_without_prices': int((~has_y & panel.filing_date.between(*SAMPLE_A)).sum()),
              'B_without_prices': int((~has_y & panel.filing_date.between(*SAMPLE_B)).sum())}
    print(counts, flush=True)

    # 5: main test
    main_result = fit(A, 'llm_composite', y='abn_r_0_1', permute=True)
    main_result['verdict'] = ('支持' if main_result['p_one_sided'] < 0.05 and main_result['permutation_p_one_sided'] < 0.05
                              else '支持されない')
    print('main', main_result, f'({time.time() - t0:.0f}s)', flush=True)

    # 6: secondary tests, Holm over all of them (S4 is three regressions)
    secondary = {
        'S1_drift': fit(A, 'llm_composite', y='abn_drift_2_21'),
        'S2_sample_B_chrono': fit(B, 'chrono_composite', y='abn_r_0_1'),
        'S3_sample_A_chrono': fit(A, 'chrono_composite', y='abn_r_0_1'),
        'S4_tone': fit(A, 'llm_tone', y='abn_r_0_1'),
        'S4_outlook': fit(A, 'llm_outlook', y='abn_r_0_1'),
        'S4_guidance': fit(A, 'llm_guidance', y='abn_r_0_1'),
    }
    adjusted = holm([v['p_one_sided'] for v in secondary.values()])
    for (k, v), p in zip(secondary.items(), adjusted):
        v['p_holm'] = float(p)

    # 6: robustness (not used for the verdict)
    robust = {
        'R1_window_0_0': fit(A, 'llm_composite', y='abn_r_0_0'),
        'R1_window_m1_1': fit(A, 'llm_composite', y='abn_r_m1_1'),
        'R2_without_open': fit(A[A.session != 'open'], 'llm_composite', y='abn_r_0_1'),
        'R3_vs_sp500': fit(A.assign(abn_sp=A.ret_r_0_1 - sp500_returns(closes, days, A)), 'llm_composite', y='abn_sp'),
        'R4_no_winsorizing': fit(A, 'llm_composite', y='abn_r_0_1', cut=0),
        'R5_winsorize_0.5': fit(A, 'llm_composite', y='abn_r_0_1', cut=0.005),
        'R5_winsorize_2.5': fit(A, 'llm_composite', y='abn_r_0_1', cut=0.025),
        'R6_ranks': fit(A, 'llm_composite', y='abn_r_0_1', rank=True),
        'R7_B_without_covid': fit(B[~B.filing_date.between(*COVID)], 'chrono_composite', y='abn_r_0_1'),
    }

    summary = {'code_commit': commit, 'score_runs': {**RUNS, 'tfidf': tfidf_run}, 'counts': counts,
               'main': main_result, 'secondary': secondary, 'robustness': robust,
               'seconds': round(time.time() - t0, 1)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    panel.to_csv(OUT / 'panel.csv', index=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def sp500_returns(closes, days, frame):
    """Return of ^GSPC over the main window of each release (R3)."""
    index = {d: n for n, d in enumerate(days)}
    s = closes['^GSPC']
    out = []
    for e in frame.event_date:
        k = index.get(e)
        if k is None or k == 0 or k + 1 >= len(days):
            out.append(np.nan)
            continue
        p0, p1 = s.get(days[k - 1]), s.get(days[k + 1])
        out.append(p1 / p0 - 1 if p0 and p1 else np.nan)
    return np.array(out)


if __name__ == '__main__':
    main()
