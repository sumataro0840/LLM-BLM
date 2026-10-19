"""
Development of the fixes (修正の実験方針.md, 3-4): every score variant is built once, then the keep rule is applied
mechanically on each split's development period. No choice is made by hand. Run after score.py has finished:

    python experiments/F_fixes/develop.py

Writes results/dev_selection.json (what each split keeps) and results/dev_details.json (every comparison).

Development samples (the reaction must be realized inside the development period):
    firm level   releases filed from 2016-01-01 whose main window ended on or before the last development day
    sector level month ends from 2016-01 whose next month ends on or before the last development day

Keep rule (3.4), for each split and model:
  1. the fixed variant beats the current one on the main metric (firm: R2 gain, paired t of the squared-residual
     reduction; sector: mean IC, Newey-West t of the monthly difference) with t >= 1;
  2. it is not worse (delta >= 0) in either half of the development period;
  3. among several variants that pass, the best one by delta; a simpler passing variant replaces it when the best is
     not better than it by t >= 1 (simplicity: the order of the lists below).
Order: fix 1 against the current reading; only if fix 1 is kept, the fix 2 / fix 4 / fix 5 variants against fix 1
(fix 5, the compressed text, only if score.py's check says so; it is the least simple); then fix 3 (aggregation)
on the kept reading.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.F_fixes import common as C
from experiments.F_fixes import score as S

OUT = Path(__file__).resolve().parent / 'results'
SCORES = C.CACHE / 'scores.csv'

# reading variants of each model: (current, fix 1, {later fixes in order of simplicity})
MODELS = {
    'dictionary': ('dictionary_net_tone', 'dict_clean', {}),
    'finbert': ('finbert_net_tone', 'fb_head200', {'fix2_headtail': 'fb_headtail', 'fix2_all': 'fb_all',
                                                   'fix5': 'fb_comp'}),
    'tfidf': ('tfidf_pred', 'tfidf_clean', {}),
    'chrono': ('chrono_base', 'chrono_mean', {'fix4_mid': 'chrono_mid', 'fix4_last': 'chrono_last',
                                              'fix5': 'chrono_comp'}),
}


# ---------- every score variant (built once; rolling models only use earlier releases) ----------

def _chrono_matrix(p, names_keys):
    """Rows of p from the saved vectors (concatenating the given (folder, key) sources)."""
    accs, mats = [], []
    for name, key in names_keys:
        a, X = name(key) if callable(name) else S.load_chrono(name, key)
        accs.append(a)
        mats.append(X)
    acc, X = np.concatenate(accs), np.concatenate(mats)
    pos = pd.Series(np.arange(len(acc)), index=acc).reindex(p.accession)
    if pos.isna().any():
        raise SystemExit(f'{int(pos.isna().sum())} releases have no ChronoGPT vector; finish score.py first')
    return X[pos.to_numpy().astype(int)]


def assemble(conn):
    if SCORES.exists():
        return pd.read_csv(SCORES)
    p = C.base_panel(conn)
    out = p[['accession']].copy()
    d = pd.read_csv(C.CACHE / 'dictionary_clean.csv').set_index('accession').net_tone
    out['dict_clean'] = p.accession.map(d).to_numpy()
    f = pd.read_csv(C.CACHE / 'finbert_clean.csv').set_index('accession')
    for k, c in [('head200', 'fb_head200'), ('headtail', 'fb_headtail'), ('all', 'fb_all')]:
        out[c] = p.accession.map(f[k]).to_numpy()
    if (C.CACHE / 'finbert_compressed.csv').exists():
        fc = pd.read_csv(C.CACHE / 'finbert_compressed.csv').set_index('accession').net_tone
        out['fb_comp'] = p.accession.map(fc).to_numpy()
    print('tfidf on the cleaned text ...', flush=True)
    from events.scorers import tfidf as T
    path_of = dict(zip(p.accession, p.text_path))
    out['tfidf_clean'] = C.rolling_tfidf(p, lambda a: C.clean_text(path_of[a], T.MAX_CHARS)).reindex(p.accession).to_numpy()
    print('ChronoGPT ridges ...', flush=True)
    raw = _chrono_matrix(p, [(lambda key: C.e5.emb.load(2015), None), ('chrono_raw_v2015_ext', 'mean_last')])
    out['chrono_base'] = C.rolling_ridge(p, raw).reindex(p.accession).to_numpy()
    sources = [('chrono_clean_v2015', 'mean_last', 'chrono_mean'), ('chrono_clean_v2015', 'mean_mid', 'chrono_mid'),
               ('chrono_clean_v2015', 'last_token', 'chrono_last')]
    if (C.CACHE / 'chrono_comp_v2015').exists():
        sources.append(('chrono_comp_v2015', 'mean_last', 'chrono_comp'))
    for folder, key, col in sources:
        X = _chrono_matrix(p, [(folder, key)])
        out[col] = C.rolling_ridge(p, X).reindex(p.accession).to_numpy()
    out.to_csv(SCORES, index=False)
    return out


def panel_with_scores(conn):
    p = C.base_panel(conn)
    return p.merge(assemble(conn), on='accession', how='left')


# ---------- the keep rule ----------

def dev_firm(p, split):
    return p[(p.filing_date >= C.START) & (p.window_end <= C.SPLITS[split])]


def halves_firm(A, split):
    return A[A.window_end <= C.HALF[split]], A[A.window_end > C.HALF[split]]


def dev_months(months, split):
    """Month ends whose next month end is on or before the last development day."""
    nxt = dict(zip(months[:-1], months[1:]))
    return [m for m in months[:-1] if m >= C.START and nxt[m] <= C.SPLITS[split]]


def passes(full, h1, h2):
    return full['delta'] > 0 and full['t'] >= 1 and h1['delta'] >= 0 and h2['delta'] >= 0


def choose(current, candidates, compare, log):
    """Apply rules 1-3. candidates: [(label, variant)] in order of simplicity; compare(a, b) -> (full, h1, h2) with
    delta/t of b against a. Returns the label and variant kept."""
    passing = []
    for label, var in candidates:
        full, h1, h2 = compare(current[1], var)
        ok = passes(full, h1, h2)
        log.append({'against': current[0], 'variant': label, 'full': full, 'half1': h1, 'half2': h2, 'passes': ok})
        if ok:
            passing.append((label, var, full['delta']))
    if not passing:
        return current
    best = max(passing, key=lambda x: x[2])
    for label, var, _ in passing:                     # simplest first
        if label == best[0]:
            break
        full, _, _ = compare(var, best[1])
        log.append({'simpler_vs_best': [label, best[0]], 'full': full})
        if full['t'] < 1:
            return label, var
    return best[0], best[1]


def run_split(p, split, sd, months, fix5):
    A = dev_firm(p, split)
    h1, h2 = halves_firm(A, split)
    log, selection = [], {}

    def firm(a, b):
        return C.firm_compare(A, a, b), C.firm_compare(h1, a, b), C.firm_compare(h2, a, b)

    dm = dev_months(months, split)
    dm1, dm2 = [m for m in dm if m <= C.HALF[split]], [m for m in dm if m > C.HALF[split]]

    def sector(a, b):
        return (C.sector_compare(p, a, b, sd, dm), C.sector_compare(p, a, b, sd, dm1), C.sector_compare(p, a, b, sd, dm2))

    for model, (base, fix1, later) in MODELS.items():
        mlog = []
        reading = choose(('current', base), [('fix1', fix1)], firm, mlog)
        if reading[0] == 'fix1' and later:
            options = [(k, v) for k, v in later.items() if k != 'fix5' or fix5]
            reading = choose(reading, options, firm, mlog)
        agg = choose(('latest92_cap', (reading[1], 'latest92_cap')),
                     [(h, (reading[1], h)) for h in C.AGGREGATIONS[1:]], sector, mlog)
        selection[model] = {'reading': reading[0], 'column': reading[1], 'aggregation': agg[1][1]}
        log.append({'model': model, 'steps': mlog})
        print(split, model, selection[model], flush=True)
    return selection, log


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from experiments.common import data as D
    import subprocess
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    p = panel_with_scores(conn)
    sd = C.sector_data(conn)
    months = sorted(sd)
    fix5 = json.loads((C.CACHE / 'fix5_check.json').read_text(encoding='utf-8'))['run_fix5']
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=C.REPO).stdout.strip()
    selection, details = {'code_commit': commit, 'fix5_run': fix5}, {}
    for split in ['B', 'A', 'C']:
        selection[split], details[split] = run_split(p, split, sd, months, fix5)
    (OUT / 'dev_selection.json').write_text(json.dumps(selection, indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'dev_details.json').write_text(json.dumps(details, indent=1, ensure_ascii=False, default=float),
                                          encoding='utf-8')
    print(json.dumps(selection, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
