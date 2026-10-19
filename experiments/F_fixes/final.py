"""
The final comparison of the fix experiments (preregistration.md). Run once, after develop.py, bert.py and gpt.py:

    python experiments/F_fixes/final.py

For each split (B primary, A and C as checks) on its final period, and on gpt-4o-mini's period (2023-11 ..):
  primary   firm level, split B: does ChronoGPT (fixed) explain the reaction beyond the numeric controls and the three
            NLP scores (fixed)? One-sided t (standard errors clustered by event date) and the within-date permutation
            (1,000); 'supported' when both p < 0.05.
  table     firm level: R2 gain over the numeric controls of each model; out-of-sample AUC of a rolling logistic
            regression of the reaction's sign on the numeric features with the model's score, against the numeric
            features alone. Sector level: mean monthly IC and the sector Black-Litterman portfolio after costs.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from events.evaluate import permutation_test, regress
from experiments.F_fixes import common as C
from experiments.F_fixes import develop as DV

OUT = Path(__file__).resolve().parent / 'results'
NLP = ['dictionary', 'finbert', 'tfidf']
N_PERMUTATIONS = 1000


def final_months(months, split, first=None):
    start = first or C.SPLITS[split]
    nxt = dict(zip(months[:-1], months[1:]))
    return [m for m in months[:-1] if m >= start and nxt[m] <= C.END]


def primary(F, chrono_col, nlp_cols, permute):
    F = F.dropna(subset=[chrono_col, *nlp_cols])
    d = C._prepared(F, chrono_col)
    for c in nlp_cols:
        d[c] = C.e2.standardize(F[c])
    xs = [c for c in C.NUM_CONTROLS if d[c].std() > 0] + list(nlp_cols)
    row = regress(d, 'y', ['_score', *xs], C.FIXED).loc['_score']
    out = {'coef_bp_per_sd': float(row.coef * 1e4), 't': float(row.t), 'p_one_sided': float(C.e2.one_sided(row)),
           'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4), float((row.coef + 1.96 * row.se) * 1e4)],
           'n': int(row.n), 'event_dates': int(row.clusters)}
    if permute:
        out['permutation_p_one_sided'] = float(permutation_test(d, 'y', '_score', xs, C.FIXED, n=N_PERMUTATIONS,
                                                                seed=0)['p_one_sided'])
        out['verdict'] = '支持' if out['p_one_sided'] < 0.05 and out['permutation_p_one_sided'] < 0.05 else '支持されない'
    return out


def entries(sel, split, p):
    """(name, [score columns], aggregation for the sector level) of every compared model."""
    e = [('logistic', ['logit'], 'latest92_cap'),
         ('chrono_base', ['chrono_base'], 'latest92_cap'),
         ('chrono_fixed', [sel['chrono']['column']], sel['chrono']['aggregation'])]
    for m in NLP:
        e += [(f'{m}_base', [DV.MODELS[m][0]], 'latest92_cap'), (f'{m}_fixed', [sel[m]['column']], sel[m]['aggregation'])]
    e += [('nlp_base', [DV.MODELS[m][0] for m in NLP], None), ('nlp_fixed', [sel[m]['column'] for m in NLP], None),
          (f'bert_ft', [f'bert_{split}'], 'latest92_cap')]
    return e


def evaluate(p, F, months, ents, sd, logit_cache):
    firm, sector = {}, {}
    for name, cols, how in ents:
        if not all(c in p for c in cols) or F[cols].notna().all(axis=1).sum() < 200:
            continue
        firm[name] = {'r2_gain': C.r2_gain(F, cols), 'n': int(F[cols].notna().all(axis=1).sum())}
        if name != 'logistic':
            key = tuple(cols)
            if key not in logit_cache:
                logit_cache[key] = C.rolling_logit(p, cols)
            G = F.assign(_a=F.accession.map(logit_cache[('logit',)]), _b=F.accession.map(logit_cache[key]))
            firm[name]['auc'] = C.auc_diff_ci(G, '_a', '_b')
        if how:
            ic = C.sector_ics(p, cols[0], sd, months, how)
            sector[name] = {'ic_mean': float(ic.mean()), 'ic_t': C.newey_west_t(ic), 'months': int(ic.notna().sum()),
                            'aggregation': how, 'blm': C.blm_summary(C.sector_blm(p, cols[0], how, sd, months))}
        print('  ', name, firm[name].get('r2_gain'), firm[name].get('auc', {}).get('diff'), flush=True)
    return firm, sector


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from experiments.common import data as D
    conn = D.connect()
    selection = json.loads((OUT / 'dev_selection.json').read_text(encoding='utf-8'))
    p = DV.panel_with_scores(conn)
    for split in ['B', 'A', 'C']:
        b = pd.read_csv(C.CACHE / f'bert_ft_{split}.csv').set_index('accession').bert_ft
        p[f'bert_{split}'] = p.accession.map(b)
    gpt_path = C.CACHE / 'gpt_clean.csv'
    if gpt_path.exists():
        g = pd.read_csv(gpt_path).set_index('accession')
        p['gpt_clean'] = p.accession.map(g.tone + g.outlook + g.guidance_change.fillna(0))
    p['logit'] = p.accession.map(C.rolling_logit(p))
    sd = C.sector_data(conn)
    months = sorted(sd)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=C.REPO).stdout.strip()
    summary = {'code_commit': commit, 'selection': selection, 'splits': {}}
    logit_cache = {('logit',): p.set_index('accession').logit}
    for split in ['B', 'A', 'C']:
        sel = selection[split]
        F = p[p.filing_date >= C.final_start(split)]
        print(split, len(F), flush=True)
        res = {'primary': primary(F, sel['chrono']['column'], [sel[m]['column'] for m in NLP], permute=True)}
        res['firm'], res['sector'] = evaluate(p, F, final_months(months, split), entries(sel, split, p), sd, logit_cache)
        summary['splits'][split] = res
    # gpt-4o-mini's period, with split B's fixed models
    sel = selection['B']
    G = p[p.filing_date >= C.GPT_START]
    gm = final_months(months, 'B', first='2024-01-01')
    ents = entries(sel, 'B', p) + [('gpt4o_base', ['llm_composite'], 'latest92_cap')]
    if 'gpt_clean' in p:
        ents.append(('gpt4o_fix1', ['gpt_clean'], 'latest92_cap'))
    short = {'primary': primary(G, sel['chrono']['column'], [sel[m]['column'] for m in NLP], permute=False)}
    short['firm'], short['sector'] = evaluate(p, G, gm, ents, sd, logit_cache)
    summary['gpt_period'] = short
    b = summary['splits']['B']
    summary['fallback_check_required'] = bool(b['primary']['verdict'] != '支持' and
                                              b['firm']['chrono_fixed']['r2_gain'] < b['firm']['nlp_fixed']['r2_gain'])
    (OUT / 'final_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                            encoding='utf-8')
    print(json.dumps({s: summary['splits'][s]['primary'] for s in summary['splits']}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
