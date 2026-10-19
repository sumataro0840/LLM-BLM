"""
gpt-4o-mini on the cleaned text (修正の実験方針.md, 6,論点 3): run only if the development kept fix 1 for at least one of the
four models in the primary split B. Same prompt, model version and settings as experiment 2 (events/scorers/llm.py);
only the text changes (cleaned, then cut at the same 15,000 characters). Releases filed from 2023-11-01.
Costs about USD 3 (OpenAI). Run after develop.py:

    python experiments/F_fixes/gpt.py

Scores: data/cache/fixes/gpt_clean.csv (outside Git), with the composite tone + outlook + guidance_change (0 when null).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.F_fixes import common as C

SELECTION = Path(__file__).resolve().parent / 'results' / 'dev_selection.json'


def required(selection):
    return any(selection['B'][m]['reading'] != 'current' for m in selection['B'] if isinstance(selection['B'][m], dict))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from experiments.common import data as D
    import events.scorers.llm as L
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    out = C.CACHE / 'gpt_clean.csv'
    if not required(selection):
        (C.CACHE / 'gpt_clean_skipped.json').write_text(json.dumps({'run': False, 'reason': 'fix 1 not kept in split B'}),
                                                        encoding='utf-8')
        print('fix 1 was not kept in split B: gpt-4o-mini is not re-scored')
        return
    p = C.base_panel(D.connect())
    rel = p[p.filing_date >= C.GPT_START][['accession', 'text_path', 'filing_date']]
    done = set(pd.read_csv(out).accession) if out.exists() else set()
    rel = rel[~rel.accession.isin(done)].reset_index(drop=True)
    print(f'{len(rel)} releases to score', flush=True)
    L.texts = SimpleNamespace(read=lambda path, n=None: C.clean_text(path, n))      # the only change: cleaned text
    rows, cost = [], 0.0
    for k, r in enumerate(L.LLMScorer().score(rel), 1):
        v = r.values
        rows.append({'accession': r.accession, 'tone': v['tone'], 'outlook': v['outlook'],
                     'guidance_change': v['guidance_change'], 'cost_usd': r.call['cost_usd']})
        cost += r.call['cost_usd']
        if k % 500 == 0 or k == len(rel):
            pd.DataFrame(rows).to_csv(out, mode='a', header=not out.exists(), index=False)
            rows = []
            print(f'  {k}/{len(rel)}  USD {cost:.2f}', flush=True)


if __name__ == '__main__':
    main()
