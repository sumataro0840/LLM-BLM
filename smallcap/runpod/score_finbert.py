"""
Pod side: the official FinBERT scorer (events/scorers/finbert.py, as run_events.cmd_score uses it) on the bundle's
release list, minus the releases the PC already scored. Writes JSON lines: a header with the scorer's params, then
one line per release with its values and call, as db.save_scores takes them.

    python score_finbert.py OUT.jsonl SKIP.txt      # run in bundle/LLM-BLM
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from events.scorers.finbert import FinBERTScorer  # noqa: E402


def main():
    out, skip = Path(sys.argv[1]), set(Path(sys.argv[2]).read_text().split())
    rel = pd.read_csv(REPO / 'data' / 'cache' / 'smallcap_releases.csv', dtype=str, keep_default_na=False)
    todo = rel[~rel.accession.isin(skip)].reset_index(drop=True)
    scorer = FinBERTScorer()
    with open(out, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'header': True, 'method': scorer.method, 'model': scorer.model, 'params': scorer.params(),
                            'n_todo': len(todo)}) + '\n')
        for i, r in enumerate(scorer.score(todo), 1):
            f.write(json.dumps({'accession': r.accession, 'values': r.values, 'call': r.call}) + '\n')
            if i % 1000 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)}', flush=True)
    print('finbert done', flush=True)


if __name__ == '__main__':
    main()
