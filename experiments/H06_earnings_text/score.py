"""
H06 step 2: gpt-4o-mini reads each needed earnings release (text only, no numbers from us) and scores it.

    python experiments/H06_earnings_text/score.py [test]

A release is needed when it is the latest one filed before a decision date (strictly before, at most
120 days earlier) for a stock of that date's universe. Answers are logged in results/scores.jsonl.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common import llm

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
LOG = OUT / 'scores.jsonl'
MAX_DAYS = 120
MAX_CHARS = 15000

SYSTEM = (
    "You read one quarterly earnings press release of a US company and score what it says. "
    "Use only the text of the release. Do not use anything you know about the company from elsewhere, "
    "about later events, or about how the stock price moved. Judge the substance, not the usual promotional wording "
    "that every release uses.")
QUESTION = """Score the release:
- guidance_change: the company's forward guidance compared with its previous guidance, as the release states it. -2 = cut sharply, -1 = lowered, 0 = reaffirmed or unchanged, 1 = raised, 2 = raised sharply. null when the release gives no forward guidance or does not say how it compares.
- outlook: what the release says about the business over the coming quarters (demand, orders, pipeline, margins). -2 = clearly deteriorating, -1 = somewhat weaker, 0 = stable or not discussed, 1 = somewhat stronger, 2 = clearly improving.
- tone: management's overall tone about the quarter and the business. -2 = clearly negative, -1 = cautious, 0 = neutral, 1 = positive, 2 = clearly positive.
- risk_flags: the problems the release itself reports (empty when none).

The release (it may be cut off):
"""
SCHEMA = {
    'name': 'earnings_release_scores', 'strict': True,
    'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['guidance_change', 'outlook', 'tone', 'risk_flags'],
        'properties': {
            'guidance_change': {'anyOf': [{'type': 'integer', 'enum': [-2, -1, 0, 1, 2]}, {'type': 'null'}]},
            'outlook': {'type': 'integer', 'enum': [-2, -1, 0, 1, 2]},
            'tone': {'type': 'integer', 'enum': [-2, -1, 0, 1, 2]},
            'risk_flags': {'type': 'array', 'items': {'type': 'string', 'enum': [
                'demand_weakness', 'margin_pressure', 'cost_inflation', 'restructuring_or_layoffs', 'impairment_or_writedown',
                'legal_or_regulatory', 'supply_chain', 'currency_or_macro', 'debt_or_liquidity', 'management_change']}},
        },
    },
}
PARAMS = {'temperature': 0, 'max_tokens': 120, 'response_format': {'type': 'json_schema', 'json_schema': SCHEMA}}


def needed_releases(all_data, filings):
    """(date, symbol) -> the release used, by the rule above."""
    f = filings.dropna(subset=['file']).copy()
    f['symbol'] = f.symbols.str.split(',')
    f = f.explode('symbol')
    rows = []
    for d, x in all_data.items():
        lo = (pd.Timestamp(d) - pd.Timedelta(days=MAX_DAYS)).strftime('%Y-%m-%d')
        cand = f[(f.filing_date < d) & (f.filing_date >= lo) & f.symbol.isin(x['universe'])]
        latest = cand.sort_values('filing_date').groupby('symbol').tail(1)
        rows.append(latest.assign(date=d)[['date', 'symbol', 'cik', 'accession', 'filing_date', 'file']])
    return pd.concat(rows, ignore_index=True)


def main(test=False):
    conn = D.connect()
    all_data = D.load_all(conn)
    filings = pd.read_csv(OUT / 'filings.csv', dtype={'cik': int})
    use = needed_releases(all_data, filings)
    use.to_csv(OUT / 'releases_used.csv', index=False)
    todo = use.drop_duplicates(['cik', 'accession'])
    done = llm.done_keys(LOG, ['cik', 'accession'])
    todo = todo[[(c, a) not in done for c, a in zip(todo.cik, todo.accession)]]
    if test:
        todo = todo.head(30)
    print(f'{len(use)} (date, stock) pairs use {use.drop_duplicates(["cik", "accession"]).shape[0]} releases; {len(todo)} to score', flush=True)
    api = llm.client()

    def one(r):
        text = (D.REPO / r.file).read_text(encoding='utf-8')[:MAX_CHARS]
        try:
            llm.chat(api, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': QUESTION + text}], LOG,
                     {'cik': int(r.cik), 'accession': r.accession, 'filing_date': r.filing_date}, **PARAMS)
        except Exception as e:
            return f'{r.accession}: {str(e)[:150]}'
    with ThreadPoolExecutor(8) as pool:
        errors = [e for e in pool.map(one, todo.itertuples()) if e]
    print('failed:', len(errors), errors[:3])
    print(llm.spending(LOG))


if __name__ == '__main__':
    main(test=len(sys.argv) > 1 and sys.argv[1] == 'test')
