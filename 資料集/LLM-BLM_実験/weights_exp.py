"""State in the prompt how much weight each block gets, and see whether the predictions follow it."""
import sys, json, sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/kazuki/develop/工房/LLM-BLM'); sys.path.insert(0, '.')
import exp
from marketdb.db.stock import load_quarters
from llm import config
from llm.client import make_client, sample_expected_returns
HERE = '/private/tmp/claude-501/-Users-kazuki-develop---/9d2180f2-ca7f-42f9-a43d-a8ea8ccec7c4/scratchpad/industry/'
exp.DATES = ['2025-06-30', '2025-12-31', '2026-05-29']
WEIGHTS = {'p50_f50': (50, 50), 'p20_f80': (20, 80), 'p0_f100': (0, 100)}
def system(price, fund):
    return exp.SYSTEM + ("\nWeigh the two groups of information as follows when you form the prediction: "
                         f"[Price behaviour] {price}%, [Fundamentals] {fund}%.")
conn = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
tasks = exp.build(conn, load_quarters(conn))
tasks = tasks[(tasks.group == 'industrials') & (tasks.cond == 'with')].drop(columns='cond')
jobs = [(name, i) for name in WEIGHTS for i in range(len(tasks))]
print(len(tasks), 'stock-dates |', len(jobs), 'requests', flush=True)
client = make_client(config.OPENAI_API_KEY)
done = [0]
def ask(job):
    name, i = job
    try: a = sample_expected_returns(client, config.OPENAI_MODEL, system(*WEIGHTS[name]), tasks.prompt.iloc[i], 30, 1.0)[0]
    except Exception as e: a = str(e)[:100]
    done[0] += 1
    if done[0] % 100 == 0: print(done[0], flush=True)
    return a
with ThreadPoolExecutor(8) as pool: answers = list(pool.map(ask, jobs))
out = pd.concat([tasks.assign(cond=name) for name in WEIGHTS], ignore_index=True)
out['answers'] = [json.dumps(a) for a in answers]
out.to_pickle(HERE + 'weights.pkl'); print('saved', len(out), 'failed', sum(isinstance(a, str) for a in answers), flush=True)
