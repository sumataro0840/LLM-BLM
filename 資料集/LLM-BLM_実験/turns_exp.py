"""Three-turn conversation (industry lens -> sector outlook -> stock data) vs one turn, numeric vs 5-level answer."""
import sys, json, sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
HERE = '/Users/kazuki/develop/工房/LLM-BLM_実験/'
sys.path.insert(0, '/Users/kazuki/develop/工房/LLM-BLM'); sys.path.insert(0, HERE)
import industry_exp as exp
from marketdb.db.stock import load_quarters
from llm import config
from llm.client import make_client

exp.DATES = ['2025-06-30', '2025-12-31', '2026-05-29']
MODEL = config.OPENAI_MODEL
ROLE = "You are a financial analyst."
RULES = ("Every return is a decimal fraction, not a percent: 0.01 means +1%.\n"
         "Percentile = position among the stocks of the same sector (0 = lowest, 100 = highest).\n"
         "\"n/a\" means the value is not available.")
ASK_NUMBER = ("Using only the information given in this conversation, predict the average daily return of the stock over the next month. "
              "Answer the expected average daily return as a decimal fraction (answer 0.001 for +0.1% per day, never 0.1).")
ASK_RATING = ("Using only the information given in this conversation, rate the stock's expected return over the next month "
              "relative to the other stocks of its sector: -2 = much lower, -1 = lower, 0 = about the same, 1 = higher, 2 = much higher.")
INDICATORS = [label for _, label in exp.PRICE + exp.FUND]
SCHEMA = {'number': {"type": "object", "properties": {"expected_return": {"type": "number"}}, "required": ["expected_return"], "additionalProperties": False},
          'rating': {"type": "object", "properties": {"rating": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}}, "required": ["rating"], "additionalProperties": False}}

def turn1_question(sector, sub):
    return (f"We will look at a company in the sub-industry \"{sub}\" (sector: {sector}). These indicators will be given:\n- " + "\n- ".join(INDICATORS) +
            "\n\nIn at most 80 words: in this sub-industry, which of these indicators say something about the stock's return over the next month, "
            "which are misleading because of how the business or its accounting works, and how should they be read?")

def turn2_question(conn, sector, date):
    m = pd.read_sql("SELECT metric, value, z_score FROM sector_metric_scores WHERE as_of_date = ? AND gics_sector = ?", conn, params=(date, sector), index_col='metric')
    px = dict(conn.execute("SELECT symbol, close FROM prices WHERE date = ? AND symbol IN ('^VIX', '^VIX3M', '^SKEW', '^IRX')", (date,)).fetchall())
    rep = conn.execute("SELECT report_id, published_date FROM reports WHERE published_date <= ? ORDER BY published_date DESC LIMIT 1", (date,)).fetchone()
    txt = dict(conn.execute("SELECT item, value_text FROM report_values WHERE report_run_id = 2 AND report_id = ? AND item IN ('us_summary', 'key_risks')", (rep[0],)).fetchall())
    f = lambda k: f"{m.value[k]:.3g} ({m.z_score[k]:+.2f})"
    return (f"Here is the state of the sector and of the market on {date}.\n\n[Sector: {sector}]  (value, z-score among the 11 sectors)\n"
            f"Momentum 3 / 6 / 12 months, latest month excluded: {f('momentum_3m')} / {f('momentum_6m')} / {f('momentum_12m')}\n"
            f"30-day volatility, per year: {f('volatility_30d')}\nPrice / earnings: {f('pe_trailing')}\n\n[Market]\n"
            f"VIX (percent per year): {px['^VIX']:.1f}\nVIX / 3-month VIX: {px['^VIX'] / px['^VIX3M']:.2f}\nSKEW index: {px['^SKEW']:.1f}\n"
            f"3-month Treasury bill yield (percent per year): {px['^IRX']:.2f}\n\n[Monthly outlook of an asset manager, published {rep[1]}]\n"
            f"Summary: {txt['us_summary']}\nKey risks: {txt['key_risks']}\n\n"
            "In at most 80 words: what is the outlook for this sector over the next month, and what would make a stock in it do better or worse than the sector?")

def chat(client, messages, kind=None, n=1, temperature=1.0):
    kw = {} if kind is None else {'response_format': {"type": "json_schema", "json_schema": {"name": kind, "strict": True, "schema": SCHEMA[kind]}}}
    c = client.chat.completions.create(model=MODEL, messages=messages, n=n, temperature=temperature, **kw)
    if kind is None: return c.choices[0].message.content
    out = []
    for ch in c.choices:
        try: out.append(float(list(json.loads(ch.message.content).values())[0]))
        except Exception: out.append(None)
    return out

if __name__ == '__main__':
    conn = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
    tasks = exp.build(conn, load_quarters(conn))
    tasks = tasks[tasks.group.isin(['industrials', 'reit']) & (tasks.cond == 'with')].drop(columns='cond').reset_index(drop=True)
    info = pd.read_sql("SELECT symbol, gics_sector, gics_sub_industry FROM tickers", conn, index_col='symbol')
    tasks['sector'] = tasks.symbol.map(info.gics_sector); tasks['sub'] = tasks.symbol.map(info.gics_sub_industry)
    client = make_client(config.OPENAI_API_KEY)
    with ThreadPoolExecutor(8) as pool:
        subs = tasks[['sector', 'sub']].drop_duplicates().values.tolist()
        t1 = {(s, b): (turn1_question(s, b), a) for (s, b), a in zip(subs, pool.map(
            lambda sb: chat(client, [{"role": "system", "content": ROLE}, {"role": "user", "content": turn1_question(*sb)}], temperature=0), subs))}
        def second(sbd):
            s, b, d = sbd; q2 = turn2_question(conn2(), s, d)
            return q2, chat(client, [{"role": "system", "content": ROLE}, {"role": "user", "content": t1[(s, b)][0]}, {"role": "assistant", "content": t1[(s, b)][1]},
                                     {"role": "user", "content": q2}], temperature=0)
        import threading; local = threading.local()
        def conn2():
            if not hasattr(local, 'c'): local.c = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
            return local.c
        keys = tasks[['sector', 'sub', 'date']].drop_duplicates().values.tolist()
        t2 = {tuple(k): v for k, v in zip(keys, pool.map(second, keys))}
    json.dump({'turn1': [{'sector': s, 'sub': b, 'question': q, 'answer': a} for (s, b), (q, a) in t1.items()],
               'turn2': [{'sector': s, 'sub': b, 'date': d, 'question': q, 'answer': a} for (s, b, d), (q, a) in t2.items()]},
              open(HERE + 'turns_texts.json', 'w'), ensure_ascii=False, indent=1)
    print('turn 1:', len(t1), '| turn 2:', len(t2), flush=True)

    def messages(r, cond):
        ask = ASK_RATING if cond.endswith('rating') else ASK_NUMBER
        sysm = {"role": "system", "content": ROLE + "\n" + RULES}
        if cond.startswith('one'):
            return [sysm, {"role": "user", "content": r.prompt + "\n\n" + ask}]
        q1, a1 = t1[(r.sector, r['sub'])]; q2, a2 = t2[(r.sector, r['sub'], r.date)]
        return [sysm, {"role": "user", "content": q1}, {"role": "assistant", "content": a1}, {"role": "user", "content": q2}, {"role": "assistant", "content": a2},
                {"role": "user", "content": "Here is the company.\n\n" + r.prompt + "\n\n" + ask}]
    CONDS = ['three_number', 'three_rating', 'one_rating']
    jobs = [(c, i) for c in CONDS for i in range(len(tasks))]
    print(len(tasks), 'stock-dates |', len(jobs), 'requests', flush=True)
    done = [0]
    def ask(job):
        c, i = job
        try: a = chat(client, messages(tasks.iloc[i], c), 'rating' if c.endswith('rating') else 'number', n=30)
        except Exception as e: a = str(e)[:100]
        done[0] += 1
        if done[0] % 100 == 0: print(done[0], flush=True)
        return a
    with ThreadPoolExecutor(8) as pool: answers = list(pool.map(ask, jobs))
    out = pd.concat([tasks.assign(cond=c) for c in CONDS], ignore_index=True)
    out['answers'] = [json.dumps(a) for a in answers]
    out.to_pickle(HERE + 'turns_results.pkl'); print('saved', len(out), 'failed', sum(isinstance(a, str) for a in answers), flush=True)
    print(json.dumps(messages(tasks.iloc[0], 'three_rating'), indent=1)[:200])
