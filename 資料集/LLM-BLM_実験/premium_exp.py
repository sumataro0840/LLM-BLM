"""Risk premia: ask, per kind of risk and per month, whether bearing it will be rewarded next month.
No stock-level numbers are shown. The answer is one digit (1-5); its token probabilities are read."""
import sys, json, math, sqlite3, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
HERE = '/Users/kazuki/develop/工房/LLM-BLM_実験/'
DB = "file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro"
sys.path.insert(0, '/Users/kazuki/develop/工房/LLM-BLM')
from llm import config
from llm.client import make_client

SYSTEM = ("You are a financial analyst. Use only the information given below. "
          "Answer with a single digit from 1 to 5 and nothing else.")
SCALE = "1 = clearly lower, 2 = somewhat lower, 3 = about the same, 4 = somewhat higher, 5 = clearly higher."
FACTORS = {   # name -> (high side, low side)
    'market': ("US stocks (the S&P 500)", "3-month Treasury bills"),
    'beta': ("S&P 500 stocks with a high beta to the index", "S&P 500 stocks with a low beta"),
    'volatility': ("S&P 500 stocks whose prices have been the most volatile", "the least volatile S&P 500 stocks"),
    'value': ("S&P 500 stocks with a low price/earnings ratio (cheap)", "S&P 500 stocks with a high price/earnings ratio (expensive)"),
    'momentum': ("S&P 500 stocks that rose the most over the past 12 months", "S&P 500 stocks that rose the least"),
    'safety': ("S&P 500 companies with little long-term debt relative to equity", "S&P 500 companies with much long-term debt"),
    'size': ("the smaller companies of the S&P 500", "the largest companies of the S&P 500"),
}
SECTORS = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care',
           'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Utilities']

def state(conn, date):
    """Market state and the latest published outlook on `date` (text block, and the numbers for the analysis)."""
    px = pd.read_sql("SELECT date, symbol, close FROM prices WHERE symbol IN ('^GSPC', '^VIX', '^VIX3M', '^SKEW', '^IRX') AND date <= ? "
                     "AND date >= date(?, '-400 days')", conn, params=(date, date)).pivot(index='date', columns='symbol', values='close').ffill()
    sp = px['^GSPC']; last = px.iloc[-1]
    realized = sp.pct_change().iloc[-30:].std() * math.sqrt(252) * 100
    num = {'vix': last['^VIX'], 'vix_ratio': last['^VIX'] / last['^VIX3M'], 'skew': last['^SKEW'], 'tbill': last['^IRX'], 'realized_vol': realized,
           'vol_premium': last['^VIX'] - realized, 'sp_1m': sp.iloc[-1] / sp.iloc[-22] - 1, 'sp_12m': sp.iloc[-1] / sp.iloc[-253] - 1 if len(sp) > 253 else np.nan}
    rep = conn.execute("SELECT report_id, published_date FROM reports WHERE published_date <= ? ORDER BY published_date DESC LIMIT 1", (date,)).fetchone()
    txt = dict(conn.execute("SELECT item, value_text FROM report_values WHERE report_run_id = 2 AND report_id = ? AND item IN ('us_summary', 'key_risks')", (rep[0],)).fetchall())
    block = (f"Date: {date}\n\n[Market]\nS&P 500 return over the last month: {num['sp_1m']:+.1%}\nS&P 500 return over the last 12 months: {num['sp_12m']:+.1%}\n"
             f"VIX, implied volatility of the S&P 500 (percent per year): {num['vix']:.1f}\nRealized volatility of the S&P 500 over 30 days (percent per year): {realized:.1f}\n"
             f"VIX / 3-month VIX: {num['vix_ratio']:.2f}\nSKEW index: {num['skew']:.1f}\n3-month Treasury bill yield (percent per year): {num['tbill']:.2f}\n\n"
             f"[Monthly outlook of an asset manager, published {rep[1]}]\nSummary: {txt['us_summary']}\nKey risks: {txt['key_risks']}")
    return block, num

def sector_block(conn, sector, date):
    m = pd.read_sql("SELECT metric, value, z_score FROM sector_metric_scores WHERE as_of_date = ? AND gics_sector = ?", conn, params=(date, sector), index_col='metric')
    f = lambda k: f"{m.value[k]:.3g} ({m.z_score[k]:+.2f})"
    return (f"\n\n[Sector: {sector}]  (value, z-score among the 11 sectors)\nMomentum 3 / 6 / 12 months, latest month excluded: {f('momentum_3m')} / {f('momentum_6m')} / {f('momentum_12m')}\n"
            f"30-day volatility, per year: {f('volatility_30d')}\nPrice / earnings: {f('pe_trailing')}")

def question(high, low):
    return f"\n\nQuestion: over the next month, will {high} earn a higher or a lower return than {low}?\n{SCALE}"

def ask(client, prompt):
    """Probabilities of the digits 1-5 for the first token, and 30 sampled answers of the same request."""
    c = client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                       max_tokens=1, temperature=1.0, n=30, logprobs=True, top_logprobs=20)
    top = c.choices[0].logprobs.content[0].top_logprobs
    p = {t.token: math.exp(t.logprob) for t in top if t.token in list('12345')}
    return {'probs': [p.get(d, 0.0) for d in '12345'], 'samples': [ch.message.content for ch in c.choices]}

if __name__ == '__main__':
    conn = sqlite3.connect(DB, uri=True)
    dates = [r[0] for r in conn.execute("SELECT DISTINCT as_of_date FROM llm_views ORDER BY 1")]
    if len(sys.argv) > 1 and sys.argv[1] == 'test': dates = dates[:1]
    rows = []
    for d in dates:
        block, num = state(conn, d)
        for name, (high, low) in FACTORS.items():
            rows.append({'date': d, 'kind': 'factor', 'name': name, 'prompt': block + question(high, low), **num})
        for s in SECTORS:
            rows.append({'date': d, 'kind': 'sector', 'name': s, 'prompt': block + sector_block(conn, s, d) + question(f"the {s} sector", "the S&P 500 as a whole"), **num})
    t = pd.DataFrame(rows); print(len(t), 'requests', flush=True); print(t.prompt.iloc[3]); print('...'); print(t.prompt.iloc[-1][-420:], flush=True)
    client = make_client(config.OPENAI_API_KEY)
    def safe(p):
        try: return ask(client, p)
        except Exception as e: return {'error': str(e)[:150]}
    with ThreadPoolExecutor(8) as pool: out = list(pool.map(safe, t.prompt))
    t['answer'] = [json.dumps(o) for o in out]
    print('failed', sum('error' in o for o in out), [o for o in out if 'error' in o][:1])
    for o in out[:3]: print([round(x, 3) for x in o.get('probs', [])], 'sum', round(sum(o.get('probs', [0])), 3), '| samples', pd.Series(o.get('samples', [])).value_counts().to_dict())
    if not (len(sys.argv) > 1 and sys.argv[1] == 'test'): t.to_pickle(HERE + 'premium_results.pkl'); print('saved')
