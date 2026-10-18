"""Does the LLM discount fundamentals that mislead in an industry?  with / n-a / n-a repeated, per group."""
import sys, json, sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/kazuki/develop/工房/LLM-BLM'); sys.path.insert(0, '/Users/kazuki/develop/工房/LLM-BLM_実験')
import marketdb.transform.stock as st
from marketdb.db.sector import load_member_caps
from marketdb.db.stock import load_quarters
from llm import config
from llm.client import make_client, sample_expected_returns

DATES = ['2024-06-28', '2024-12-31', '2025-06-30', '2025-12-31', '2026-05-29']
REIT4 = ['pe_trailing', 'net_income_growth_yoy', 'net_margin', 'roe']
BANKS = st.INDUSTRY_EXCLUSIONS[0][0]; REITS = st.INDUSTRY_EXCLUSIONS[2][0]
GROUPS = {  # name -> (selector on members, metrics hidden in the n/a condition)
    'reit': (lambda m: m.gics_sub_industry.isin(REITS), REIT4),
    'bank': (lambda m: m.gics_sub_industry.isin(BANKS), ['net_margin', 'debt_to_equity']),
    'industrials': (lambda m: m.gics_sector == 'Industrials', REIT4),   # control: metrics that do mean something
}
st.INDUSTRY_EXCLUSIONS = []      # raw values for every industry (nothing is written to the database)
PRICE = [('momentum_12m', '12-month momentum, latest month excluded'), ('return_1m', 'Return over the latest month'),
         ('relative_momentum_12m', "12-month momentum minus the sector's"), ('volatility_30d', '30-day volatility, per year'),
         ('beta_1y', 'Beta to the S&P 500')]
FUND = [('pe_trailing', 'Price / earnings, last 4 quarters'), ('pb', 'Price / book'), ('revenue_growth_yoy', 'Revenue growth, year on year'),
        ('net_income_growth_yoy', 'Net income growth, year on year'), ('net_margin', 'Net margin'), ('roe', 'Return on equity'),
        ('debt_to_equity', 'Long-term debt / equity')]
SYSTEM = ("You are a financial analyst. Using only the information given below, predict the average daily return of the stock over the next month.\n"
          "Every return is a decimal fraction, not a percent: 0.01 means +1%.\n"
          "Percentile = position among the stocks of the same sector (0 = lowest, 100 = highest).\n"
          "\"n/a\" means the value is not available.\n"
          "Answer the expected average daily return as a decimal fraction (answer 0.001 for +0.1% per day, never 0.1).")

def fmt(v, p):
    if pd.isna(v): return 'n/a'
    return f"{float(f'{v:.3g}')}" + ('' if pd.isna(p) else f" ({p:.0f})")

def build(conn, quarters):
    tickers = pd.read_sql("SELECT symbol, security FROM tickers", conn, index_col='symbol').security
    rows = []
    for date in DATES:
        members = load_member_caps(conn, date)
        price = pd.read_sql("SELECT symbol, metric, value FROM stock_metrics WHERE as_of_date = ?", conn, params=(date,)).pivot(index='symbol', columns='metric', values='value')
        fund = st.fundamental_metrics(quarters, members.market_cap, members.gics_sub_industry, date, 150)
        table = members.join(price[[m for m, _ in PRICE]]).join(fund)
        cols = [m for m, _ in PRICE + FUND]
        pct = table.groupby('gics_sector')[cols].transform(lambda s: s.rank(pct=True) * 100 if s.notna().sum() >= 5 else s * np.nan)
        for group, (select, hidden) in GROUPS.items():
            for symbol, r in table[select(table)].iterrows():
                if pd.isna(r.get('fundamentals_age_days')): continue
                def prompt(hide):
                    lines = ["[Stock]", f"Ticker: {symbol}", f"Security: {tickers[symbol]}", f"Sector: {r.gics_sector}", f"Sub-Industry: {r.gics_sub_industry}", "",
                             "[Price behaviour]  (value, percentile in sector)"]
                    lines += [f"{label}: {fmt(r[m], pct.loc[symbol, m])}" for m, label in PRICE]
                    end = (pd.Timestamp(date) - pd.Timedelta(days=int(r.fundamentals_age_days))).date(); filed = (pd.Timestamp(date) - pd.Timedelta(days=int(r.filing_age_days))).date()
                    lines += ["", f"[Fundamentals]  quarter ended {end}, filed {filed}  (value, percentile in sector)"]
                    lines += [f"{label}: {'n/a' if m in hide else fmt(r[m], pct.loc[symbol, m])}" for m, label in FUND]
                    return "\n".join(lines)
                base = {'date': date, 'group': group, 'symbol': symbol, **{m: r[m] for m in cols}, **{m + '_pct': pct.loc[symbol, m] for m in cols}}
                for cond, hide in [('with', []), ('na', hidden), ('na2', hidden)]:
                    rows.append({**base, 'cond': cond, 'prompt': prompt(hide)})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    conn = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
    tasks = build(conn, load_quarters(conn))
    print(tasks.groupby(['group', 'cond']).size().unstack().to_string())
    print(tasks[(tasks.group == 'reit') & (tasks.cond == 'with')].prompt.iloc[0]); print('---'); print(tasks[(tasks.group == 'reit') & (tasks.cond == 'na')].prompt.iloc[0])
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        client = make_client(config.OPENAI_API_KEY)
        def ask(p):
            try: return sample_expected_returns(client, config.OPENAI_MODEL, SYSTEM, p, 30, 1.0)[0]
            except Exception as e: return str(e)[:100]
        with ThreadPoolExecutor(8) as pool: answers = list(pool.map(ask, tasks.prompt))
        tasks['answers'] = [json.dumps(a) for a in answers]
        tasks.to_pickle('results.pkl'); print('saved', len(tasks), 'failed', sum(isinstance(a, str) for a in answers))
