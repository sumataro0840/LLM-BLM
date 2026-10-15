"""
Fetch financial statements of every S&P500 member in the database from SEC EDGAR.

    python marketdb/build_fundamentals.py

Steps:
  1. company ids (CIK): Wikipedia for current members, Wikidata for the others  -> tickers.cik
  2. reported values for periods ending on or after FUNDAMENTALS_START_DATE     -> financial_items, financial_facts
Tickers that already have facts are skipped, so the script can be re-run after an interruption.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marketdb import config
from marketdb.sources.edgar import choose_cik, load_company_facts, load_wikipedia_ciks, lookup_wikidata_ciks
from marketdb.sources.universe import to_yf_symbol
from marketdb.db.connection import connect, init_db
from marketdb.db.fundamentals import ensure_ticker_columns, save_ciks, save_facts

# Ids that neither Wikipedia nor Wikidata gives (checked on EDGAR by company name)
MANUAL_CIKS = {
    'DAY': 1725057,   # Dayforce, Inc. (formerly Ceridian HCM Holding, ticker CDAY)
    'WBD': 1437107,   # Warner Bros. Discovery, Inc.
}
# Earlier ids of companies that changed registrant (facts of all ids are stored under the ticker)
PREVIOUS_CIKS = {
    'XOM': [34088],     # Exxon Mobil Corp before ExxonMobil Holdings Corp (2026-07)
    'BLK': [1364742],   # BlackRock, Inc. (now BlackRock Finance) before the new parent (2024-10)
    'BG': [1144519],    # Bunge Ltd before Bunge Global SA (2023-11)
    'FERG': [1832433],  # Ferguson plc before Ferguson Enterprises Inc. (2024-08)
    'SKYD': [813828],   # Paramount Global before Paramount Skydance / Skydance Corp (2025-08)
    'GOOGL': [1288776], # Google Inc. before Alphabet Inc. (2015-10)
    'GOOG': [1288776],
    # yfinance continues the prices of the earlier company under the new ticker, so its statements are needed too
    'VTRS': [69499, 1623613],   # Mylan Inc., Mylan N.V. before Viatris (2020-11)
    'SW': [1636023, 1732845],   # WestRock (2015-07 .. 2018-11, 2018-11 .. 2024-07) before Smurfit Westrock
}
# Same company as another ticker in the table: not fetched twice
SAME_COMPANY = {'CDAY': 'DAY'}

START = config.FUNDAMENTALS_START_DATE


def step_ciks(conn):
    rows = conn.execute("SELECT symbol, security FROM tickers WHERE symbol IN (SELECT symbol FROM index_membership) AND cik IS NULL").fetchall()
    rows = [(s, n) for s, n in rows if s not in SAME_COMPANY]
    if not rows:
        print("[1] cik: nothing to look up")
        return
    wikipedia = {to_yf_symbol(s): c for s, c in load_wikipedia_ciks().items()}
    found = {s: wikipedia[s] for s, _ in rows if s in wikipedia}
    found.update({s: MANUAL_CIKS[s] for s, _ in rows if s in MANUAL_CIKS and s not in found})
    save_ciks(conn, found)
    rest = [(s, n) for s, n in rows if s not in found]
    if rest:
        candidates = {}
        for attempt in range(3):   # Wikidata may answer 429 (it allows about one query per minute)
            try:
                candidates = lookup_wikidata_ciks([s.replace('-', '.') for s, _ in rest])
                break
            except Exception as e:
                print(f"    Wikidata lookup failed ({str(e)[:60]}), retrying in 70 seconds", flush=True)
                time.sleep(70)
        for symbol, security in rest:
            options = candidates.get(symbol.replace('-', '.'))
            if options:
                cik = choose_cik(symbol, security, options)
                if cik:
                    found[symbol] = cik
                time.sleep(0.2)
    save_ciks(conn, found)
    missing = [s for s, _ in rows if s not in found]
    print(f"[1] cik: {len(found)} found (Wikipedia {sum(s in wikipedia for s in found)}) | not found: {missing}")


def step_facts(conn):
    todo = conn.execute("""
        SELECT symbol, cik FROM tickers
        WHERE cik IS NOT NULL AND fundamentals_fetched_at IS NULL ORDER BY symbol""").fetchall()

    def fetch(row):
        symbol, cik = row
        rows, errors = [], []
        for one in [cik, *PREVIOUS_CIKS.get(symbol, [])]:
            try:
                rows += load_company_facts(one, START)
            except Exception as e:
                errors.append(f"{one}: {str(e)[:60]}")
            time.sleep(0.15)   # SEC asks for at most 10 requests per second
        rows.sort(key=lambda r: r[8])   # earliest filing first: it wins when two ids report the same period
        return symbol, rows, errors

    failed = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, (symbol, rows, errors) in enumerate(pool.map(fetch, todo), 1):
            if rows:
                save_facts(conn, symbol, rows)
            if errors or not rows:
                failed.append((symbol, errors or 'no facts'))
            if i % 50 == 0:
                print(f"    {i}/{len(todo)}", flush=True)
    n_facts = conn.execute("SELECT COUNT(*) FROM financial_facts").fetchone()[0]
    n_items = conn.execute("SELECT COUNT(*) FROM financial_items").fetchone()[0]
    print(f"[2] financial_facts: {n_facts:,} rows, {n_items:,} items | fetched now: {len(todo)} tickers | problems: {failed}")


def report(conn):
    print("[check] tickers without financial facts:")
    print("   ", [r[0] for r in conn.execute("""
        SELECT symbol FROM tickers t WHERE symbol IN (SELECT symbol FROM index_membership)
        AND NOT EXISTS (SELECT 1 FROM financial_facts f WHERE f.symbol = t.symbol) ORDER BY symbol""")])
    print("[check] tickers whose facts start late or end early (company id may have changed):")
    for symbol, first, last, n in conn.execute("""
        SELECT symbol, MIN(filed_date), MAX(period_end), COUNT(*) FROM financial_facts
        GROUP BY symbol HAVING MIN(filed_date) > date(?, '+15 months') OR COUNT(*) < 500 ORDER BY symbol""", (START,)):
        print(f"    {symbol}: first filed {first}, last period {last}, {n} facts")


def main():
    conn = connect(config.DB_PATH)
    init_db(conn)
    ensure_ticker_columns(conn)
    print(f"DB: {config.DB_PATH} | periods ending on or after {START}", flush=True)
    step_ciks(conn)
    step_facts(conn)
    report(conn)


if __name__ == "__main__":
    main()
