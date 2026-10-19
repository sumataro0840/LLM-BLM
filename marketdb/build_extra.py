"""
Fetch the additional data of 設計_追加データ.md (stage 1) into the database.

    python marketdb/build_extra.py filings     # every SEC filing of the companies in tickers (data.sec.gov)
    python marketdb/build_extra.py volumes     # daily volume (yfinance)
    python marketdb/build_extra.py factors     # Fama-French 5 factors and momentum, daily and monthly
    python marketdb/build_extra.py macro       # FRED series
    python marketdb/build_extra.py short       # FINRA daily short-sale volume (from 2019)
    python marketdb/build_extra.py pageviews   # Wikipedia page views (from 2015-07)
    python marketdb/build_extra.py insiders    # SEC insider transactions (needs SEC_USER_AGENT)
    python marketdb/build_extra.py 13f         # SEC 13F holdings per CUSIP (needs SEC_USER_AGENT)
    python marketdb/build_extra.py cusips      # CUSIP -> ticker of the largest holdings (OpenFIGI)
    python marketdb/build_extra.py all

What is already stored is skipped, so a step can be run again after an interruption.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marketdb import config
from marketdb.db import extra as db
from marketdb.db.connection import connect, init_db
from marketdb.db.market import load_trading_days
from marketdb.sources import extra as src

FIRST = '2014-01-01'
LAST = '2026-09-30'


def symbols_with_prices(conn):
    return [r[0] for r in conn.execute('SELECT DISTINCT symbol FROM prices ORDER BY symbol')]


def step_filings(conn):
    from marketdb.build_fundamentals import PREVIOUS_CIKS
    tickers = pd.read_sql('SELECT symbol, cik FROM tickers WHERE cik IS NOT NULL', conn)
    done = db.done(conn, 'filings', 'symbol')
    todo = tickers[~tickers.symbol.isin(done)]
    print(f'filings: {len(todo)} of {len(tickers)} companies to fetch', flush=True)

    def one(row):
        frames = []
        for cik in [row.cik, *PREVIOUS_CIKS.get(row.symbol, [])]:
            try:
                frames.append(src.filings_of(cik, FIRST))
            except Exception as e:
                print(f'  {row.symbol} {cik}: {str(e)[:80]}', flush=True)
        df = pd.concat(frames) if frames else pd.DataFrame()
        return row.symbol, df

    with ThreadPoolExecutor(4) as pool:
        for k, (symbol, df) in enumerate(pool.map(one, todo.itertuples()), 1):
            if not df.empty:
                db.save(conn, 'filings', df.assign(symbol=symbol))
            if k % 50 == 0:
                print(f'  {k}/{len(todo)}', flush=True)


def step_volumes(conn):
    import yfinance as yf
    symbols = sorted(set(symbols_with_prices(conn)) - db.done(conn, 'volumes', 'symbol'))
    print(f'volumes: {len(symbols)} symbols', flush=True)
    for i in range(0, len(symbols), 100):
        part = symbols[i:i + 100]
        data = yf.download(part, start=FIRST, end='2026-10-01', auto_adjust=False, progress=False)['Volume']
        long = data.stack().rename('volume').reset_index()
        long.columns = ['date', 'symbol', 'volume']
        long['date'] = pd.to_datetime(long.date).dt.strftime('%Y-%m-%d')
        db.save(conn, 'volumes', long[long.volume.notna()])
        print(f'  {i + len(part)}/{len(symbols)}', flush=True)


def step_factors(conn):
    for name, freq in src.FRENCH_SETS:
        df = src.french(name, freq)
        n = db.save(conn, 'factor_returns', df)
        print(f'factors {name}: {len(df)} values, {n} new, {df.date.min()} .. {df.date.max()}', flush=True)


def step_macro(conn):
    for sid in src.FRED_SERIES:
        df = src.fred(sid)
        n = db.save(conn, 'macro_series', df)
        print(f'macro {sid}: {df.value.notna().sum()} values, {n} new, {df.date.min()} .. {df.date.max()}', flush=True)


def step_short(conn):
    symbols = set(symbols_with_prices(conn))
    days = [d for d in load_trading_days(conn, '2019-01-01', LAST)]
    done = db.done(conn, 'short_volume', 'date')
    todo = [d for d in days if d not in done]
    print(f'short volume: {len(todo)} of {len(days)} days to fetch', flush=True)
    missing = []
    with ThreadPoolExecutor(4) as pool:
        for k, (day, df) in enumerate(pool.map(lambda d: (d, src.short_volume_day(d, symbols)), todo), 1):
            if df is None:
                missing.append(day)
            else:
                db.save(conn, 'short_volume', df)
            if k % 100 == 0:
                print(f'  {k}/{len(todo)}', flush=True)
    print(f'  days without a file: {len(missing)} ({missing[:5]} ...)', flush=True)


def step_pageviews(conn):
    tickers = pd.read_sql('SELECT symbol, cik FROM tickers WHERE cik IS NOT NULL', conn)
    titles = src.wikipedia_articles(sorted(set(int(c) for c in tickers.cik)))
    tickers['article'] = tickers.cik.map(lambda c: titles.get(int(c)))
    print(f'pageviews: {tickers.article.notna().sum()} of {len(tickers)} companies have an article', flush=True)
    done = db.done(conn, 'pageviews', 'symbol')
    todo = tickers[tickers.article.notna() & ~tickers.symbol.isin(done)]
    def one(r):
        try:
            return r.symbol, src.pageviews(r.article)
        except Exception as e:                      # one article failing does not stop the others
            print(f'  {r.symbol} ({r.article}): {str(e)[:80]}', flush=True)
            return r.symbol, None

    with ThreadPoolExecutor(1) as pool:
        for k, (symbol, df) in enumerate(pool.map(one, todo.itertuples()), 1):
            if df is not None:
                db.save(conn, 'pageviews', df.assign(symbol=symbol))
            if k % 100 == 0:
                print(f'  {k}/{len(todo)}', flush=True)


def step_insiders(conn):
    """SEC Insider Transactions Data Sets, one file per quarter from 2014 Q1 (needs SEC_USER_AGENT)."""
    from marketdb.build_fundamentals import PREVIOUS_CIKS
    tickers = pd.read_sql('SELECT symbol, cik FROM tickers WHERE cik IS NOT NULL', conn)
    of_cik = {}
    for r in tickers.itertuples():
        for cik in [int(r.cik), *PREVIOUS_CIKS.get(r.symbol, [])]:
            of_cik.setdefault(cik, []).append(r.symbol)
    done = {r[0] for r in conn.execute("SELECT DISTINCT substr(filing_date, 1, 4) || 'q' || "
                                       "((CAST(substr(filing_date, 6, 2) AS INTEGER) + 2) / 3) FROM insider_transactions")}
    for year in range(2014, 2027):
        for q in range(1, 5):
            key = f'{year}q{q}'
            if key in done or (year, q) > (2026, 3):
                continue
            df = src.insider_quarter(year, q, of_cik)
            if df is None:
                print(f'insiders {key}: no file', flush=True)
                continue
            n = db.save(conn, 'insider_transactions', df)
            print(f'insiders {key}: {len(df)} rows, {n} new', flush=True)


def step_13f(conn):
    """SEC Form 13F data sets (needs SEC_USER_AGENT); one file at a time, files already stored are skipped."""
    links = src.thirteen_f_files()
    done = db.done(conn, 'holdings_13f_parts', 'source_file')
    todo = [l for l in links if l.rsplit('/', 1)[1] not in done]
    print(f'13F: {len(todo)} of {len(links)} files to fetch', flush=True)
    for link in todo:
        df = src.thirteen_f_parts(link)
        db.save(conn, 'holdings_13f_parts', df)
        print(f'  {link.rsplit("/", 1)[1]}: {len(df)} CUSIP-periods, periods {sorted(df.period.unique())[-3:]}', flush=True)


def step_cusips(conn, top=1500):
    """Map the CUSIPs of the largest holdings (top `top` by value in each period) to tickers with OpenFIGI."""
    h = pd.read_sql('SELECT cusip, period, value FROM holdings_13f', conn)
    h['rank'] = h.groupby('period').value.rank(ascending=False)
    cand = sorted(set(h[h['rank'] <= top].cusip) - db.done(conn, 'cusip_map', 'cusip'))
    print(f'CUSIPs to map: {len(cand)}', flush=True)
    symbols = {r[0] for r in conn.execute('SELECT symbol FROM tickers')}
    for i in range(0, len(cand), 200):
        part = cand[i:i + 200]
        found = src.openfigi_tickers(part)
        rows = []
        for c in part:
            ticker, name = found.get(c, (None, None))
            sym = ticker.replace('/', '-').replace('.', '-') if ticker else None
            rows.append({'cusip': c, 'symbol': sym if sym in symbols else None, 'figi_name': name, 'figi_ticker': ticker})
        db.save(conn, 'cusip_map', pd.DataFrame(rows))
        print(f'  {i + len(part)}/{len(cand)}', flush=True)


STEPS = {'filings': step_filings, 'volumes': step_volumes, 'factors': step_factors, 'macro': step_macro,
         'short': step_short, 'pageviews': step_pageviews, 'insiders': step_insiders, '13f': step_13f,
         'cusips': step_cusips}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    names = sys.argv[1:]
    if not names or any(n not in STEPS and n != 'all' for n in names):
        print(__doc__)
        return
    conn = connect(config.DB_PATH)
    conn.execute('PRAGMA busy_timeout = 60000')
    init_db(conn)
    failed = []
    for name in (list(STEPS) if 'all' in names else names):
        try:
            STEPS[name](conn)
        except Exception as e:                      # one source failing does not stop the others
            print(f'{name} failed: {str(e)[:200]}', flush=True)
            failed.append(name)
    if failed:
        raise SystemExit(f'failed: {failed}')


if __name__ == '__main__':
    main()
