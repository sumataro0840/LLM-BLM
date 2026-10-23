"""
Small-cap universe (design: 設計_小型株.md). Everything goes to a separate database, data/smallcap.sqlite (the schema
of the main database plus the sc_* tables below); the main database is not touched. The SEC steps need
SEC_USER_AGENT (environment or .env), as for run_events.py.

    python -m smallcap.build population        # XBRL frames: public float of every filer, 2015-2025 -> sc_float, sc_band
    python -m smallcap.build submissions       # EDGAR submissions of every company ever in the band -> sc_companies, sc_filings
    python -m smallcap.build prices            # yfinance: band companies listed now + index and ETFs -> tickers, prices
    python -m smallcap.build select [--n 1000] # random sample of companies with prices and earnings releases -> sc_sample
    python -m smallcap.build releases          # 8-K Item 2.02 EX-99.1 texts of the sample -> earnings_releases
    python -m smallcap.build shares            # XBRL frames: shares outstanding per quarter -> sc_shares
    python -m smallcap.build fundamentals      # XBRL company facts of the sample -> financial_facts
    python -m smallcap.build surprise          # SUE and revenue surprise of every release -> release_surprises
    python -m smallcap.build status

The band of a year is the ranks 1001-3000 of the public float among filers located in the US (a Russell-2000-like
cut made from SEC data, so that companies that later disappear are in it too). Stock reactions are not computed here.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from events import edgar, texts  # noqa: E402
from events.import_h06 import trading_days  # noqa: E402
from events.timing import timing  # noqa: E402
from marketdb.db import events as db_events  # noqa: E402
from marketdb.db.connection import connect, init_db  # noqa: E402
from marketdb.db.market import now_utc, save_prices  # noqa: E402
from marketdb.sources.prices import download_prices  # noqa: E402

DB_PATH = REPO / 'data' / 'smallcap.sqlite'
YEARS = range(2015, 2026)                       # years of the public float (measured at the second fiscal quarter)
BAND = (1001, 3000)                             # ranks of the public float
FLOAT_MIN, FLOAT_MAX = 1e6, 4e12                # values outside are reporting errors (wrong scale)
FIRST, LAST = '2015-01-01', '2026-09-30'        # filings and prices (the main database ends on 2026-09-30)
MAJOR = {'Nasdaq', 'NYSE', 'NYSE American', 'NYSE MKT'}
BENCHMARKS = ['^GSPC', 'IWM', 'IJR', 'SPY', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
FRAMES = 'https://data.sec.gov/api/xbrl/frames/dei/{tag}/{unit}/CY{year}Q{q}I.json'

SC_SCHEMA = """
CREATE TABLE IF NOT EXISTS sc_float (           -- dei:EntityPublicFloat of a fiscal year (one value per company and year)
    cik INTEGER NOT NULL, year INTEGER NOT NULL, entity_name TEXT, loc TEXT,
    float_end TEXT NOT NULL,                    -- the day the float was measured (end of the second fiscal quarter)
    public_float REAL NOT NULL, accn TEXT, valid INTEGER NOT NULL,   -- valid = 0: dropped as a reporting error
    PRIMARY KEY (cik, year));
CREATE TABLE IF NOT EXISTS sc_band (            -- rank of the valid float among US filers in a year
    year INTEGER NOT NULL, cik INTEGER NOT NULL, float_rank INTEGER NOT NULL, in_band INTEGER NOT NULL,
    PRIMARY KEY (year, cik));
CREATE TABLE IF NOT EXISTS sc_companies (       -- from EDGAR submissions (state when fetched)
    cik INTEGER PRIMARY KEY, name TEXT, sic TEXT, sic_description TEXT,
    tickers TEXT, exchanges TEXT,               -- JSON lists as EDGAR shows them now (empty when no longer listed)
    ticker TEXT, exchange TEXT,                 -- the first ticker on a major US exchange (NULL = none now)
    n_earnings INTEGER,                         -- 8-K with Item 2.02 filed 2016-01-01..2026-09-30
    first_filing TEXT, last_filing TEXT,
    last_25 TEXT, last_15 TEXT,                 -- last Form 25 (delisting) / Form 15 (deregistration)
    last_merger TEXT,                           -- last merger proxy, tender offer reply or 8-K Item 5.01 (change in control)
    last_bankruptcy TEXT,                       -- last 8-K with Item 1.03
    fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sc_filings (         -- 8-K filings with Item 2.02 of the companies above
    accession TEXT PRIMARY KEY, cik INTEGER NOT NULL, filing_date TEXT NOT NULL, accepted TEXT NOT NULL,
    primary_doc TEXT);
CREATE TABLE IF NOT EXISTS sc_sample (          -- the companies whose releases are fetched
    cik INTEGER PRIMARY KEY, symbol TEXT NOT NULL, seed INTEGER NOT NULL, drawn_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sc_shares (          -- dei:EntityCommonStockSharesOutstanding (cover page of 10-Q / 10-K)
    cik INTEGER NOT NULL, share_date TEXT NOT NULL, shares REAL NOT NULL, accn TEXT,
    PRIMARY KEY (cik, share_date));
"""


def open_db():
    conn = connect(DB_PATH)
    init_db(conn)
    conn.executescript(SC_SCHEMA)
    return conn


def frame(tag, unit, year, q):
    raw = edgar.get(FRAMES.format(tag=tag, unit=unit, year=year, q=q))
    return json.loads(raw)['data'] if raw else []


# ---------- population ----------

def cmd_population(conn, args):
    rows = []
    for year in YEARS:
        for q in range(1, 5):
            for r in frame('EntityPublicFloat', 'USD', year, q):
                rows.append({'cik': r['cik'], 'year': year, 'entity_name': r.get('entityName'), 'loc': r.get('loc'),
                             'float_end': r['end'], 'public_float': float(r['val']), 'accn': r.get('accn')})
        print(year, sum(1 for r in rows if r['year'] == year), flush=True)
    f = pd.DataFrame(rows).sort_values('float_end').drop_duplicates(['cik', 'year'], keep='last')
    # reporting errors: outside the plausible range, or 200 times away from the company's own median
    med = f[(f.public_float >= FLOAT_MIN) & (f.public_float <= FLOAT_MAX)].groupby('cik').public_float.median()
    ratio = f.public_float / f.cik.map(med)
    f['valid'] = ((f.public_float >= FLOAT_MIN) & (f.public_float <= FLOAT_MAX)
                  & ((ratio < 200) & (ratio > 1 / 200))).astype(int)
    conn.execute('DELETE FROM sc_float')
    f.to_sql('sc_float', conn, if_exists='append', index=False)
    band = f[(f.valid == 1) & f['loc'].fillna('').str.startswith('US')].copy()
    band['float_rank'] = band.groupby('year').public_float.rank(ascending=False, method='first').astype(int)
    band['in_band'] = band.float_rank.between(*BAND).astype(int)
    conn.execute('DELETE FROM sc_band')
    band[['year', 'cik', 'float_rank', 'in_band']].to_sql('sc_band', conn, if_exists='append', index=False)
    conn.commit()
    print(f'{len(f)} company-years ({(f.valid == 0).sum()} not used: below 1 million, above 4 trillion or 200x off); band: '
          f'{band.in_band.sum()} company-years, {band[band.in_band == 1].cik.nunique()} companies')


# ---------- submissions ----------

MERGER_FORMS = {'DEFM14A', 'DEFM14C', 'PREM14A', 'PREM14C', 'SC 14D9'}


def submissions(cik):
    """Company facts and the filings of interest from EDGAR submissions (older pages read when they reach 2014)."""
    sub = json.loads(edgar.get(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json'))
    pages = [sub['filings']['recent']]
    for extra in sub['filings'].get('files', []):
        if extra['filingTo'] >= '2014-01-01':
            pages.append(json.loads(edgar.get(f"https://data.sec.gov/submissions/{extra['name']}")))
    tickers, exchanges = sub.get('tickers') or [], sub.get('exchanges') or []
    major = [(t, e) for t, e in zip(tickers, exchanges) if e in MAJOR]
    info = {'cik': int(cik), 'name': sub.get('name'), 'sic': sub.get('sic'), 'sic_description': sub.get('sicDescription'),
            'tickers': json.dumps(tickers), 'exchanges': json.dumps(exchanges),
            'ticker': major[0][0] if major else None, 'exchange': major[0][1] if major else None,
            'n_earnings': 0, 'first_filing': None, 'last_filing': None, 'last_25': None, 'last_15': None,
            'last_merger': None, 'last_bankruptcy': None, 'fetched_at': now_utc()}
    filings = []
    for p in pages:
        for form, date, acc, items, accepted, primary in zip(p['form'], p['filingDate'], p['accessionNumber'],
                                                            p['items'], p['acceptanceDateTime'], p['primaryDocument']):
            if date < '2014-01-01':
                continue
            info['first_filing'] = min(info['first_filing'] or date, date)
            info['last_filing'] = max(info['last_filing'] or date, date)
            items = items or ''
            last = lambda k: max(info[k] or date, date)
            if form in ('25', '25-NSE'):
                info['last_25'] = last('last_25')
            if form.startswith('15-') or form.startswith('15F'):
                info['last_15'] = last('last_15')
            if form in MERGER_FORMS or (form == '8-K' and '5.01' in items):
                info['last_merger'] = last('last_merger')
            if form == '8-K' and '1.03' in items:
                info['last_bankruptcy'] = last('last_bankruptcy')
            if form == '8-K' and '2.02' in items and FIRST <= date <= LAST:
                filings.append({'accession': acc, 'cik': int(cik), 'filing_date': date, 'accepted': accepted,
                                'primary_doc': primary})
                if date >= '2016-01-01':
                    info['n_earnings'] += 1
    return info, filings


def cmd_submissions(conn, args):
    ciks = [r[0] for r in conn.execute('SELECT DISTINCT cik FROM sc_band WHERE in_band = 1 ORDER BY cik')]
    done = {r[0] for r in conn.execute('SELECT cik FROM sc_companies')}
    todo = [c for c in ciks if c not in done]
    print(f'{len(ciks)} companies ever in the band, {len(todo)} to fetch', flush=True)
    t0 = time.time()

    def one(c):
        try:
            return submissions(c)
        except Exception as e:                  # a company whose history cannot be read is reported and skipped
            print(f'  {c}: {e}', flush=True)
            return None
    with ThreadPoolExecutor(4) as pool:
        for i, res in enumerate(pool.map(one, todo), 1):
            if res:
                info, filings = res
                pd.DataFrame([info]).to_sql('sc_companies', conn, if_exists='append', index=False)
                conn.executemany('INSERT OR IGNORE INTO sc_filings VALUES (?, ?, ?, ?, ?)',
                                 [tuple(f.values()) for f in filings])
            if i % 200 == 0 or i == len(todo):
                conn.commit()
                print(f'  {i}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)


# ---------- prices ----------

def yf_symbol(ticker):
    return ticker.replace('.', '-')


def cmd_prices(conn, args):
    comp = pd.read_sql('SELECT cik, name, ticker FROM sc_companies WHERE ticker IS NOT NULL', conn)
    comp['symbol'] = comp.ticker.map(yf_symbol)
    comp = comp.drop_duplicates('symbol')
    fetched = now_utc()
    conn.executemany('INSERT OR IGNORE INTO tickers (symbol, security, cik, fetched_at) VALUES (?, ?, ?, ?)',
                     [(r.symbol, r.name, int(r.cik), fetched) for r in comp.itertuples()]
                     + [(b, b, None, fetched) for b in BENCHMARKS])
    conn.commit()
    have = {r[0] for r in conn.execute('SELECT DISTINCT symbol FROM prices')}
    symbols = [s for s in BENCHMARKS + sorted(comp.symbol) if s not in have]
    print(f'{len(symbols)} symbols to download', flush=True)
    for k in range(0, len(symbols), 100):
        chunk = symbols[k:k + 100]
        try:
            px = download_prices(chunk, '2014-12-01', LAST)
            save_prices(conn, px)
        except Exception as e:
            print(f'  chunk {k}: {e}', flush=True)
        n = conn.execute('SELECT COUNT(DISTINCT symbol) FROM prices').fetchone()[0]
        print(f'  {min(k + 100, len(symbols))}/{len(symbols)}: {n} symbols with prices', flush=True)
        time.sleep(2)


# ---------- sample ----------

def eligible(conn):
    """Companies in the band at least once, listed now with prices, with at least 4 earnings 8-Ks since 2016."""
    return pd.read_sql("""
        SELECT c.cik, t.symbol, c.n_earnings
        FROM sc_companies c JOIN tickers t ON t.cik = c.cik
        WHERE c.n_earnings >= 4
          AND EXISTS (SELECT 1 FROM prices p WHERE p.symbol = t.symbol)
        ORDER BY c.cik""", conn).drop_duplicates('cik')


def cmd_select(conn, args):
    e = eligible(conn)
    taken = pd.read_sql('SELECT cik FROM sc_sample', conn)
    if len(taken):
        print(f'already drawn: {len(taken)} companies (delete sc_sample to draw again)')
        return
    rng = random.Random(args.seed)
    ciks = sorted(e.cik)
    pick = set(rng.sample(ciks, min(args.n, len(ciks))))
    s = e[e.cik.isin(pick)]
    conn.executemany('INSERT INTO sc_sample VALUES (?, ?, ?, ?)',
                     [(int(r.cik), r.symbol, args.seed, now_utc()) for r in s.itertuples()])
    conn.commit()
    print(f'{len(e)} eligible companies, {len(s)} drawn (seed {args.seed})')


# ---------- releases ----------

def cmd_releases(conn, args):
    sample = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    symbol_of = dict(zip(sample.cik, sample.symbol))
    filings = pd.read_sql('SELECT * FROM sc_filings', conn)
    filings = filings[filings.cik.isin(symbol_of)]
    known = {r[0] for r in conn.execute('SELECT accession FROM earnings_releases')}
    todo = [{'cik': int(r.cik), 'accession': r.accession, 'filing_date': r.filing_date, 'accepted': r.accepted,
             'primary': r.primary_doc} for r in filings.itertuples() if r.accession not in known]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(sample)} companies, {len(filings)} filings, {len(todo)} to fetch', flush=True)
    days = trading_days(conn)
    t0, batch = time.time(), []

    def one(row):
        try:
            return edgar.fetch_text(row)
        except Exception as e:
            print(f"  {row['accession']}: {e}", flush=True)
            return None
    with ThreadPoolExecutor(4) as pool:
        for i, row in enumerate(pool.map(one, todo), 1):
            if row is not None:
                accepted_at, accepted_et, session, event_date = timing(row['accepted'], days)
                sha = n_chars = kind = None
                if row.get('text_path'):
                    body = texts.read(row['text_path'])
                    sha, n_chars, kind = texts.sha256_of(texts.absolute(row['text_path'])), len(body), texts.doc_type(body)
                batch.append({'accession': row['accession'], 'symbol': symbol_of[row['cik']], 'cik': row['cik'],
                              'filing_date': row['filing_date'], 'accepted_at': accepted_at, 'accepted_et': accepted_et,
                              'session': session, 'event_date': event_date, 'document': row.get('document'),
                              'doc_type': kind, 'how_chosen': row['how_chosen'], 'text_path': row.get('text_path'),
                              'text_sha256': sha, 'n_chars': n_chars})
            if len(batch) >= 200 or i == len(todo):
                if batch:
                    db_events.save_releases(conn, pd.DataFrame(batch))
                batch = []
                print(f'  {i}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)


# ---------- financial statements and surprises ----------

def cmd_fundamentals(conn, args):
    """XBRL company facts of the sample (as first filed, as for the S&P500), for the earnings surprises."""
    from marketdb.sources.edgar import load_company_facts
    from marketdb.db.fundamentals import save_facts
    todo = conn.execute('SELECT t.symbol, t.cik FROM tickers t JOIN sc_sample s ON s.symbol = t.symbol '
                        'WHERE t.fundamentals_fetched_at IS NULL ORDER BY t.symbol').fetchall()
    print(f'{len(todo)} companies to fetch', flush=True)
    t0 = time.time()
    for i, (symbol, cik) in enumerate(todo, 1):
        try:
            save_facts(conn, symbol, load_company_facts(int(cik), '2013-01-01'))
        except Exception as e:
            print(f'  {symbol}: {str(e)[:80]}', flush=True)
        time.sleep(0.15)               # SEC asks for at most 10 requests per second
        if i % 100 == 0 or i == len(todo):
            print(f'  {i}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)


def cmd_surprise(conn, args):
    import run_events
    run_events.cmd_surprise(conn, argparse.Namespace(start=None, end=None))


# ---------- shares ----------

def cmd_shares(conn, args):
    rows = []
    for year in range(2015, 2027):
        for q in range(1, 5):
            if (year, q) > (2026, 3):
                break
            for r in frame('EntityCommonStockSharesOutstanding', 'shares', year, q):
                rows.append((r['cik'], r['end'], float(r['val']), r.get('accn')))
    conn.executemany('INSERT OR REPLACE INTO sc_shares VALUES (?, ?, ?, ?)', rows)
    conn.commit()
    print(f'{len(rows)} values')


# ---------- status ----------

def cmd_status(conn, args):
    for q in ['SELECT COUNT(*) FROM sc_float', 'SELECT COUNT(*), COUNT(DISTINCT cik) FROM sc_band WHERE in_band = 1',
              'SELECT COUNT(*), SUM(ticker IS NOT NULL) FROM sc_companies', 'SELECT COUNT(*) FROM sc_filings',
              'SELECT COUNT(DISTINCT symbol) FROM prices', 'SELECT COUNT(*) FROM sc_sample',
              'SELECT COUNT(*), SUM(text_path IS NOT NULL), COUNT(DISTINCT cik) FROM earnings_releases',
              'SELECT COUNT(*) FROM sc_shares']:
        print(q, conn.execute(q).fetchone())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ['population', 'submissions', 'prices', 'shares', 'fundamentals', 'surprise', 'status']:
        sub.add_parser(name)
    s = sub.add_parser('select')
    s.add_argument('--n', type=int, default=1000)
    s.add_argument('--seed', type=int, default=0)
    r = sub.add_parser('releases')
    r.add_argument('--limit', type=int)
    args = p.parse_args()
    conn = open_db()
    globals()['cmd_' + args.cmd](conn, args)


if __name__ == '__main__':
    main()
