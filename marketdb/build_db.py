"""
Build the SQLite database for the study period (DB_START_DATE .. DB_END_DATE in .env).

    python marketdb/build_db.py

Steps (each one is skipped when its table already has rows, so stored data is never refetched):
  1. S&P500 constituent changes (Wikipedia)          -> index_changes
  2. membership periods and tickers                  -> index_membership, tickers
  3. daily prices of every member and ^GSPC          -> prices  (no data -> tickers.excluded_reason)
  4. shares outstanding on the current split basis   -> shares_outstanding
     yfinance has them from about late 2015; earlier dates (and tickers yfinance does not know) come from
     the financial statements on SEC EDGAR, which are fetched first (marketdb/build_fundamentals.py)
  5-6. members without prices or shares for most of their membership are excluded
     (tickers.excluded_reason) and their prices / shares are deleted
  7. sector of stocks that left the index (old Wikipedia revisions) -> tickers.gics_*
  8. indices listed in INDEX_TICKERS that are not stored yet           -> prices
  9. sector ETFs listed in SECTOR_ETFS (from ETF_START_DATE)           -> tickers, prices

Tickers are stored as the latest yfinance tickers ('BRK-B', not 'BRK.B'): old tickers often
fail on yfinance or now point to another company.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marketdb import config
from marketdb.build_fundamentals import SAME_COMPANY, step_ciks, step_facts
from marketdb.sources.prices import download_prices
from marketdb.sources.shares import class_ratios, download_shares, download_splits
from marketdb.sources.universe import load_index_changes, load_sp500_table, load_sp500_table_at, to_yf_symbol
from marketdb.transform.membership import build_membership
from marketdb.transform.shares import clip_to_period, extend_back, to_current_split_basis
from marketdb.db.connection import connect, init_db
from marketdb.db.fundamentals import ensure_ticker_columns, load_reported_shares
from marketdb.db.index import save_index_changes, save_membership
from marketdb.db.market import fill_gics, mark_excluded, missing_tickers, save_prices, save_shares, save_tickers

# Tickers that changed after the change was recorded on Wikipedia (old -> latest)
TICKER_ALIASES = {
    'SATS': 'ECHO',  # EchoStar: added on 2026-03-23 as SATS, now listed as ECHO
    'FLT': 'CPAY',   # Corpay: added on 2018-06-20 as FleetCor (FLT), renamed in 2024
    'RE': 'EG',      # Everest Group: added on 2017-06-19 as Everest Re (RE), renamed in 2023
}
INDEX_TICKERS = [
    ('^GSPC', 'S&P 500 Index'),
    ('^IRX', '13 Week Treasury Bill Yield (percent per year, used as the risk-free rate)'),
    ('^VIX', 'CBOE Volatility Index (implied volatility of the S&P 500 over 30 days, percent per year)'),
    ('^VIX3M', 'CBOE 3-Month Volatility Index (implied volatility over 3 months, percent per year)'),
    ('^SKEW', 'CBOE SKEW Index (price of protection against a large fall of the S&P 500)'),
]
# One ETF per GICS sector (Select Sector SPDR); stored in tickers with the sector it tracks
SECTOR_ETFS = [
    ('XLC', 'Communication Services'), ('XLY', 'Consumer Discretionary'), ('XLP', 'Consumer Staples'),
    ('XLE', 'Energy'), ('XLF', 'Financials'), ('XLV', 'Health Care'), ('XLI', 'Industrials'),
    ('XLK', 'Information Technology'), ('XLB', 'Materials'), ('XLRE', 'Real Estate'), ('XLU', 'Utilities'),
]
# Companies with several share classes in the index: yfinance reports the company-wide share count
# for each class, so it is split by the current class ratio (otherwise the company is counted twice)
MULTI_CLASS = [('GOOGL', 'GOOG'), ('FOXA', 'FOX'), ('NWSA', 'NWS')]
# EDGAR gives about one value per quarter: the outlier check compares a value with those of the next quarters
EDGAR_WINDOW_DAYS = 100

START, END = config.DB_START_DATE, config.DB_END_DATE


def is_empty(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def step_index_changes(conn):
    changes = load_index_changes()
    for col in ['added_symbol', 'removed_symbol']:
        changes[col] = changes[col].map(lambda s: to_yf_symbol(s, TICKER_ALIASES) if s else None)
    save_index_changes(conn, changes)
    print(f"[1] index_changes: {len(changes)} rows")


def step_membership(conn):
    current = load_sp500_table()
    current['symbol'] = current['Symbol'].map(lambda s: to_yf_symbol(s, TICKER_ALIASES))
    changes = pd.read_sql("SELECT date, added_symbol, added_name, removed_symbol, removed_name FROM index_changes", conn)
    periods, issues = build_membership(current['symbol'], changes, START, END)

    names = {}
    for _, r in changes.iterrows():
        for s, n in [(r['added_symbol'], r['added_name']), (r['removed_symbol'], r['removed_name'])]:
            if isinstance(s, str):
                names[s] = n
    gics = {r['symbol']: (r['Security'], r['GICS Sector'], r['GICS Sub-Industry']) for _, r in current.iterrows()}
    symbols = sorted({s for s, _, _ in periods})
    save_tickers(conn, [(s, *gics[s]) if s in gics else (s, names.get(s, s), None, None) for s in symbols])
    save_tickers(conn, [(s, name, None, None) for s, name in INDEX_TICKERS])
    save_membership(conn, periods)

    n_start = sum(1 for _, st, en in periods if st is None or st <= START)
    print(f"[2] tickers: {len(symbols)} stocks + {len(INDEX_TICKERS)} index | membership periods: {len(periods)} "
          f"| members on {START}: {n_start}")
    for issue in issues:
        print(f"    ! {issue}  -> add it to TICKER_ALIASES")


def step_prices(conn):
    symbols = [r[0] for r in conn.execute("SELECT symbol FROM tickers ORDER BY symbol")]
    prices = download_prices(symbols, START, END)
    save_prices(conn, prices)
    got = set(prices['Adj Close'].dropna(axis=1, how='all').columns)
    missing = [s for s in symbols if s not in got]
    mark_excluded(conn, missing, 'no price data on yfinance (delisted)')
    print(f"[3] prices: {len(got)} tickers, {conn.execute('SELECT COUNT(*) FROM prices').fetchone()[0]} rows "
          f"| excluded (no data): {len(missing)} {missing}")


def step_shares(conn):
    # the financial statements first: their share counts fill the dates yfinance does not have
    ensure_ticker_columns(conn)
    step_ciks(conn)
    step_facts(conn)

    symbols = [r[0] for r in conn.execute(
        "SELECT symbol FROM tickers WHERE excluded_reason IS NULL AND symbol IN (SELECT symbol FROM index_membership) ORDER BY symbol")]
    # one year before START so that a value exists on the first day
    since = (pd.Timestamp(START) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    until = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    reported = load_reported_shares(conn, since)
    ratio = {s: r for classes in MULTI_CLASS for s, r in class_ratios(classes).items()}

    def fetch(symbol):
        splits = download_splits(symbol)
        try:
            yahoo = to_current_split_basis(download_shares(symbol, since, until), splits)
        except Exception:
            yahoo = pd.Series(dtype='int64')
        edgar = to_current_split_basis(reported.get(SAME_COMPANY.get(symbol, symbol), pd.Series(dtype=float)),
                                       splits, window_days=EDGAR_WINDOW_DAYS)
        if symbol in ratio:
            yahoo, edgar = [(s * ratio[symbol]).round().astype('int64') for s in (yahoo, edgar)]
        return symbol, yahoo, edgar

    sources = {'yfinance': [], 'yfinance + EDGAR before it': [], 'EDGAR only': [], 'none': []}
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (symbol, yahoo, edgar) in enumerate(pool.map(fetch, symbols), 1):
            shares = clip_to_period(extend_back(yahoo, edgar), START, END)
            if shares.empty:
                source = 'none'
            elif yahoo.empty:
                source = 'EDGAR only'
            else:
                source = 'yfinance' if shares.index[0] >= yahoo.index[0] else 'yfinance + EDGAR before it'
            sources[source].append(symbol)
            results.append((symbol, shares))
            if i % 100 == 0:
                print(f"    {i}/{len(symbols)}")
    # saved only when every ticker is done: a run that stops halfway leaves the table empty, so the next run
    # does this step again (otherwise the next steps would exclude the tickers that were not reached)
    for symbol, shares in results:
        if not shares.empty:
            save_shares(conn, symbol, shares)
    print(f"    share classes split by: { {k: round(v, 3) for k, v in ratio.items()} }")
    print(f"[4] shares_outstanding: {conn.execute('SELECT COUNT(*) FROM shares_outstanding').fetchone()[0]} rows | "
          + ' | '.join(f"{k}: {len(v)}" for k, v in sources.items()))
    print(f"    EDGAR only: {sources['EDGAR only']}")
    print(f"    none: {sources['none']}")


def _exclude(conn, symbols, reason):
    """Mark as excluded and delete their prices and shares."""
    if symbols:
        mark_excluded(conn, symbols, reason)
        marks = ','.join('?' * len(symbols))
        conn.execute(f"DELETE FROM prices WHERE symbol IN ({marks})", symbols)
        conn.execute(f"DELETE FROM shares_outstanding WHERE symbol IN ({marks})", symbols)
        conn.commit()


def exclude_without_member_prices(conn, min_coverage=0.5):
    """Drop membership periods whose prices cover less than min_coverage of their days, and exclude members
    left without any period. A delisted ticker can be reused by another security (FOXA: 21st Century Fox
    until 2019, then Fox Corp), so prices may exist only for the later company's period."""
    rows = conn.execute("""
        SELECT m.membership_id, m.symbol,
               (SELECT COUNT(*) FROM prices p WHERE p.symbol = m.symbol
                  AND (m.start_date IS NULL OR p.date >= m.start_date) AND (m.end_date IS NULL OR p.date < m.end_date)) AS n,
               (SELECT COUNT(*) FROM prices g WHERE g.symbol = '^GSPC'
                  AND (m.start_date IS NULL OR g.date >= m.start_date) AND (m.end_date IS NULL OR g.date < m.end_date)) AS days
        FROM index_membership m JOIN tickers t USING (symbol)
        WHERE t.excluded_reason IS NULL""").fetchall()
    uncovered = [(i, s) for i, s, n, days in rows if days and n < days * min_coverage]
    covered = {s for i, s, n, days in rows if (i, s) not in uncovered}
    symbols = sorted({s for _, s in uncovered} - covered)
    dropped = sorted({s for _, s in uncovered} & covered)
    conn.executemany("DELETE FROM index_membership WHERE membership_id = ?",
                     [(i,) for i, s in uncovered if s in covered])
    conn.commit()
    _exclude(conn, symbols, 'no price data during membership (ticker reused by another security)')
    print(f"[5] excluded (no prices during membership): {symbols}")
    print(f"    periods without prices dropped (ticker reused, the later company is kept): {dropped}")


def exclude_without_shares(conn, max_missing=0.5):
    """Exclude members that have no shares report for more than max_missing of their membership days
    (renamed tickers often keep only the history after the rename). Their prices and shares are deleted."""
    rows = conn.execute("""
        SELECT m.symbol,
               SUM(NOT EXISTS (SELECT 1 FROM shares_outstanding s WHERE s.symbol = p.symbol AND s.date <= p.date)) AS missing,
               COUNT(*) AS days
        FROM index_membership m JOIN tickers t USING (symbol)
        JOIN prices p ON p.symbol = m.symbol
         AND (m.start_date IS NULL OR p.date >= m.start_date) AND (m.end_date IS NULL OR p.date < m.end_date)
        WHERE t.excluded_reason IS NULL
        GROUP BY m.symbol""").fetchall()
    symbols = [s for s, missing, days in rows if missing > days * max_missing]
    _exclude(conn, symbols, 'no shares history on yfinance or EDGAR')
    print(f"[6] excluded (no shares history): {symbols}")


def step_index_prices(conn):
    """Indices added to INDEX_TICKERS after the database was built are fetched here."""
    save_tickers(conn, [(s, name, None, None) for s, name in INDEX_TICKERS])
    missing = missing_tickers(conn, [s for s, _ in INDEX_TICKERS], START, END)
    if missing:
        save_prices(conn, download_prices(missing, START, END))
    print(f"[8] index prices added: {missing}")


def step_sector_etfs(conn):
    """Sector ETFs start earlier than the stocks: the 12-month momentum of the first month needs 13 months of prices."""
    save_tickers(conn, [(s, f"Select Sector SPDR ETF ({sector})", sector, None) for s, sector in SECTOR_ETFS])
    missing = missing_tickers(conn, [s for s, _ in SECTOR_ETFS], config.ETF_START_DATE, END)
    if missing:
        save_prices(conn, download_prices(missing, config.ETF_START_DATE, END))
    print(f"[9] sector ETF prices added: {missing}")


def fill_missing_gics(conn):
    """Stocks that left the index are not on the current Wikipedia page, so their sector is read from
    older revisions of the page (newest first: the classification closest to today is kept)."""
    def missing():
        return {r[0] for r in conn.execute(
            "SELECT symbol FROM tickers WHERE gics_sector IS NULL AND symbol IN (SELECT symbol FROM index_membership)")}

    before = missing()
    # every quarter, plus the first day of the period for stocks that left right after it
    dates = [*pd.date_range(START, END, freq='QS').strftime('%Y-%m-%d')[::-1], START] if before else []
    for date in dates:
        if not missing():
            break
        table = load_sp500_table_at(date).dropna(subset=['GICS Sector'])
        fill_gics(conn, [(to_yf_symbol(r['Symbol'], TICKER_ALIASES), r['GICS Sector'], r['GICS Sub-Industry'])
                         for _, r in table.iterrows()])
    print(f"[7] GICS filled from old Wikipedia revisions: {len(before) - len(missing())} | still missing: {sorted(missing())}")


def report(conn):
    """Checks that need a human look (nothing is changed here)."""
    print("\n[check] members whose prices do not cover their membership period:")
    rows = conn.execute("""
        SELECT m.symbol, COALESCE(m.start_date, ?) AS s, COALESCE(m.end_date, ?) AS e,
               (SELECT COUNT(*) FROM prices p WHERE p.symbol = m.symbol
                  AND p.date >= COALESCE(m.start_date, ?) AND p.date < COALESCE(m.end_date, '9999')) AS n,
               (SELECT COUNT(DISTINCT date) FROM prices
                  WHERE date >= COALESCE(m.start_date, ?) AND date < COALESCE(m.end_date, '9999')) AS days
        FROM index_membership m JOIN tickers t USING (symbol)
        WHERE t.excluded_reason IS NULL""", (START, END, START, START)).fetchall()
    for symbol, s, e, n, days in rows:
        if days and n < days * 0.99:
            print(f"    {symbol}: {n}/{days} days ({s} .. {e})")
    print("[check] daily moves over 50% (a reused ticker would show up here):")
    big = conn.execute("""
        SELECT symbol, date, r FROM (
          SELECT symbol, date, adj_close / LAG(adj_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS r FROM prices)
        WHERE ABS(r) > 0.5 ORDER BY ABS(r) DESC""").fetchall()
    for symbol, date, r in big:
        print(f"    {symbol} {date}: {r:+.0%}")


def main():
    conn = connect(config.DB_PATH)
    init_db(conn)
    print(f"DB: {config.DB_PATH} | period: {START} .. {END}")
    for table, step in [('index_changes', step_index_changes), ('index_membership', step_membership),
                        ('prices', step_prices), ('shares_outstanding', step_shares)]:
        if is_empty(conn, table):
            step(conn)
        else:
            print(f"skip: {table} already has data")
    exclude_without_member_prices(conn)
    exclude_without_shares(conn)
    fill_missing_gics(conn)
    step_index_prices(conn)
    step_sector_etfs(conn)
    report(conn)


if __name__ == "__main__":
    main()
