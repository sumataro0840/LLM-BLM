"""
Build the SQLite database for the study period (DB_START_DATE .. DB_END_DATE in .env).

    python scripts/build_db.py

Steps (each one is skipped when its table already has rows, so stored data is never refetched):
  1. S&P500 constituent changes (Wikipedia)          -> index_changes
  2. membership periods and tickers                  -> index_membership, tickers
  3. daily prices of every member and ^GSPC          -> prices  (no data -> tickers.excluded_reason)
  4. shares outstanding on the current split basis   -> shares_outstanding
  5-6. members without prices or shares for most of their membership are excluded
     (tickers.excluded_reason) and their prices / shares are deleted

Tickers are stored as the latest yfinance tickers ('BRK-B', not 'BRK.B'): old tickers often
fail on yfinance or now point to another company.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from blm.config import data as data_config
from blm.data.membership import build_membership
from blm.data.prices import download_prices
from blm.data.shares import class_ratios, clip_to_period, download_shares_and_splits, to_current_split_basis
from blm.data.universe import load_index_changes, load_sp500_table, to_yf_symbol
from blm.db.connection import connect, init_db
from blm.db.index import save_index_changes, save_membership
from blm.db.market import mark_excluded, save_prices, save_shares, save_tickers

# Tickers that changed after the change was recorded on Wikipedia (old -> latest)
TICKER_ALIASES = {
    'SATS': 'ECHO',  # EchoStar: added on 2026-03-23 as SATS, now listed as ECHO
}
INDEX_TICKERS = [('^GSPC', 'S&P 500 Index')]
# Companies with several share classes in the index: yfinance reports the company-wide share count
# for each class, so it is split by the current class ratio (otherwise the company is counted twice)
MULTI_CLASS = [('GOOGL', 'GOOG'), ('FOXA', 'FOX'), ('NWSA', 'NWS')]

START, END = data_config.DB_START_DATE, data_config.DB_END_DATE


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
    symbols = [r[0] for r in conn.execute(
        "SELECT symbol FROM tickers WHERE excluded_reason IS NULL AND symbol NOT LIKE '^%' ORDER BY symbol")]
    # one year before START so that a value exists on the first day
    since = (pd.Timestamp(START) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    until = (pd.Timestamp(END) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    def fetch(symbol):
        try:
            shares = to_current_split_basis(*download_shares_and_splits(symbol, since, until))
            return symbol, clip_to_period(shares, START, END), None
        except Exception as e:
            return symbol, None, str(e)[:80]

    ratio = {s: r for classes in MULTI_CLASS for s, r in class_ratios(classes).items()}
    failed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (symbol, shares, error) in enumerate(pool.map(fetch, symbols), 1):
            if shares is None or shares.empty:
                failed.append((symbol, error or 'no data'))
            else:
                save_shares(conn, symbol, (shares * ratio[symbol]).round().astype('int64') if symbol in ratio else shares)
            if i % 100 == 0:
                print(f"    {i}/{len(symbols)}")
    print(f"    share classes split by: { {k: round(v, 3) for k, v in ratio.items()} }")
    print(f"[4] shares_outstanding: {len(symbols) - len(failed)} tickers, "
          f"{conn.execute('SELECT COUNT(*) FROM shares_outstanding').fetchone()[0]} rows | no shares data: {failed}")


def _exclude(conn, symbols, reason):
    """Mark as excluded and delete their prices and shares."""
    if symbols:
        mark_excluded(conn, symbols, reason)
        marks = ','.join('?' * len(symbols))
        conn.execute(f"DELETE FROM prices WHERE symbol IN ({marks})", symbols)
        conn.execute(f"DELETE FROM shares_outstanding WHERE symbol IN ({marks})", symbols)
        conn.commit()


def exclude_without_member_prices(conn, min_coverage=0.5):
    """Exclude members whose prices cover less than min_coverage of their membership days.
    A delisted ticker can be reused by another security, so prices may exist only outside the membership."""
    rows = conn.execute("""
        SELECT m.symbol,
               (SELECT COUNT(*) FROM prices p WHERE p.symbol = m.symbol
                  AND (m.start_date IS NULL OR p.date >= m.start_date) AND (m.end_date IS NULL OR p.date < m.end_date)) AS n,
               (SELECT COUNT(*) FROM prices g WHERE g.symbol = '^GSPC'
                  AND (m.start_date IS NULL OR g.date >= m.start_date) AND (m.end_date IS NULL OR g.date < m.end_date)) AS days
        FROM index_membership m JOIN tickers t USING (symbol)
        WHERE t.excluded_reason IS NULL""").fetchall()
    symbols = [s for s, n, days in rows if days and n < days * min_coverage]
    _exclude(conn, symbols, 'no price data during membership (ticker reused by another security)')
    print(f"[5] excluded (no prices during membership): {symbols}")


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
    _exclude(conn, symbols, 'no shares history on yfinance (ticker changed)')
    print(f"[6] excluded (no shares history): {symbols}")


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
    conn = connect(data_config.DB_PATH)
    init_db(conn)
    print(f"DB: {data_config.DB_PATH} | period: {START} .. {END}")
    for table, step in [('index_changes', step_index_changes), ('index_membership', step_membership),
                        ('prices', step_prices), ('shares_outstanding', step_shares)]:
        if is_empty(conn, table):
            step(conn)
        else:
            print(f"skip: {table} already has data")
    exclude_without_member_prices(conn)
    exclude_without_shares(conn)
    report(conn)


if __name__ == "__main__":
    main()
