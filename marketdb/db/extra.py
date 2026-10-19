"""
Save / load the additional data (設計_追加データ.md). Values are kept as first fetched (INSERT OR IGNORE).
"""
import pandas as pd

from marketdb.db.market import now_utc

COLUMNS = {
    'filings': ['accession', 'cik', 'symbol', 'form', 'filing_date', 'accepted_at', 'items', 'primary_doc'],
    'volumes': ['symbol', 'date', 'volume'],
    'factor_returns': ['dataset', 'freq', 'date', 'factor', 'value'],
    'macro_series': ['series_id', 'date', 'value', 'available_date'],
    'short_volume': ['symbol', 'date', 'short_volume', 'short_exempt_volume', 'total_volume'],
    'pageviews': ['symbol', 'article', 'date', 'views'],
    'holdings_13f_parts': ['source_file', 'cusip', 'period', 'n_filers', 'shares', 'value', 'issuer_name'],
    'cusip_map': ['cusip', 'symbol', 'figi_name', 'figi_ticker'],
    'insider_transactions': ['accession', 'trans_sk', 'symbol', 'issuer_cik', 'form', 'filing_date', 'available_date',
                             'trans_date', 'code', 'acquired_disposed', 'shares', 'price', 'shares_after',
                             'direct_indirect', 'owner_cik', 'owner_relationship', 'owner_title'],
}


def save(conn, table, df):
    """Insert the rows of df (columns of COLUMNS[table]); rows already stored are kept as they are."""
    if df is None or df.empty:
        return 0
    cols = COLUMNS[table]
    stamp = now_utc()
    rows = [tuple(None if pd.isna(v) else v for v in r) + (stamp,) for r in df[cols].itertuples(index=False)]
    cur = conn.executemany(f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}, fetched_at) "
                           f"VALUES ({', '.join('?' * (len(cols) + 1))})", rows)
    conn.commit()
    return cur.rowcount


def done(conn, table, column):
    """Distinct values of a column already stored (to continue an interrupted fetch)."""
    return {r[0] for r in conn.execute(f'SELECT DISTINCT {column} FROM {table}')}


def load(conn, table, where='', params=()):
    return pd.read_sql(f'SELECT * FROM {table} {where}', conn, params=list(params))
