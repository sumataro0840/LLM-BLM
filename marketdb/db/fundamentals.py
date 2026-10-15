"""
Save financial statement values (financial_items / financial_facts), and read the share counts in them.
"""
import pandas as pd

from marketdb.db.market import now_utc

# Share counts in the statements; all of them are used and wrong ones are dropped later (to_current_split_basis):
# the cover page count, the balance sheet count and the average count of a quarter.
# Companies with several share classes report them per class, which EDGAR's company facts leave out.
SHARE_TAGS = ['EntityCommonStockSharesOutstanding', 'CommonStockSharesOutstanding',
              'WeightedAverageNumberOfSharesOutstandingBasic']
# An average over a longer period (half year, year) is too far from the filing date
MAX_AVERAGE_DAYS = 100
# Fewer shares than this are reporting errors (a count in thousands, or 1)
MIN_SHARES = 1_000_000


def ensure_ticker_columns(conn):
    """Add the columns used here to a database created before they existed."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(tickers)")}
    for column, column_type in [('cik', 'INTEGER'), ('fundamentals_fetched_at', 'TEXT')]:
        if column not in existing:
            conn.execute(f"ALTER TABLE tickers ADD COLUMN {column} {column_type}")
    conn.commit()


def save_ciks(conn, ciks):
    """ciks: {symbol: cik}"""
    conn.executemany("UPDATE tickers SET cik = ? WHERE symbol = ?", [(c, s) for s, c in ciks.items()])
    conn.commit()


def save_facts(conn, symbol, rows):
    """rows: (taxonomy, tag, unit, label, period_start, period_end, value, form, filed_date) from marketdb.sources.edgar."""
    items = {(r[0], r[1], r[2]): r[3] for r in rows}
    conn.executemany("INSERT OR IGNORE INTO financial_items (taxonomy, tag, unit, label) VALUES (?, ?, ?, ?)",
                     [(*key, label) for key, label in items.items()])
    ids = {(t, g, u): i for i, t, g, u in conn.execute("SELECT item_id, taxonomy, tag, unit FROM financial_items")}
    conn.executemany(
        "INSERT OR IGNORE INTO financial_facts (symbol, item_id, period_start, period_end, value, form, filed_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(symbol, ids[(r[0], r[1], r[2])], r[4], r[5], r[6], r[7], r[8]) for r in rows],
    )
    conn.execute("UPDATE tickers SET fundamentals_fetched_at = ? WHERE symbol = ?", (now_utc(), symbol))
    conn.commit()


def load_reported_shares(conn, since):
    """{symbol: share counts filed on or after `since`, indexed by filing date (several values per day)}.

    A filing gives its counts on the split basis of its filing date (earlier periods are restated for splits),
    so the value is dated by the filing, which is also the day it became known.
    """
    rows = pd.read_sql(f"""
        SELECT f.symbol, f.filed_date, f.value FROM financial_facts f JOIN financial_items i USING (item_id)
        WHERE i.unit = 'shares' AND i.tag IN ({','.join('?' * len(SHARE_TAGS))})
          AND f.filed_date >= ? AND f.value >= ?
          AND julianday(f.period_end) - julianday(COALESCE(f.period_start, f.period_end)) <= ?
        ORDER BY f.symbol, f.filed_date""", conn, params=[*SHARE_TAGS, since, MIN_SHARES, MAX_AVERAGE_DAYS])
    return {symbol: pd.Series(g['value'].values, index=g['filed_date'].values, dtype=float)
            for symbol, g in rows.groupby('symbol')}
