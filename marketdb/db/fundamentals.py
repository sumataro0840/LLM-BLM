"""
Save financial statement values (financial_items / financial_facts).
"""
from marketdb.db.market import now_utc


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
