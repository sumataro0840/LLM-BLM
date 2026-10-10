"""
Save S&P500 constituent changes and membership periods.
"""
from blm.db.market import now_utc


def save_index_changes(conn, changes):
    fetched_at = now_utc()
    conn.executemany(
        "INSERT OR IGNORE INTO index_changes (date, added_symbol, added_name, removed_symbol, removed_name, reason, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(r['date'], r['added_symbol'], r['added_name'], r['removed_symbol'], r['removed_name'], r['reason'], fetched_at)
         for _, r in changes.iterrows()],
    )
    conn.commit()


def save_membership(conn, periods):
    """periods: [(symbol, start_date or None, end_date or None)]"""
    conn.executemany("INSERT INTO index_membership (symbol, start_date, end_date) VALUES (?, ?, ?)", periods)
    conn.commit()
