"""
Stock reactions to the releases, computed from the daily prices every time (nothing is stored; 設計.md, 7).

A window (a, b) counts trading days from the event date (day 0 = first close that can reflect the release):
it runs from the close of day a - 1 to the close of day b. The main window is (0, 1).
Abnormal return = stock return - return of the sector ETF (the S&P500 when the ETF has no price yet).

These functions look at outcomes: they are run on the real data only after the pre-registration (設計.md, 9.2).
"""
import numpy as np
import pandas as pd

WINDOWS = {"r_0_1": (0, 1), "r_0_0": (0, 0), "r_m1_1": (-1, 1), "drift_2_6": (2, 6), "drift_2_21": (2, 21)}
PRE_WINDOW = (-21, -1)          # return before the release (a control, known at the release)
MARKET = "^GSPC"
OLD_SECTORS = {"Telecommunication Services": "Communication Services", "Telecommunications Services": "Communication Services"}


def sector_benchmarks(conn):
    """symbol -> sector ETF, from the sector names in tickers."""
    t = pd.read_sql("SELECT symbol, gics_sector FROM tickers", conn)
    members = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM index_membership"))
    etf_of_sector = {s: sym for sym, s in zip(t.symbol, t.gics_sector) if sym not in members and s and not sym.startswith("^")}
    sector = t.set_index("symbol").gics_sector.replace(OLD_SECTORS)
    return {sym: etf_of_sector.get(sec) for sym, sec in sector.items() if sym in members}


def load_closes(conn, symbols, start, end):
    marks = ", ".join("?" * len(symbols))
    df = pd.read_sql(f"SELECT symbol, date, adj_close FROM prices WHERE symbol IN ({marks}) AND date BETWEEN ? AND ?",
                     conn, params=[*symbols, start, end])
    return df.pivot(index="date", columns="symbol", values="adj_close").sort_index()


def window_return(closes, symbol, days, k, window):
    """Return of symbol over a window around trading day number k (NaN when a price is missing)."""
    a, b = window
    i, j = k + a - 1, k + b
    if symbol not in closes or i < 0 or j >= len(days):
        return np.nan
    p0, p1 = closes[symbol].get(days[i]), closes[symbol].get(days[j])
    if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1):
        return np.nan
    return p1 / p0 - 1


def reactions(releases, closes, trading_days, benchmarks, windows=WINDOWS):
    """releases: frame with accession, symbol, event_date. Returns one row per release with, for every window,
    the stock return (ret_*), the benchmark used (bench), its return (bench_*) and the abnormal return (abn_*)."""
    index = {d: n for n, d in enumerate(trading_days)}
    rows = []
    for r in releases.itertuples():
        row = {"accession": r.accession}
        k = index.get(r.event_date) if isinstance(r.event_date, str) else None
        etf = benchmarks.get(r.symbol)
        for name, window in {**windows, "pre": PRE_WINDOW}.items():
            if k is None:
                row[f"ret_{name}"] = row[f"bench_{name}"] = row[f"abn_{name}"] = np.nan
                continue
            stock = window_return(closes, r.symbol, trading_days, k, window)
            bench = window_return(closes, etf, trading_days, k, window) if etf else np.nan
            used = etf
            if np.isnan(bench):
                bench, used = window_return(closes, MARKET, trading_days, k, window), MARKET
            row[f"ret_{name}"], row[f"bench_{name}"], row[f"abn_{name}"] = stock, bench, stock - bench
            if name == "r_0_1":
                row["benchmark"] = used
        rows.append(row)
    return pd.DataFrame(rows)


def log_market_cap(conn, releases, trading_days):
    """Log market cap at the close before the event date (a control known at the release)."""
    index = {d: n for n, d in enumerate(trading_days)}
    shares = pd.read_sql("SELECT symbol, date, shares FROM shares_outstanding", conn)
    by_symbol = {s: g.set_index("date").shares.sort_index() for s, g in shares.groupby("symbol")}
    out = {}
    for r in releases.itertuples():
        k = index.get(r.event_date) if isinstance(r.event_date, str) else None
        if k is None or k == 0 or r.symbol not in by_symbol:
            out[r.accession] = np.nan
            continue
        day = trading_days[k - 1]
        row = conn.execute("SELECT close FROM prices WHERE symbol = ? AND date = ?", (r.symbol, day)).fetchone()
        s = by_symbol[r.symbol]
        s = s[s.index <= day]
        out[r.accession] = np.log(row[0] * s.iloc[-1]) if row and len(s) else np.nan
    return pd.Series(out, name="log_market_cap")
