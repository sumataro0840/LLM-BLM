"""
Earnings and revenue surprises of the releases from the financial statements (設計.md, 6).

No analyst forecasts are available, so the surprise is measured against the same quarter a year earlier
(seasonal random walk, as in Bernard and Thomas):
    sue = (eps - eps a year earlier) / std of the same seasonal differences over the previous 8 quarters

The quarter's own numbers are taken from the 10-Q/10-K filed after the release (the release made them public);
only quarters filed before the release go into the history (std and mean growth).
"""
import numpy as np
import pandas as pd

MAX_DAYS_AFTER_QUARTER = 100    # a release reports the latest quarter that ended at most this many days before
YEAR_BACK = (350, 380)          # days between a quarter and the same quarter a year earlier (52/53-week years)
HISTORY = 8
MIN_HISTORY = 4


def load_quarters(conn, symbols):
    marks = ", ".join("?" * len(symbols))
    df = pd.read_sql(f"SELECT symbol, period_end, filed_date, eps_diluted AS eps, revenue FROM fundamentals_quarterly "
                     f"WHERE symbol IN ({marks})", conn, params=list(symbols))
    return df.sort_values(["symbol", "period_end"]).reset_index(drop=True)


def with_year_ago(q):
    """Add eps_lag4 / revenue_lag4: the quarter of the same symbol that ended 350-380 days earlier."""
    out = []
    for _, g in q.groupby("symbol", sort=False):
        g = g.copy()
        ends = pd.to_datetime(g.period_end)
        eps_lag, rev_lag = [], []
        for end in ends:
            gap = (end - ends).dt.days
            match = g[(gap >= YEAR_BACK[0]) & (gap <= YEAR_BACK[1])]
            eps_lag.append(match.eps.iloc[-1] if len(match) else np.nan)
            rev_lag.append(match.revenue.iloc[-1] if len(match) else np.nan)
        g["eps_lag4"], g["revenue_lag4"] = eps_lag, rev_lag
        out.append(g)
    q = pd.concat(out, ignore_index=True) if out else q.assign(eps_lag4=[], revenue_lag4=[])
    q["eps_diff"] = q.eps - q.eps_lag4
    q["revenue_growth"] = q.revenue / q.revenue_lag4 - 1
    return q


def surprise_of(release_date, quarters):
    """Surprise of one release (release_date 'YYYY-MM-DD', the date of acceptance in New York) from the
    quarters of its company (output of with_year_ago, one symbol)."""
    ends = pd.to_datetime(quarters.period_end)
    day = pd.Timestamp(release_date)
    target = quarters[(ends < day) & (ends >= day - pd.Timedelta(days=MAX_DAYS_AFTER_QUARTER))]
    empty = {"period_end": None, "eps": None, "eps_lag4": None, "sue": None, "revenue_growth": None,
             "revenue_surprise": None, "n_history": 0}
    if target.empty:
        return empty
    t = target.iloc[-1]
    known = quarters[(quarters.filed_date < release_date) & (quarters.period_end < t.period_end)]
    diffs = known.eps_diff.dropna().tail(HISTORY)
    growths = known.revenue_growth.replace([np.inf, -np.inf], np.nan).dropna().tail(4)
    scale = diffs.std(ddof=1) if len(diffs) >= MIN_HISTORY else np.nan
    sue = t.eps_diff / scale if pd.notna(t.eps_diff) and scale > 0 else np.nan
    growth = t.revenue_growth if np.isfinite(t.revenue_growth) else np.nan
    rev_surprise = growth - growths.mean() if pd.notna(growth) and len(growths) >= 2 else np.nan
    return {"period_end": t.period_end, "eps": t.eps, "eps_lag4": t.eps_lag4, "sue": sue,
            "revenue_growth": growth, "revenue_surprise": rev_surprise, "n_history": int(len(diffs))}


def compute(conn, releases):
    """releases: frame with accession, symbol, accepted_et. Returns one row per release (SURPRISE_COLUMNS)."""
    quarters = with_year_ago(load_quarters(conn, sorted(releases.symbol.unique())))
    by_symbol = {s: g for s, g in quarters.groupby("symbol")}
    rows = []
    for r in releases.itertuples():
        q = by_symbol.get(r.symbol)
        values = surprise_of(r.accepted_et[:10], q) if q is not None else surprise_of(r.accepted_et[:10], quarters.iloc[0:0])
        rows.append({"accession": r.accession, **values})
    return pd.DataFrame(rows)
