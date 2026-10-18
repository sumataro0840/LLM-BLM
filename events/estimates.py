"""
Analyst consensus EPS and the reported EPS of each quarter (Alpha Vantage, EARNINGS endpoint).

The free key allows 25 requests a day; one request returns the whole history of one symbol, so
`run_events.py estimates` fetches up to 25 symbols per run and continues on the next day.
Answers are cached in data/cache/alphavantage/ (outside Git: the license is personal and non-sublicensable).
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from events import config
from marketdb.db.market import now_utc

URL = "https://www.alphavantage.co/query"
SOURCE = "alphavantage:EARNINGS"
WAIT_SECONDS = 15                 # stay well below any per-minute limit of the free key


class LimitReached(Exception):
    """The source says the daily (or per-minute) allowance is used up; nothing is cached."""


def provider_symbol(symbol):
    """yfinance writes share classes with '-' (BRK-B); Alpha Vantage uses '.' (BRK.B)."""
    return symbol.replace("-", ".")


def cache_path(symbol):
    return config.ESTIMATES_DIR / f"{symbol}.json"


def fetch(symbol, api_key):
    """Download one symbol (or read the cache). Raises LimitReached when the allowance is used up."""
    path = cache_path(symbol)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), False
    query = urllib.parse.urlencode({"function": "EARNINGS", "symbol": provider_symbol(symbol), "apikey": api_key})
    with urllib.request.urlopen(f"{URL}?{query}", timeout=60) as res:
        data = json.load(res)
    if "Note" in data or "Information" in data:
        # rate-limit and premium messages come back with HTTP 200; their text can contain the key, so it is not shown
        raise LimitReached(next(k for k in ("Note", "Information") if k in data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")     # also caches "no data" answers (empty or an error message)
    return data, True


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse(symbol, data):
    """Quarterly rows of one answer (an empty frame when the source has no data for the symbol)."""
    rows = []
    for q in data.get("quarterlyEarnings", []):
        time_of_day = q.get("reportTime")
        rows.append({"symbol": symbol, "fiscal_date_ending": q.get("fiscalDateEnding"),
                     "reported_date": q.get("reportedDate"),
                     "report_time": time_of_day if time_of_day in ("pre-market", "post-market") else None,
                     "reported_eps": _num(q.get("reportedEPS")), "estimated_eps": _num(q.get("estimatedEPS")),
                     "surprise": _num(q.get("surprise")), "surprise_pct": _num(q.get("surprisePercentage")),
                     "source": SOURCE, "fetched_at": now_utc()})
    return pd.DataFrame(rows, columns=["symbol", "fiscal_date_ending", "reported_date", "report_time", "reported_eps",
                                       "estimated_eps", "surprise", "surprise_pct", "source", "fetched_at"])


def match_releases(releases, estimates, max_days=3):
    """Pair each release with the estimate row of the same symbol whose reported date is closest to the day the
    release was accepted (New York time), within max_days. Returns the releases with the matched columns."""
    out = []
    by_symbol = {s: g for s, g in estimates.groupby("symbol")}
    for r in releases.itertuples():
        g = by_symbol.get(r.symbol)
        day = pd.Timestamp(r.accepted_et[:10])
        best = None
        if g is not None:
            gap = (pd.to_datetime(g.reported_date) - day).dt.days
            near = g[gap.abs() <= max_days]
            if len(near):
                best = near.loc[gap[near.index].abs().idxmin()]
                best_gap = int(gap[best.name])
        out.append({"accession": r.accession, "symbol": r.symbol, "session": r.session,
                    "matched": best is not None,
                    "days_from_acceptance": best_gap if best is not None else None,
                    "report_time": best.report_time if best is not None else None,
                    "fiscal_date_ending": best.fiscal_date_ending if best is not None else None})
    return pd.DataFrame(out)


def run_daily(symbols, api_key, limit, on_rows, log=print):
    """Fetch up to `limit` new symbols; cached ones are parsed without a request. Stops at the allowance."""
    requested = 0
    for symbol in symbols:
        if requested >= limit and not cache_path(symbol).exists():
            break
        try:
            data, was_request = fetch(symbol, api_key)
        except LimitReached as e:
            log(f"  allowance used up ({e}); continue tomorrow")
            break
        if was_request:
            requested += 1
            time.sleep(WAIT_SECONDS)
        rows = parse(symbol, data)
        on_rows(symbol, rows)
        if was_request:
            log(f"  {symbol}: {len(rows)} quarters (request {requested}/{limit})")
    return requested
