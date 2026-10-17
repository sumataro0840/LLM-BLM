"""
When the market can react to a release: acceptance time in New York, session and the event date (設計.md, 5).
"""
from bisect import bisect_left
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EDGAR_CUTOFF = time(17, 30)     # filings accepted later are disseminated the next business day


def parse_utc(accepted):
    """EDGAR's acceptanceDateTime ('2026-07-16T11:36:19.000Z') or an ISO time with an offset -> aware UTC time."""
    text = accepted.strip().replace("Z", "+00:00")
    t = datetime.fromisoformat(text)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def to_new_york(accepted):
    return parse_utc(accepted).astimezone(NEW_YORK)


def session_of(t_et):
    """'pre' (before 9:30), 'open' (9:30-16:00), 'post' (16:00-17:30) or 'late' (from 17:30), New York time."""
    clock = t_et.time()
    if clock < MARKET_OPEN:
        return "pre"
    if clock < MARKET_CLOSE:
        return "open"
    if clock < EDGAR_CUTOFF:
        return "post"
    return "late"


def event_date(t_et, session, trading_days):
    """First trading day whose close can reflect the release (trading_days: sorted 'YYYY-MM-DD').
    Before or during the session: that day if it is a trading day; after the close: the next trading day.
    None when the day is after the last known trading day."""
    day = t_et.date().isoformat()
    if trading_days and day > trading_days[-1]:
        return None
    i = bisect_left(trading_days, day)
    if session in ("post", "late") and i < len(trading_days) and trading_days[i] == day:
        i += 1
    return trading_days[i] if i < len(trading_days) else None


def timing(accepted, trading_days):
    """(accepted_at in UTC, accepted_et, session, event_date) of one acceptance time."""
    t_utc = parse_utc(accepted)
    t_et = t_utc.astimezone(NEW_YORK)
    session = session_of(t_et)
    return (t_utc.isoformat(timespec="seconds"), t_et.replace(tzinfo=None).isoformat(timespec="seconds"),
            session, event_date(t_et, session, trading_days))
