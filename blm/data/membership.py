"""
Rebuild S&P500 membership periods from the current constituents and the change history.
"""


def _symbol(value):
    """Empty cells may come as None or NaN."""
    return value if isinstance(value, str) and value else None


def build_membership(current_symbols, changes, start, end):
    """Membership periods overlapping [start, end].

    current_symbols: today's constituents; changes: rows of (date, added_symbol, removed_symbol)
    with symbols already converted to the latest tickers. Changes are undone from the newest one
    back to `start`: an added stock leaves the list, a removed stock comes back.

    Returns (periods, issues):
      periods: [(symbol, start_date or None, end_date or None)]  None = before start / still a member
      issues:  changes that do not fit the list (usually a ticker that changed later)
    """
    members = set(current_symbols)
    open_end = {s: None for s in members}
    periods, issues = [], []
    for _, r in changes[changes['date'] >= start].sort_values('date', ascending=False).iterrows():
        added, removed = _symbol(r['added_symbol']), _symbol(r['removed_symbol'])
        if added:
            if added in members:
                members.discard(added)
                periods.append((added, r['date'], open_end.pop(added)))
            else:
                issues.append(f"{r['date']}: {added} was added but is not in the list")
        if removed:
            if removed in members:
                issues.append(f"{r['date']}: {removed} was removed but is already in the list")
            members.add(removed)
            open_end[removed] = r['date']
    periods += [(s, None, e) for s, e in open_end.items()]
    periods = [(s, st, en) for s, st, en in periods if (st is None or st <= end) and (en is None or en > start)]
    return sorted(periods, key=lambda p: (p[0], p[1] or '')), issues
