"""
S&P500 constituents (ticker, company name, GICS sector / sub-industry) and their change history.
"""
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from marketdb.config import SP500_URL, SP500_HISTORY_URL, USER_AGENT, WIKIPEDIA_API_URL, WIKIPEDIA_REVISION_URL


def to_yf_symbol(symbol, aliases=None):
    """Wikipedia ticker -> latest yfinance ticker ('BRK.B' -> 'BRK-B', old tickers via aliases)."""
    symbol = (aliases or {}).get(symbol, symbol)
    return symbol.replace('.', '-')


def load_sp500_table(tickers=None):
    tables = pd.read_html(SP500_URL, storage_options={'User-Agent': USER_AGENT})[0]
    sp500_table = tables[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    if tickers:
        sp500_table = sp500_table[sp500_table['Symbol'].isin(tickers)]
    return sp500_table


def _get(url, attempts=4, wait=30):
    """Wikipedia answers HTTP 429 after several requests in a row, so wait and try again."""
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT})).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(wait)


def load_sp500_table_at(date):
    """The constituents table as it was on Wikipedia at the start of `date` (same columns as load_sp500_table)."""
    query = urllib.parse.urlencode({
        'action': 'query', 'prop': 'revisions', 'titles': urllib.parse.unquote(SP500_URL.rsplit('/', 1)[1]),
        'rvlimit': 1, 'rvstart': f'{date}T00:00:00Z', 'rvdir': 'older', 'rvprop': 'ids', 'format': 'json'})
    pages = json.loads(_get(f'{WIKIPEDIA_API_URL}?{query}'))['query']['pages']
    revid = next(iter(pages.values()))['revisions'][0]['revid']
    table = pd.read_html(io.BytesIO(_get(WIKIPEDIA_REVISION_URL.format(revid=revid))))[0]
    # older revisions (before about 2021) name two of the columns differently
    table = table.rename(columns={'Ticker symbol': 'Symbol', 'GICS Sub Industry': 'GICS Sub-Industry'})
    return table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]


def load_index_changes():
    """All constituent changes listed on Wikipedia (columns: date, added_symbol, added_name, removed_symbol, removed_name, reason)."""
    changes = pd.read_html(SP500_HISTORY_URL, storage_options={'User-Agent': USER_AGENT})[0]
    changes.columns = ['date', 'added_symbol', 'added_name', 'removed_symbol', 'removed_name', 'reason', 'refs']
    changes['date'] = pd.to_datetime(changes['date']).dt.strftime('%Y-%m-%d')
    changes = changes.drop(columns='refs')
    return changes.astype(object).where(changes.notna(), None)
