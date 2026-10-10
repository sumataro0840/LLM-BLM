"""
S&P500 constituents (ticker, company name, GICS sector / sub-industry) and their change history.
"""
import pandas as pd

from blm.config.data import SP500_URL, SP500_HISTORY_URL, USER_AGENT


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


def load_index_changes():
    """All constituent changes listed on Wikipedia (columns: date, added_symbol, added_name, removed_symbol, removed_name, reason)."""
    changes = pd.read_html(SP500_HISTORY_URL, storage_options={'User-Agent': USER_AGENT})[0]
    changes.columns = ['date', 'added_symbol', 'added_name', 'removed_symbol', 'removed_name', 'reason', 'refs']
    changes['date'] = pd.to_datetime(changes['date']).dt.strftime('%Y-%m-%d')
    changes = changes.drop(columns='refs')
    return changes.astype(object).where(changes.notna(), None)
