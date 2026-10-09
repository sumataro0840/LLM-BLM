"""
S&P500 constituents (ticker, company name, GICS sector / sub-industry).
"""
import pandas as pd

from blm.config.data import SP500_URL, USER_AGENT


def load_sp500_table(tickers=None):
    tables = pd.read_html(SP500_URL, storage_options={'User-Agent': USER_AGENT})[0]
    sp500_table = tables[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    if tickers:
        sp500_table = sp500_table[sp500_table['Symbol'].isin(tickers)]
    return sp500_table
