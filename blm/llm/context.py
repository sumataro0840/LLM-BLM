"""
Information passed to the LLM for each ticker.
"""


def build_context(ticker, sp500_table, returns):
    row = sp500_table[sp500_table['Symbol'] == ticker]
    return {
        'ticker': ticker,
        'Security': row['Security'].values[0],
        'GICS Sector': row['GICS Sector'].values[0],
        'GICS Sub-Industry': row['GICS Sub-Industry'].values[0],
        'pct_change': returns[ticker].tolist()
    }
