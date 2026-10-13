"""
Stock indicators on a decision date: price behaviour from daily prices,
fundamentals from the latest quarter filed by that date.
"""
import numpy as np
import pandas as pd

from marketdb.transform.sector import volatility

MONTH_DAYS = 21          # trading days in a month
YEAR_DAYS = 252          # trading days in a year
# Metrics that mislead in some industries because of how the business or its accounting works.
# They are left out (n/a) there, by GICS sub-industry; everything else in the sector keeps all metrics.
INDUSTRY_EXCLUSIONS = [
    # borrowing is the raw material of lenders and dealers, and their revenue is reported net of interest
    (['Diversified Banks', 'Regional Banks', 'Consumer Finance', 'Investment Banking & Brokerage',
      'Asset Management & Custody Banks'],
     ['net_margin', 'debt_to_equity']),
    # net income includes unrealized gains and losses on a large stock portfolio (Berkshire Hathaway)
    (['Multi-Sector Holdings'],
     ['pe_trailing', 'net_income_growth_yoy', 'net_margin', 'roe']),
    # REITs: depreciation of buildings and gains on property sales make net income a poor measure of earnings
    # (the industry uses funds from operations, which the database does not have)
    (['Data Center REITs', 'Health Care REITs', 'Hotel & Resort REITs', 'Industrial REITs',
      'Multi-Family Residential REITs', 'Office REITs', 'Other Specialized REITs', 'Retail REITs',
      'Self-Storage REITs', 'Single-Family Residential REITs', 'Telecom Tower REITs', 'Timber REITs'],
     ['pe_trailing', 'net_income_growth_yoy', 'net_margin', 'roe']),
]


def excluded_metrics(sub_industry):
    """Fundamental metrics that are not computed for this sub-industry."""
    return [metric for industries, metrics in INDUSTRY_EXCLUSIONS if sub_industry in industries for metric in metrics]


def price_metrics(prices, market, sector_prices, sectors):
    """Price behaviour of each stock on the last row of `prices`.

    prices: daily adjusted closes up to the decision date (columns = stocks); market: the S&P 500 on the same days;
    sector_prices: sector ETFs on the same days (columns = sector names); sectors: sector of each stock.
    A stock without enough history gets NaN for that metric.
    """
    def momentum(p):
        # the latest month is left out because short-term returns tend to reverse
        return p.iloc[-1 - MONTH_DAYS] / p.iloc[-YEAR_DAYS] - 1 if len(p) >= YEAR_DAYS else p.iloc[-1] * np.nan

    out = pd.DataFrame(index=prices.columns)
    out['momentum_12m'] = momentum(prices)
    out['return_1m'] = prices.iloc[-1] / prices.iloc[-1 - MONTH_DAYS] - 1
    out['relative_momentum_12m'] = out['momentum_12m'] - sectors.map(momentum(sector_prices))
    out['volatility_30d'] = volatility(prices)

    returns = prices.pct_change().iloc[-YEAR_DAYS:]
    market_returns = market.pct_change().iloc[-YEAR_DAYS:]
    beta = returns.apply(lambda r: r.cov(market_returns)) / market_returns.var()
    # a beta from less than half a year of returns is not reliable
    out['beta_1y'] = beta.where(returns.notna().sum() >= YEAR_DAYS // 2)
    return out


def fundamental_metrics(quarters, market_caps, sub_industries, as_of_date, max_age_days):
    """Fundamentals of each stock from the quarters filed by `as_of_date`.

    quarters: columns symbol, period_end, filed_date, revenue, net_income, total_equity, long_term_debt (dates as Timestamps).
    A stock whose latest quarter ended more than max_age_days ago gets no fundamentals at all: its statements are
    late or missing in the data, and old figures must not be shown as the latest ones.
    A ratio whose denominator is not positive is left out (NaN), and so are the metrics listed in
    INDUSTRY_EXCLUSIONS for the stock's sub-industry (sub_industries: Series indexed by symbol).
    """
    as_of = pd.Timestamp(as_of_date)
    known = quarters[quarters['filed_date'] <= as_of].sort_values('period_end')
    rows = {}
    for symbol, q in known.groupby('symbol'):
        if symbol not in market_caps.index:
            continue
        latest = q.iloc[-1]
        age = (as_of - latest['period_end']).days
        if age > max_age_days:
            continue
        cap, equity = market_caps[symbol], latest['total_equity']
        row = {'fundamentals_age_days': age, 'filing_age_days': (as_of - latest['filed_date']).days}

        year = q.tail(4)
        span = (year['period_end'].iloc[-1] - year['period_end'].iloc[0]).days
        if len(year) == 4 and 250 <= span <= 300:
            # min_count: a missing quarter makes the sum missing instead of too small
            income, revenue = year['net_income'].sum(min_count=4), year['revenue'].sum(min_count=4)
            row['pe_trailing'] = cap / income if income > 0 else np.nan
            row['net_margin'] = income / revenue if revenue > 0 else np.nan
            row['roe'] = income / equity if equity > 0 else np.nan
        row['pb'] = cap / equity if equity > 0 else np.nan
        row['debt_to_equity'] = latest['long_term_debt'] / equity if equity > 0 else np.nan

        # the same quarter one year before (fiscal quarters end within a few days of each other)
        gap = (latest['period_end'] - q['period_end']).dt.days
        before = q[gap.between(350, 380)]
        if len(before):
            before = before.iloc[-1]
            for metric, item in [('revenue_growth_yoy', 'revenue'), ('net_income_growth_yoy', 'net_income')]:
                row[metric] = latest[item] / before[item] - 1 if before[item] > 0 else np.nan

        for metric in excluded_metrics(sub_industries.get(symbol)):
            row[metric] = np.nan
        rows[symbol] = row
    return pd.DataFrame.from_dict(rows, orient='index')
