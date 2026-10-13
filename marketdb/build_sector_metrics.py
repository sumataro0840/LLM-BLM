"""
Compute the sector indicators for the last trading day of every month and store them in sector_metrics.

    python marketdb/build_sector_metrics.py

Only data known on each date is used: ETF prices up to it and financial statements filed by it.
Run marketdb/build_db.py and marketdb/build_fundamentals.py first. Every run recomputes all dates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marketdb import config
from marketdb.transform.sector import MOMENTUM_MONTHS, momentum, trailing_year, valuation, volatility
from marketdb.db.connection import connect, init_db
from marketdb.db.market import load_prices
from marketdb.db.sector import (load_filed_quarters, load_member_caps, load_month_ends, load_sector_etfs,
                                save_sector_metrics)


def main():
    conn = connect(config.DB_PATH)
    init_db(conn)
    etfs = load_sector_etfs(conn)
    prices = load_prices(conn, list(etfs), config.ETF_START_DATE, config.DB_END_DATE).rename(columns=etfs)
    month_end_prices = prices.groupby(prices.index.to_period('M')).tail(1)

    for date in load_month_ends(conn):
        until = pd.Timestamp(date)
        metrics = pd.DataFrame({f'momentum_{m}m': momentum(month_end_prices.loc[:until], m) for m in MOMENTUM_MONTHS})
        metrics['volatility_30d'] = volatility(prices.loc[:until])
        members = load_member_caps(conn, date)
        metrics = metrics.join(valuation(members['market_cap'], members['gics_sector'],
                                         trailing_year(load_filed_quarters(conn, date))))
        save_sector_metrics(conn, date, metrics)
        print(f"{date}: {metrics.notna().sum().to_dict()}")


if __name__ == "__main__":
    main()
