"""
Compute the stock indicators of the index members for the last trading day of every month
and store them in stock_metrics.

    python marketdb/build_stock_metrics.py

Only data known on each date is used: prices up to it and financial statements filed by it.
Run marketdb/build_db.py and marketdb/build_fundamentals.py first. Every run recomputes all dates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marketdb import config
from marketdb.transform.stock import MONTH_DAYS, YEAR_DAYS, fundamental_metrics, price_metrics
from marketdb.db.connection import connect, init_db
from marketdb.db.market import last_trading_days, load_prices
from marketdb.db.sector import load_member_caps, load_month_ends, load_sector_etfs
from marketdb.db.stock import load_quarters, save_stock_metrics


def main():
    conn = connect(config.DB_PATH)
    init_db(conn)
    etfs = load_sector_etfs(conn)
    quarters = load_quarters(conn)

    for date in load_month_ends(conn):
        members = load_member_caps(conn, date)
        # one day more than a year of returns
        days = last_trading_days(conn, date, YEAR_DAYS + 1)
        if len(days) <= MONTH_DAYS:
            # the first month of the database: not a month of returns yet
            print(f"{date}: skipped (only {len(days)} trading days of prices)")
            continue
        prices = load_prices(conn, [*members.index, *etfs, '^GSPC'], days[0], date)
        metrics = price_metrics(prices[prices.columns.intersection(members.index)], prices['^GSPC'],
                                # XLRE (2015-10) and XLC (2018-06) did not trade before: their sectors get NaN
                                prices.reindex(columns=list(etfs)).rename(columns=etfs), members['gics_sector'])
        metrics = metrics.join(fundamental_metrics(quarters, members['market_cap'], members['gics_sub_industry'], date,
                                                   config.FUNDAMENTALS_MAX_AGE_DAYS))
        save_stock_metrics(conn, date, metrics)
        print(f"{date}: members {len(members)} | {metrics.notna().sum().to_dict()}")


if __name__ == "__main__":
    main()
