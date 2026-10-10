"""
Compute portfolio weights for every decision date of an LLM run and store them in the database.
Settings (strategies, coefficients, LLM run) are read from .env.

    python run_blm.py

Every strategy gets the same universe on each date. Each execution stores new portfolios
(portfolio_runs); the returns are read from the view portfolio_returns.
"""
from tqdm import tqdm

from blm import config
from blm.inputs import load_inputs
from blm.strategies import COMMON_PARAMS, STRATEGIES
from marketdb.db.connection import connect, init_db
from marketdb.db.llm import load_as_of_dates
from marketdb.db.portfolio import create_portfolio, save_weights

conn = connect(config.DB_PATH)
init_db(conn)

portfolio_ids = {}
for name in config.STRATEGIES:
    _, uses_views, params = STRATEGIES[name]
    portfolio_ids[name] = create_portfolio(conn, name, config.LLM_RUN_ID if uses_views else None,
                                           {**COMMON_PARAMS, **params}, config.NOTE)
print(f"portfolios: {portfolio_ids}")

for as_of_date in tqdm(load_as_of_dates(conn, config.LLM_RUN_ID)):
    inputs = load_inputs(conn, as_of_date, config.LLM_RUN_ID, config.COV_WINDOW_DAYS, config.MIN_HISTORY_RATIO)
    for name in config.STRATEGIES:
        weights = STRATEGIES[name][0](inputs)
        save_weights(conn, portfolio_ids[name], as_of_date, weights)

print(f"finished: portfolio_id {min(portfolio_ids.values())} .. {max(portfolio_ids.values())}")
