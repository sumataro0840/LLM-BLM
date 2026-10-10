"""
Collect LLM views (expected returns) for S&P500 stocks and store them in the database.
Settings (months, tickers, number of samples, run to continue) are read from .env.

    python run.py

For each month: the members of the index on the last trading day are asked, using only the daily
returns of that month. Every ticker is saved as soon as it is answered, so an interrupted run can be
continued by setting RUN_ID in .env.
"""
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from llm import config
from llm.returns import month_ranges, to_daily_returns
from llm.context import build_context
from llm.prompts import SYSTEM_PROMPT, INPUT_MONTHS, TARGET_MONTHS, make_user_prompt
from llm.client import make_client, sample_expected_returns
from marketdb.db.connection import connect, init_db
from marketdb.db.index import load_members
from marketdb.db.llm import answered_symbols, create_run, load_run, save_views
from marketdb.db.market import last_trading_day_before, load_prices, load_ticker_table, load_trading_days

client = make_client(config.OPENAI_API_KEY)
conn = connect(config.DB_PATH)
init_db(conn)

if config.RUN_ID is None:
    run_id = create_run(conn, config.OPENAI_MODEL, SYSTEM_PROMPT, config.TEMPERATURE, config.N_SAMPLES,
                        INPUT_MONTHS, TARGET_MONTHS, config.RUN_NOTE)
    print(f"new run: RUN_ID={run_id}")
else:
    run_id = config.RUN_ID
    run = load_run(conn, run_id)
    # a run must hold answers to one question only
    current = {'model': config.OPENAI_MODEL, 'system_prompt': SYSTEM_PROMPT, 'temperature': config.TEMPERATURE,
               'n_samples': config.N_SAMPLES, 'input_months': INPUT_MONTHS, 'target_months': TARGET_MONTHS}
    changed = [key for key, value in current.items() if run[key] != value]
    if changed:
        raise SystemExit(f"RUN_ID={run_id} was started with different settings: {changed}. Clear RUN_ID to start a new run.")
    print(f"continue run: RUN_ID={run_id}")

ticker_table = load_ticker_table(conn)


def ask(context):
    """One ticker -> (answers, model version), or the error when the request failed."""
    try:
        return sample_expected_returns(client, config.OPENAI_MODEL, SYSTEM_PROMPT, make_user_prompt(context),
                                       config.N_SAMPLES, config.TEMPERATURE)
    except Exception as e:
        return e


for month_start, month_end in month_ranges(config.START_MONTH, config.END_MONTH):
    days = load_trading_days(conn, month_start, month_end)
    as_of_date = days[-1]
    members = load_members(conn, as_of_date)
    tickers = [t for t in members if not config.TICKERS or t in config.TICKERS]

    # from the last close of the previous month, so that the first day of the month has a return too
    prices = load_prices(conn, tickers, last_trading_day_before(conn, month_start), as_of_date)
    returns = to_daily_returns(prices)

    done = answered_symbols(conn, run_id, as_of_date)
    contexts = []
    for ticker in tickers:
        if ticker in done:
            continue
        context = build_context(ticker, ticker_table, returns)
        # a stock listed during the month has no returns for its first days
        context['pct_change'] = [r for r in context['pct_change'] if r == r]
        if context['pct_change']:
            contexts.append(context)

    failed = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        results = pool.map(ask, contexts)
        for context, result in tqdm(zip(contexts, results), total=len(contexts), desc=f"{month_start[:7]} (as of {as_of_date})"):
            if isinstance(result, Exception):
                failed.append((context['ticker'], str(result)[:100]))
                continue
            answers, model_version = result
            save_views(conn, run_id, as_of_date, context['ticker'], answers, model_version)
    print(f"{month_start[:7]}: members {len(members)} | asked {len(contexts) - len(failed)} | already done {len(done)} | failed {failed}")

print(f"finished: RUN_ID={run_id} (set it in .env to continue this run)")
