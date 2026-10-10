"""
Collect LLM views (expected returns) for S&P500 stocks.
Settings (period, tickers, number of samples, output directory) are read from .env.

    python run.py
"""
from tqdm import tqdm

from llm import config
from llm.returns import month_ranges, to_daily_returns
from llm.context import build_context
from llm.prompts import SYSTEM_PROMPT, make_user_prompt
from llm.client import make_client, sample_expected_returns
from llm.output.files import prepare_dirs, save_prices, save_returns, save_views
from marketdb.sources.prices import download_prices
from marketdb.sources.universe import load_sp500_table

prepare_dirs(config.OUT_DIR)

client = make_client(config.OPENAI_API_KEY)
sp500_table = load_sp500_table(config.TICKERS)

for month_start, month_end in tqdm(month_ranges(config.START_MONTH, config.END_MONTH)):
    print(f"Processing data for {month_start} - {month_end}")
    prices = download_prices(sp500_table['Symbol'].tolist(), month_start, month_end)['Adj Close']
    returns = to_daily_returns(prices)
    save_prices(prices, config.OUT_DIR, month_start, month_end)
    save_returns(returns, config.OUT_DIR, month_start, month_end)

    data_dict = {}
    for ticker in tqdm(returns.columns):
        print("=" * 80)
        print(ticker)
        print("=" * 80)
        context = build_context(ticker, sp500_table, returns)
        context['expected_return'] = sample_expected_returns(
            client, config.OPENAI_MODEL, SYSTEM_PROMPT, make_user_prompt(context), config.N_SAMPLES
        )
        data_dict[ticker] = context

    save_views(data_dict, config.OUT_DIR, config.OPENAI_MODEL, month_start, month_end)
