"""
Collect LLM views (expected returns) for S&P500 stocks.
Settings (period, tickers, number of samples, output directory) are read from .env.

    python run.py
"""
from tqdm import tqdm

from blm.config import data as data_config
from blm.config import llm as llm_config
from blm.data.universe import load_sp500_table
from blm.data.prices import month_ranges, download_prices, to_daily_returns
from blm.llm.context import build_context
from blm.llm.prompts import SYSTEM_PROMPT, make_user_prompt
from blm.llm.client import make_client, sample_expected_returns
from blm.output.files import prepare_dirs, save_prices, save_returns, save_views

prepare_dirs(data_config.OUT_DIR)

client = make_client(llm_config.OPENAI_API_KEY)
sp500_table = load_sp500_table(data_config.TICKERS)

for month_start, month_end in tqdm(month_ranges(data_config.START_MONTH, data_config.END_MONTH)):
    print(f"Processing data for {month_start} - {month_end}")
    prices = download_prices(sp500_table['Symbol'].tolist(), month_start, month_end)
    returns = to_daily_returns(prices)
    save_prices(prices, data_config.OUT_DIR, month_start, month_end)
    save_returns(returns, data_config.OUT_DIR, month_start, month_end)

    data_dict = {}
    for ticker in tqdm(returns.columns):
        print("=" * 80)
        print(ticker)
        print("=" * 80)
        context = build_context(ticker, sp500_table, returns)
        context['expected_return'] = sample_expected_returns(
            client, llm_config.OPENAI_MODEL, SYSTEM_PROMPT, make_user_prompt(context), llm_config.N_SAMPLES
        )
        data_dict[ticker] = context

    save_views(data_dict, data_config.OUT_DIR, llm_config.OPENAI_MODEL, month_start, month_end)
