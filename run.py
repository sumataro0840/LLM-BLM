"""
Collect LLM views (expected returns) for S&P500 stocks.

    python run.py                                  # full experiment
    python run.py --start 2024-06-01 --end 2024-07-01 \
        --tickers AAPL,NVDA --n_samples 5 --out_dir ../test_run   # small test
"""
import argparse
import json
import os

from tqdm import tqdm

from blm.config import data as data_config
from blm.config import llm as llm_config
from blm.data.universe import load_sp500_table
from blm.data.prices import month_ranges, load_monthly_returns
from blm.llm.context import build_context
from blm.llm.prompts import SYSTEM_PROMPT, make_user_prompt
from blm.llm.client import make_client, sample_expected_returns

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=str, default=data_config.START_MONTH, help="first month (YYYY-MM-01)")
parser.add_argument("--end", type=str, default=data_config.END_MONTH, help="last month (YYYY-MM-01)")
parser.add_argument("--tickers", type=str, default=None, help="comma-separated tickers (default: all S&P500)")
parser.add_argument("--n_samples", type=int, default=llm_config.N_SAMPLES, help="number of LLM queries per ticker")
parser.add_argument("--out_dir", type=str, default=data_config.OUT_DIR, help="directory to save yfinance/ and responses/")
args = parser.parse_args()

yfinance_dir = os.path.join(args.out_dir, 'yfinance')
responses_dir = os.path.join(args.out_dir, 'responses')
os.makedirs(yfinance_dir, exist_ok=True)
os.makedirs(responses_dir, exist_ok=True)

client = make_client(llm_config.OPENAI_API_KEY)
sp500_table = load_sp500_table(args.tickers.split(',') if args.tickers else None)

for month_start, month_end in tqdm(month_ranges(args.start, args.end)):
    print(f"Processing data for {month_start} - {month_end}")
    returns = load_monthly_returns(sp500_table['Symbol'].tolist(), month_start, month_end, yfinance_dir)

    data_dict = {}
    for ticker in tqdm(returns.columns):
        print("=" * 80)
        print(ticker)
        print("=" * 80)
        context = build_context(ticker, sp500_table, returns)
        context['expected_return'] = sample_expected_returns(
            client, llm_config.OPENAI_MODEL, SYSTEM_PROMPT, make_user_prompt(context), args.n_samples
        )
        data_dict[ticker] = context

    # Save responses for this month
    with open(f'{responses_dir}/{llm_config.OPENAI_MODEL}_{month_start}_{month_end}.json', 'w') as f:
        json.dump(data_dict, f)
