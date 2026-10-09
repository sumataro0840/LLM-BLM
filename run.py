import numpy as np
import pandas as pd
import yfinance as yf
# from scipy.optimize import minimize
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta
class ResearchPaperExtraction(BaseModel):
    expected_return: float

"""
model
"""
# Load OPENAI_API_KEY and OPENAI_MODEL from .env
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Options for small test runs (defaults reproduce the full experiment)
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=str, default="2024-06-01", help="first month (YYYY-MM-01)")
parser.add_argument("--end", type=str, default="2024-12-01", help="last month (YYYY-MM-01)")
parser.add_argument("--tickers", type=str, default=None, help="comma-separated tickers (default: all S&P500)")
parser.add_argument("--n_samples", type=int, default=30, help="number of LLM queries per ticker")
parser.add_argument("--out_dir", type=str, default=".", help="directory to save yfinance/ and responses/")
args = parser.parse_args()

"""
data
"""

# Create date range from June 2024 to November 2024
date_range = pd.date_range(start=args.start, end=args.end, freq='MS')

# Create directories if they don't exist
yfinance_dir = os.path.join(args.out_dir, 'yfinance')
responses_dir = os.path.join(args.out_dir, 'responses')
os.makedirs(yfinance_dir, exist_ok=True)
os.makedirs(responses_dir, exist_ok=True)

# 1. S&P500 ticker data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia rejects requests without a User-Agent (HTTP 403)
tables = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0 (LLM-BLM research script)'})[0]  # Wikipedia table data
sp500_table = tables[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
if args.tickers:
    sp500_table = sp500_table[sp500_table['Symbol'].isin(args.tickers.split(','))]

for current_date in tqdm(date_range):
    month_start = current_date.strftime('%Y-%m-%d')
    month_end = (current_date + pd.DateOffset(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Processing data for {month_start} - {month_end}")
    
    # Download data using yfinance
    data = yf.download(sp500_table['Symbol'].tolist(), start=month_start, end=month_end)['Close']
    # Save raw data
    data.to_csv(f'{yfinance_dir}/data_{month_start}_{month_end}.csv')
    
    # Process returns
    returns = data.pct_change().iloc[1:]
    returns.to_csv(f'{yfinance_dir}/returns_{month_start}_{month_end}.csv')
    sp500_tickers = returns.columns

    # Organize data
    data_dict = {}
    for ticker in sp500_tickers:
        data_dict[ticker] = {
            'ticker': ticker,
            'Security': sp500_table[sp500_table['Symbol'] == ticker]['Security'].values[0],
            'GICS Sector': sp500_table[sp500_table['Symbol'] == ticker]['GICS Sector'].values[0],
            'GICS Sub-Industry': sp500_table[sp500_table['Symbol'] == ticker]['GICS Sub-Industry'].values[0],
            'pct_change': returns[ticker].tolist()
        }

    def make_system_prompt():
        return """You are a language model designed to predict stock returns. Given a time-series of daily returns as percentage change from the past month, along with the following company information in a dictionary:

        - symbol: The stock symbol
        - company name: The name of the company
        - GICS sector: The Global Industry Classification Standard (GICS) sector
        - GICS sub-industry: The GICS sub-industry

        You should predict the average daily return as a percentage change for the next month. Return a single float value representing the predicted average daily return for the next month. Do not include any additional commentary, explanations, or information. Only output the float value.
        """

    def make_user_prompt(ticker, data_dict):
        return f"""Ticker: {ticker}
        Security: {data_dict[ticker]['Security']}
        Sector: {data_dict[ticker]['GICS Sector']}
        Sub-Industry: {data_dict[ticker]['GICS Sub-Industry']}
        pct_change: {data_dict[ticker]['pct_change']}"""

    for ticker in tqdm(sp500_tickers):
        print("="*80)
        print(ticker)
        print("="*80)
        security = data_dict[ticker]['Security']
        sector = data_dict[ticker]['GICS Sector']
        sub_industry = data_dict[ticker]['GICS Sub-Industry']
        pct_change = data_dict[ticker]['pct_change']
        system_prompt = make_system_prompt()
        user_prompt = make_user_prompt(ticker, data_dict)

        # Get responses 30 times, and use the variance of those values as the confidence of the view.
        answers = []
        for _ in range(args.n_samples):
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "expected_return",
                        "strict": True,
                        "schema": {
                            **ResearchPaperExtraction.model_json_schema(),
                            "additionalProperties": False,
                        },
                    },
                },
            )
            try:
                expected_return_dict = json.loads(completion.choices[0].message.content)
                print(expected_return_dict)
            except:
                # print(completion.choices[0].message.content)
                continue
            answers.append(expected_return_dict['expected_return'])
        data_dict[ticker]['expected_return'] = answers

    # Save responses for this month
    with open(f'{responses_dir}/{model_name}_{month_start}_{month_end}.json', 'w') as f:
        json.dump(data_dict, f)
