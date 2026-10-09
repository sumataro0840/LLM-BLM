"""
File naming rules and save functions for yfinance/ and responses/.
(The names are kept identical to the original implementation.)
"""
import json
import os

PRICES_DIR = 'yfinance'
RESPONSES_DIR = 'responses'


def prepare_dirs(out_dir):
    os.makedirs(os.path.join(out_dir, PRICES_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_dir, RESPONSES_DIR), exist_ok=True)


def prices_path(out_dir, month_start, month_end):
    return os.path.join(out_dir, PRICES_DIR, f'data_{month_start}_{month_end}.csv')


def returns_path(out_dir, month_start, month_end):
    return os.path.join(out_dir, PRICES_DIR, f'returns_{month_start}_{month_end}.csv')


def views_path(out_dir, model_name, month_start, month_end):
    return os.path.join(out_dir, RESPONSES_DIR, f'{model_name}_{month_start}_{month_end}.json')


def save_prices(prices, out_dir, month_start, month_end):
    prices.to_csv(prices_path(out_dir, month_start, month_end))


def save_returns(returns, out_dir, month_start, month_end):
    returns.to_csv(returns_path(out_dir, month_start, month_end))


def save_views(views, out_dir, model_name, month_start, month_end):
    with open(views_path(out_dir, model_name, month_start, month_end), 'w') as f:
        json.dump(views, f)
