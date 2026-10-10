"""
Prompts for predicting the next month's average daily return.
(Same as the original implementation, plus the sentences that fix the unit:
the original says "percentage change" while the values are decimal fractions.)
"""

# What the prompt asks for: daily returns of the past INPUT_MONTHS -> average daily return of the next TARGET_MONTHS
INPUT_MONTHS = 1
TARGET_MONTHS = 1

SYSTEM_PROMPT = """You are a language model designed to predict stock returns. Given a time-series of daily returns from the past month, along with the following company information in a dictionary:

        - symbol: The stock symbol
        - company name: The name of the company
        - GICS sector: The Global Industry Classification Standard (GICS) sector
        - GICS sub-industry: The GICS sub-industry

        Every return is a decimal fraction, not a percent: 0.01 means +1% and -0.005 means -0.5%.

        You should predict the average daily return for the next month. Return a single float value representing the predicted average daily return for the next month, as a decimal fraction in the same unit as the input (answer 0.001 for +0.1% per day, never 0.1). Do not include any additional commentary, explanations, or information. Only output the float value.
        """

"""
Input: A dictionary containing the following keys:
- ticker: The stock symbol
- Security: The name of the company
- GICS Sector: The Global Industry Classification Standard (GICS) sector
- GICS Sub-Industry: The GICS sub-industry
- pct_change: A list of daily returns for the past month, as decimal fractions (e.g., 0.01 for +1%, -0.005 for -0.5%)
"""
def make_user_prompt(context):
    return f"""Ticker: {context['ticker']}
        Security: {context['Security']}
        Sector: {context['GICS Sector']}
        Sub-Industry: {context['GICS Sub-Industry']}
        daily returns (decimal fraction): {context['pct_change']}"""
