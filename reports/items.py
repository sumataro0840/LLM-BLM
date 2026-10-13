"""
Items read from a monthly report. Change this list to change what the LLM returns:
the question and the shape of the answer are built from it, and every run stores the list it used
(report_runs.items), so a changed list must be run as a new run.

type: 'number' (a figure printed in the report), 'score' (-2 .. +2) or 'text'.
"""
from dataclasses import asdict, dataclass

SECTORS = ['Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials',
           'Health Care', 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Utilities']
STANCE = "from the outlook table (weak / neutral / strong): -2 = weak, -1, 0 = neutral, +1, +2 = strong"


@dataclass(frozen=True)
class Item:
    name: str
    type: str
    description: str


ITEMS = [
    # figures printed in the forecast tables ({year} = year of the report, {next_year} = the year after)
    Item('sp500_latest', 'number', "S&P 500: the latest actual value in the forecast table (the column dated with a month and day); "
                                   "null if the table has no such column"),
    Item('sp500_target_this_year', 'number', "S&P 500: forecast for the end of {year}"),
    Item('sp500_target_next_year', 'number', "S&P 500: forecast for the end of {next_year}"),
    Item('fed_funds_this_year', 'number', "US policy rate, percent: forecast for the end of {year}. "
                                          "It is printed as a range; answer the lower number of the range"),
    Item('fed_funds_next_year', 'number', "US policy rate, percent: forecast for the end of {next_year} (the lower number of the range)"),
    Item('us_10y_yield_this_year', 'number', "US 10-year government bond yield, percent: forecast for the end of {year}"),
    Item('us_gdp_growth_this_year', 'number', "US real GDP growth, percent: forecast for the calendar year {year}"),
    Item('us_gdp_growth_next_year', 'number', "US real GDP growth, percent: forecast for the calendar year {next_year}"),
    # the publisher's own ratings
    Item('us_economy_stance', 'score', f"Outlook for the US economy {STANCE}"),
    Item('us_equity_stance', 'score', f"Outlook for US equities {STANCE}"),
    Item('us_bond_stance', 'score', f"Outlook for US bonds {STANCE}"),
    # judged from the text
    *[Item(f"sector_{s.lower().replace(' ', '_')}", 'score',
           f"Outlook for the US {s} sector as the text describes it: -2 = clearly negative .. +2 = clearly positive")
      for s in SECTORS],
    Item('us_summary', 'text', "The outlook for the US economy and US equities, in English, at most 100 words"),
    Item('key_risks', 'text', "The main risks to US equities named in the report, in English, at most 50 words"),
]


def items_as_dicts():
    return [asdict(item) for item in ITEMS]
