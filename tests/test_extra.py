"""
The readers of the additional data on synthetic inputs: the French CSV, FINRA's symbols and FRED's public dates.
"""
import unittest

from marketdb.sources import extra as src

DAILY = """This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.02,   -0.35,    0.03,    0.13,    0.012
19630702,    0.79,   -0.27,    0.27,   -0.08,   -0.21,    0.012
"""

MONTHLY = """This file was created by CMPT_ME_BEME_RETS using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.41,   -0.97,    0.68,   -1.18,    0.27
196308,    5.07,   -0.80,    1.80,    0.36,   -0.35,    0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
1964,   12.59,    0.34,    9.20,    1.52,    6.14,    3.54
"""


class TestExtra(unittest.TestCase):
    def test_french_daily(self):
        df = src.parse_french(DAILY, 'daily')
        self.assertEqual(len(df), 12)
        row = df[(df.date == '1963-07-01') & (df.factor == 'Mkt-RF')]
        self.assertAlmostEqual(float(row.value.iloc[0]), -0.0067)

    def test_french_monthly_stops_before_annual(self):
        df = src.parse_french(MONTHLY, 'monthly')
        self.assertEqual(sorted(df.date.unique()), ['1963-07-31', '1963-08-31'])
        self.assertAlmostEqual(float(df[(df.date == '1963-08-31') & (df.factor == 'Mkt-RF')].value.iloc[0]), 0.0507)

    def test_finra_symbols(self):
        self.assertEqual(src.finra_symbol('BRK.B'), 'BRK-B')
        self.assertEqual(src.finra_symbol('BF/B'), 'BF-B')
        self.assertEqual(src.finra_symbol('AAPL'), 'AAPL')

    def test_public_dates(self):
        self.assertEqual(list(src.available_date(['2024-03-08'], 'daily')), ['2024-03-09'])
        self.assertEqual(list(src.available_date(['2024-03-02'], 'weekly')), ['2024-03-08'])
        self.assertEqual(list(src.available_date(['2024-01-01'], 'monthly')), ['2024-02-29'])  # January's value


if __name__ == '__main__':
    unittest.main()
