-- LLM-BLM database schema (SQLite)
-- Dates are stored as ISO 8601 text: date = 'YYYY-MM-DD', *_at = 'YYYY-MM-DDTHH:MM:SS+00:00' (UTC)
-- Symbols use the yfinance format ('BRK-B', not Wikipedia's 'BRK.B')
-- Returns and market caps are computed from these tables, not stored
-- Split basis: everything is on the split basis of the fetch date (yfinance prices already are;
-- shares are converted by marketdb/build_db.py), so market cap = close x shares

-- ============================================================
-- Market data
-- ============================================================

-- Every stock that was in the S&P500 at any time in the study period, plus indices such as ^GSPC
-- and one ETF per sector (stocks are the symbols that have a row in index_membership)
CREATE TABLE IF NOT EXISTS tickers (
    symbol            TEXT PRIMARY KEY,         -- e.g. 'AAPL', 'BRK-B', '^GSPC'
    security          TEXT NOT NULL,            -- company / index name
    gics_sector       TEXT,                     -- NULL for indices; for a sector ETF, the sector it tracks
    gics_sub_industry TEXT,
    excluded_reason   TEXT,                     -- NULL = used; otherwise why it is left out (e.g. no price data)
    cik               INTEGER,                  -- SEC EDGAR company id (NULL = not found)
    fundamentals_fetched_at TEXT,               -- when financial_facts were fetched from EDGAR
    fetched_at        TEXT NOT NULL
);

-- Daily prices from yfinance (auto_adjust=False)
CREATE TABLE IF NOT EXISTS prices (
    symbol     TEXT NOT NULL REFERENCES tickers(symbol),
    date       TEXT NOT NULL,                   -- trading day
    close      REAL NOT NULL,                   -- split-adjusted only (for market cap)
    adj_close  REAL NOT NULL,                   -- split- and dividend-adjusted (for returns)
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

-- Shares outstanding converted to the split basis of the fetch date (market cap = close x shares)
-- yfinance reports raw shares, sometimes several per day; build_db.py adjusts them for later splits,
-- drops outliers and keeps the daily median
CREATE TABLE IF NOT EXISTS shares_outstanding (
    symbol     TEXT NOT NULL REFERENCES tickers(symbol),
    date       TEXT NOT NULL,                   -- report date
    shares     INTEGER NOT NULL CHECK (shares > 0),
    fetched_at TEXT NOT NULL,                   -- = the split basis date
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;

-- ============================================================
-- S&P500 membership
-- ============================================================

-- Constituent changes as listed on Wikipedia ("Historical components of the S&P 500")
CREATE TABLE IF NOT EXISTS index_changes (
    change_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,              -- effective date
    added_symbol    TEXT,                       -- NULL when only a removal
    added_name      TEXT,
    removed_symbol  TEXT,                       -- NULL when only an addition
    removed_name    TEXT,
    reason          TEXT,
    fetched_at      TEXT NOT NULL,
    UNIQUE (date, added_symbol, removed_symbol)
);

-- Membership periods built from index_changes and the current constituents
-- A stock is a member on day D when start_date <= D < end_date (NULL = open-ended)
CREATE TABLE IF NOT EXISTS index_membership (
    membership_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL REFERENCES tickers(symbol),
    start_date    TEXT,                         -- NULL: joined before the oldest recorded change
    end_date      TEXT,                         -- NULL: still a member
    UNIQUE (symbol, start_date),
    CHECK (start_date IS NULL OR end_date IS NULL OR start_date < end_date)
);

CREATE INDEX IF NOT EXISTS idx_membership_symbol ON index_membership(symbol);

-- ============================================================
-- Financial statements (SEC EDGAR XBRL "company facts")
-- ============================================================

-- Reported items: one row per XBRL tag and unit (e.g. us-gaap / NetIncomeLoss / USD)
CREATE TABLE IF NOT EXISTS financial_items (
    item_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    taxonomy TEXT NOT NULL,                     -- 'us-gaap', 'dei', ...
    tag      TEXT NOT NULL,                     -- XBRL tag
    unit     TEXT NOT NULL,                     -- 'USD', 'shares', 'USD/shares', ...
    label    TEXT,                              -- human-readable name from EDGAR
    UNIQUE (taxonomy, tag, unit)
);

-- Reported values as first filed (later restatements of the same period are not kept)
-- Flow items cover period_start .. period_end (3, 6, 9 or 12 months); balance sheet items have period_start = period_end
CREATE TABLE IF NOT EXISTS financial_facts (
    symbol       TEXT NOT NULL REFERENCES tickers(symbol),
    item_id      INTEGER NOT NULL REFERENCES financial_items(item_id),
    period_start TEXT NOT NULL,
    period_end   TEXT NOT NULL,
    value        REAL NOT NULL,
    form         TEXT NOT NULL,                 -- '10-Q', '10-K', ...
    filed_date   TEXT NOT NULL,                 -- when the value became public
    PRIMARY KEY (item_id, symbol, period_start, period_end)   -- item first: the views look values up by item
) WITHOUT ROWID;

-- Quarterly figures of the main items, one row per company and fiscal quarter
-- (3-month values: reported directly, or the difference of two year-to-date values; Q4 = full year - 9 months;
--  a quarter is 80 to 120 days long, which covers 12-, 13- and 16-week fiscal quarters)
CREATE VIEW IF NOT EXISTS fundamentals_quarterly AS
WITH concept(concept, kind, tag, priority) AS (VALUES
    -- kind: 'flow' (additive over time), 'average' (per-quarter only, not additive), 'stock' (balance sheet)
    ('revenue',           'flow',    'Revenues', 1),
    ('revenue',           'flow',    'RevenuesNetOfInterestExpense', 2),
    ('revenue',           'flow',    'RevenueFromContractWithCustomerExcludingAssessedTax', 3),
    ('revenue',           'flow',    'RevenueFromContractWithCustomerIncludingAssessedTax', 4),
    ('revenue',           'flow',    'SalesRevenueNet', 5),
    ('cost_of_revenue',   'flow',    'CostOfGoodsAndServicesSold', 1),
    ('cost_of_revenue',   'flow',    'CostOfRevenue', 2),
    ('gross_profit',      'flow',    'GrossProfit', 1),
    ('rd_expense',        'flow',    'ResearchAndDevelopmentExpense', 1),
    ('operating_income',  'flow',    'OperatingIncomeLoss', 1),
    ('pretax_income',     'flow',    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1),
    ('pretax_income',     'flow',    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments', 2),
    ('net_income',        'flow',    'NetIncomeLoss', 1),
    ('net_income',        'flow',    'ProfitLoss', 2),
    ('eps_basic',         'flow',    'EarningsPerShareBasic', 1),
    ('eps_diluted',       'flow',    'EarningsPerShareDiluted', 1),
    ('dividend_per_share','flow',    'CommonStockDividendsPerShareDeclared', 1),
    ('shares_diluted_avg','average', 'WeightedAverageNumberOfDilutedSharesOutstanding', 1),
    ('operating_cf',      'flow',    'NetCashProvidedByUsedInOperatingActivities', 1),
    ('capex',             'flow',    'PaymentsToAcquirePropertyPlantAndEquipment', 1),
    ('dividends_paid',    'flow',    'PaymentsOfDividends', 1),
    ('dividends_paid',    'flow',    'PaymentsOfDividendsCommonStock', 2),
    ('buybacks',          'flow',    'PaymentsForRepurchaseOfCommonStock', 1),
    ('total_assets',      'stock',   'Assets', 1),
    ('current_assets',    'stock',   'AssetsCurrent', 1),
    ('cash',              'stock',   'CashAndCashEquivalentsAtCarryingValue', 1),
    ('cash',              'stock',   'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 2),
    ('total_liabilities', 'stock',   'Liabilities', 1),
    ('current_liabilities','stock',  'LiabilitiesCurrent', 1),
    ('long_term_debt',    'stock',   'LongTermDebtNoncurrent', 1),
    ('long_term_debt',    'stock',   'LongTermDebt', 2),
    ('total_equity',      'stock',   'StockholdersEquity', 1),
    ('total_equity',      'stock',   'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 2)
),
picked AS (
    SELECT f.symbol, f.item_id, c.concept, c.kind, c.priority, f.period_start, f.period_end, f.value, f.filed_date,
           julianday(f.period_end) - julianday(f.period_start) AS days
    FROM concept c
    JOIN financial_items i ON i.tag = c.tag AND i.taxonomy = 'us-gaap'
    JOIN financial_facts f ON f.item_id = i.item_id
),
quarter_values AS (
    -- reported for 3 months
    SELECT symbol, concept, priority, 0 AS derived, period_end, value, filed_date
    FROM picked WHERE kind IN ('flow', 'average') AND days BETWEEN 80 AND 120
    UNION ALL
    -- year-to-date value minus the previous year-to-date value of the same fiscal year
    SELECT a.symbol, a.concept, a.priority, 1, a.period_end, a.value - b.value, a.filed_date
    FROM picked a
    JOIN financial_facts b ON b.item_id = a.item_id AND b.symbol = a.symbol AND b.period_start = a.period_start
                          AND b.period_end BETWEEN date(a.period_end, '-120 days') AND date(a.period_end, '-80 days')
    WHERE a.kind = 'flow' AND a.days > 120
    UNION ALL
    -- balance sheet values at the quarter end
    SELECT symbol, concept, priority, 0, period_end, value, filed_date
    FROM picked WHERE kind = 'stock'
),
best AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, concept, period_end ORDER BY derived, priority) AS choice
    FROM quarter_values
)
SELECT symbol,
       period_end,
       MIN(CASE WHEN concept IN ('revenue', 'net_income') THEN filed_date END) AS filed_date,
       MAX(CASE WHEN concept = 'revenue'             THEN value END) AS revenue,
       MAX(CASE WHEN concept = 'cost_of_revenue'     THEN value END) AS cost_of_revenue,
       MAX(CASE WHEN concept = 'gross_profit'        THEN value END) AS gross_profit,
       MAX(CASE WHEN concept = 'rd_expense'          THEN value END) AS rd_expense,
       MAX(CASE WHEN concept = 'operating_income'    THEN value END) AS operating_income,
       MAX(CASE WHEN concept = 'pretax_income'       THEN value END) AS pretax_income,
       MAX(CASE WHEN concept = 'net_income'          THEN value END) AS net_income,
       MAX(CASE WHEN concept = 'eps_basic'           THEN value END) AS eps_basic,
       MAX(CASE WHEN concept = 'eps_diluted'         THEN value END) AS eps_diluted,
       MAX(CASE WHEN concept = 'dividend_per_share'  THEN value END) AS dividend_per_share,
       MAX(CASE WHEN concept = 'shares_diluted_avg'  THEN value END) AS shares_diluted_avg,
       MAX(CASE WHEN concept = 'operating_cf'        THEN value END) AS operating_cf,
       MAX(CASE WHEN concept = 'capex'               THEN value END) AS capex,
       MAX(CASE WHEN concept = 'dividends_paid'      THEN value END) AS dividends_paid,
       MAX(CASE WHEN concept = 'buybacks'            THEN value END) AS buybacks,
       MAX(CASE WHEN concept = 'total_assets'        THEN value END) AS total_assets,
       MAX(CASE WHEN concept = 'current_assets'      THEN value END) AS current_assets,
       MAX(CASE WHEN concept = 'cash'                THEN value END) AS cash,
       MAX(CASE WHEN concept = 'total_liabilities'   THEN value END) AS total_liabilities,
       MAX(CASE WHEN concept = 'current_liabilities' THEN value END) AS current_liabilities,
       MAX(CASE WHEN concept = 'long_term_debt'      THEN value END) AS long_term_debt,
       MAX(CASE WHEN concept = 'total_equity'        THEN value END) AS total_equity
FROM best
WHERE choice = 1
GROUP BY symbol, period_end
HAVING filed_date IS NOT NULL;

-- ============================================================
-- LLM results
-- ============================================================

-- One execution of run.py (everything that defines the question is kept for reproducibility)
CREATE TABLE IF NOT EXISTS llm_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model         TEXT NOT NULL,                -- requested model, e.g. 'gpt-4o-mini'
    model_version TEXT,                         -- model that actually answered, e.g. 'gpt-4o-mini-2024-07-18'
    system_prompt TEXT NOT NULL,
    temperature   REAL NOT NULL,
    n_samples     INTEGER NOT NULL CHECK (n_samples > 0),
    input_months  INTEGER NOT NULL,             -- length of the daily returns given to the LLM (1 = one month, 3 = a quarter)
    target_months INTEGER NOT NULL,             -- length of the period the LLM predicts
    note          TEXT,                         -- free memo (what was changed in this run)
    created_at    TEXT NOT NULL
);

-- Each LLM answer: the view on the target_months that follow as_of_date
CREATE TABLE IF NOT EXISTS llm_views (
    run_id          INTEGER NOT NULL REFERENCES llm_runs(run_id),
    as_of_date      TEXT NOT NULL,              -- last trading day of the input returns (nothing after it was shown)
    symbol          TEXT NOT NULL REFERENCES tickers(symbol),
    sample_no       INTEGER NOT NULL,           -- 1 .. n_samples
    expected_return REAL,                       -- average daily return as a decimal fraction; NULL = answer not parsed
    PRIMARY KEY (run_id, as_of_date, symbol, sample_no)
) WITHOUT ROWID;

-- ============================================================
-- Portfolios
-- ============================================================

-- One strategy computed by run_blm.py (the coefficients used are kept for reproducibility)
CREATE TABLE IF NOT EXISTS portfolio_runs (
    portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy     TEXT NOT NULL,                 -- 'market_cap', 'equal_weight', 'mean_variance', 'black_litterman', ...
    llm_run_id   INTEGER REFERENCES llm_runs(run_id),  -- LLM answers used as views (NULL = none)
    params       TEXT NOT NULL,                 -- JSON of the coefficients, e.g. {"tau": 0.025, "risk_aversion": 2.5}
    note         TEXT,
    created_at   TEXT NOT NULL
);

-- Weights decided on as_of_date and held until the last trading day of the following month
CREATE TABLE IF NOT EXISTS portfolio_weights (
    portfolio_id INTEGER NOT NULL REFERENCES portfolio_runs(portfolio_id),
    as_of_date   TEXT NOT NULL,                 -- decision date (nothing after it was used)
    symbol       TEXT NOT NULL REFERENCES tickers(symbol),
    weight       REAL NOT NULL CHECK (weight >= 0),  -- every stock of the universe has a row (0 = not held); sums to 1
    PRIMARY KEY (portfolio_id, as_of_date, symbol)
) WITHOUT ROWID;

-- ============================================================
-- Sectors
-- ============================================================

-- Indicators of each GICS sector on a decision date, computed by marketdb/build_sector_metrics.py
-- from data known on that date (ETF prices up to it, financial statements filed by it)
CREATE TABLE IF NOT EXISTS sector_metrics (
    as_of_date  TEXT NOT NULL,                  -- last trading day of a month (nothing after it was used)
    gics_sector TEXT NOT NULL,                  -- same names as tickers.gics_sector
    metric      TEXT NOT NULL,                  -- 'momentum_3m' / 'momentum_6m' / 'momentum_12m': ETF return over that many months, the latest month left out
                                                -- 'volatility_30d': yearly volatility of the ETF's daily returns over 30 trading days (mean of the last 5 days)
                                                -- 'pe_trailing' / 'pb' / 'ps': market cap of the members / their net income, equity, revenue (last 4 quarters)
                                                -- 'valuation_coverage': share of the sector's market cap that has the 4 quarters needed for pe_trailing
    value       REAL NOT NULL,                  -- returns and volatility are decimal fractions (0.05 = 5%)
    computed_at TEXT NOT NULL,
    PRIMARY KEY (as_of_date, gics_sector, metric)
) WITHOUT ROWID;

-- ============================================================
-- Monthly reports
-- ============================================================

-- One report file in data/reports (the file itself is not stored in the database)
CREATE TABLE IF NOT EXISTS reports (
    report_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source         TEXT NOT NULL,               -- publisher, e.g. 'daiwa-am'
    report_month   TEXT NOT NULL,               -- month on the cover, 'YYYY-MM'
    published_date TEXT NOT NULL,               -- creation date of the PDF (the report can be used from this date on)
    file_name      TEXT NOT NULL,               -- e.g. 'daiwa-am_2024-04.pdf'
    file_sha256    TEXT NOT NULL,               -- to notice a replaced file
    n_pages        INTEGER NOT NULL CHECK (n_pages > 0),
    registered_at  TEXT NOT NULL,
    UNIQUE (source, report_month)
);

-- One analysis of the reports by the LLM; changing the items or the prompt starts a new run
CREATE TABLE IF NOT EXISTS report_runs (
    report_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model         TEXT NOT NULL,                -- requested model, e.g. 'gpt-4o-mini'
    model_version TEXT,                         -- model that actually answered
    system_prompt TEXT NOT NULL,
    items         TEXT NOT NULL,                -- JSON list of the items asked: [{"name": ..., "type": ..., "description": ...}]
    temperature   REAL NOT NULL,
    note          TEXT,                         -- free memo (what was changed in this run)
    created_at    TEXT NOT NULL
);

-- What the LLM read from a report: one row per item
CREATE TABLE IF NOT EXISTS report_values (
    report_run_id INTEGER NOT NULL REFERENCES report_runs(report_run_id),
    report_id     INTEGER NOT NULL REFERENCES reports(report_id),
    item          TEXT NOT NULL,                -- item name listed in report_runs.items, e.g. 'us_equity_stance'
    value_num     REAL,                         -- numbers and scores
    value_text    TEXT,                         -- summaries and other text
    -- both NULL = the report does not mention the item
    CHECK (value_num IS NULL OR value_text IS NULL),
    PRIMARY KEY (report_run_id, report_id, item)
) WITHOUT ROWID;

-- ============================================================
-- Views (computed when read; nothing is stored)
-- ============================================================

-- Market cap for every trading day, using the latest shares report on or before that day
CREATE VIEW IF NOT EXISTS market_caps AS
SELECT p.symbol,
       p.date,
       p.close,
       s.shares,
       s.date            AS shares_reported_on,
       p.close * s.shares AS market_cap
FROM prices p
JOIN shares_outstanding s
  ON s.symbol = p.symbol
 AND s.date = (SELECT MAX(date) FROM shares_outstanding
               WHERE symbol = p.symbol AND date <= p.date);

-- Daily index weights: market cap share among the S&P500 members of that day
-- (excluded stocks and days without a shares report are left out, so the weights of a day sum to 1)
CREATE VIEW IF NOT EXISTS index_weights AS
SELECT mc.date,
       mc.symbol,
       mc.market_cap,
       mc.market_cap / SUM(mc.market_cap) OVER (PARTITION BY mc.date) AS weight
FROM market_caps mc
JOIN tickers t
  ON t.symbol = mc.symbol AND t.excluded_reason IS NULL
JOIN index_membership m
  ON m.symbol = mc.symbol
 AND (m.start_date IS NULL OR m.start_date <= mc.date)
 AND (m.end_date IS NULL OR m.end_date > mc.date);

-- Black-Litterman inputs from the LLM answers: mean = view (Q), variance = uncertainty of the view (Omega)
CREATE VIEW IF NOT EXISTS llm_view_stats AS
SELECT run_id,
       as_of_date,
       symbol,
       COUNT(expected_return) AS n_answers,     -- parsed answers (n_samples when nothing failed)
       AVG(expected_return)   AS mean_return,
       AVG(expected_return * expected_return) - AVG(expected_return) * AVG(expected_return) AS variance  -- population variance (same as numpy.var)
FROM llm_views
GROUP BY run_id, as_of_date, symbol;

-- Return of each portfolio over the month after the decision date (buy on as_of_date, hold, no rebalancing)
CREATE VIEW IF NOT EXISTS portfolio_returns AS
WITH period AS (
    SELECT DISTINCT as_of_date,
           (SELECT MAX(date) FROM prices
             WHERE symbol = '^GSPC' AND date <= date(as_of_date, 'start of month', '+2 months', '-1 day')) AS end_date
    FROM portfolio_weights
)
SELECT w.portfolio_id,
       w.as_of_date,
       p.end_date,
       SUM(w.weight > 0) AS n_holdings,
       SUM(w.weight * ((SELECT adj_close FROM prices WHERE symbol = w.symbol AND date <= p.end_date ORDER BY date DESC LIMIT 1)
                       / (SELECT adj_close FROM prices WHERE symbol = w.symbol AND date = w.as_of_date) - 1)) AS period_return
FROM portfolio_weights w
JOIN period p ON p.as_of_date = w.as_of_date
GROUP BY w.portfolio_id, w.as_of_date;

-- Sector indicators with their z-score among the sectors of the same date
-- (how far a sector is from the average of the sectors, in standard deviations)
CREATE VIEW IF NOT EXISTS sector_metric_scores AS
SELECT as_of_date,
       gics_sector,
       metric,
       value,
       (value - AVG(value) OVER w)
         / NULLIF(sqrt(AVG(value * value) OVER w - AVG(value) OVER w * AVG(value) OVER w), 0) AS z_score
FROM sector_metrics
WINDOW w AS (PARTITION BY as_of_date, metric);
