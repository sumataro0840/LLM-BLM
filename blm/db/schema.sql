-- LLM-BLM database schema (SQLite)
-- Dates are stored as ISO 8601 text: date = 'YYYY-MM-DD', month = 'YYYY-MM', *_at = 'YYYY-MM-DDTHH:MM:SS+00:00' (UTC)
-- Symbols use the yfinance format ('BRK-B', not Wikipedia's 'BRK.B')
-- Returns and market caps are computed from these tables, not stored
-- Split basis: everything is on the split basis of the fetch date (yfinance prices already are;
-- shares are converted by scripts/build_db.py), so market cap = close x shares

-- ============================================================
-- Market data
-- ============================================================

-- Every stock that was in the S&P500 at any time in the study period, plus indices such as ^GSPC
CREATE TABLE IF NOT EXISTS tickers (
    symbol            TEXT PRIMARY KEY,         -- e.g. 'AAPL', 'BRK-B', '^GSPC'
    security          TEXT NOT NULL,            -- company / index name
    gics_sector       TEXT,                     -- NULL for indices and stocks no longer listed on Wikipedia
    gics_sub_industry TEXT,
    excluded_reason   TEXT,                     -- NULL = used; otherwise why it is left out (e.g. no price data)
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
-- LLM results
-- ============================================================

-- One execution of run.py (model and prompt are kept for reproducibility)
CREATE TABLE IF NOT EXISTS llm_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model         TEXT NOT NULL,                -- e.g. 'gpt-4o-mini'
    system_prompt TEXT NOT NULL,
    n_samples     INTEGER NOT NULL CHECK (n_samples > 0),
    note          TEXT,                         -- free memo (what was changed in this run)
    created_at    TEXT NOT NULL
);

-- Each LLM answer: the view on the month after `month`
CREATE TABLE IF NOT EXISTS llm_views (
    run_id          INTEGER NOT NULL REFERENCES llm_runs(run_id),
    month           TEXT NOT NULL,              -- month of the input returns (the prediction is for the next month)
    symbol          TEXT NOT NULL REFERENCES tickers(symbol),
    sample_no       INTEGER NOT NULL,           -- 1 .. n_samples
    expected_return REAL,                       -- NULL when the answer could not be parsed
    PRIMARY KEY (run_id, month, symbol, sample_no)
);

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
