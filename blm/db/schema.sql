-- LLM-BLM database schema (SQLite)
-- Dates are stored as ISO 8601 text: date = 'YYYY-MM-DD', month = 'YYYY-MM', *_at = 'YYYY-MM-DDTHH:MM:SS+00:00' (UTC)

-- Stocks and indices (S&P500 constituents from Wikipedia, plus indices such as ^GSPC)
CREATE TABLE IF NOT EXISTS tickers (
    symbol            TEXT PRIMARY KEY,         -- e.g. 'AAPL', '^GSPC'
    security          TEXT NOT NULL,            -- company / index name
    gics_sector       TEXT,                     -- NULL for indices
    gics_sub_industry TEXT,                     -- NULL for indices
    fetched_at        TEXT NOT NULL             -- when this row was fetched
);

-- Daily close prices from yfinance (returns are computed from these, not stored)
CREATE TABLE IF NOT EXISTS prices (
    symbol     TEXT NOT NULL REFERENCES tickers(symbol),
    date       TEXT NOT NULL,                   -- trading day
    close      REAL NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;

-- For queries over a period across all stocks (WHERE date BETWEEN ...)
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

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
