"""
Save / load earnings releases, their scores and surprises
(earnings_releases / release_score_runs / release_scores / release_score_calls / release_surprises).
"""
import hashlib
import json

import pandas as pd

from marketdb.db.market import now_utc

RELEASE_COLUMNS = ['accession', 'symbol', 'cik', 'filing_date', 'accepted_at', 'accepted_et', 'session', 'event_date',
                   'document', 'doc_type', 'how_chosen', 'text_path', 'text_sha256', 'n_chars', 'fetched_at']
SURPRISE_COLUMNS = ['accession', 'period_end', 'eps', 'eps_lag4', 'sue', 'revenue_growth', 'revenue_surprise', 'n_history']


def save_releases(conn, releases):
    """releases: frame with RELEASE_COLUMNS (fetched_at may be missing). A release already stored keeps its row,
    except the columns that are recomputed (timing and text), which are updated."""
    df = releases.copy()
    if 'fetched_at' not in df:
        df['fetched_at'] = now_utc()
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in df[RELEASE_COLUMNS].itertuples(index=False)]
    updates = ', '.join(f'{c} = excluded.{c}' for c in RELEASE_COLUMNS
                        if c not in ('accession', 'symbol', 'cik', 'filing_date', 'fetched_at'))
    conn.executemany(
        f"INSERT INTO earnings_releases ({', '.join(RELEASE_COLUMNS)}) VALUES ({', '.join('?' * len(RELEASE_COLUMNS))}) "
        f"ON CONFLICT (accession) DO UPDATE SET {updates}",
        rows,
    )
    conn.commit()


def load_releases(conn, start=None, end=None, with_text=False):
    """Releases filed between start and end (inclusive), oldest first."""
    sql = "SELECT * FROM earnings_releases WHERE 1 = 1"
    params = []
    if start:
        sql += " AND filing_date >= ?"
        params.append(start)
    if end:
        sql += " AND filing_date <= ?"
        params.append(end)
    if with_text:
        sql += " AND text_path IS NOT NULL"
    return pd.read_sql(sql + " ORDER BY accepted_at, accession", conn, params=params)


def update_timing(conn, rows):
    """rows: (accepted_et, session, event_date, accession)."""
    conn.executemany("UPDATE earnings_releases SET accepted_et = ?, session = ?, event_date = ? WHERE accession = ?", rows)
    conn.commit()


def params_hash(params):
    """Hash of the settings, independent of key order."""
    return hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def get_or_create_score_run(conn, method, model, params, uses_outcomes, note=None):
    """The run with exactly these settings; created when it does not exist yet (so an interrupted scoring
    continues in the same run)."""
    digest = params_hash(params)
    row = conn.execute("SELECT score_run_id, method FROM release_score_runs WHERE params_sha256 = ?", (digest,)).fetchone()
    if row:
        if row[1] != method:
            raise ValueError(f"score_run_id={row[0]} has the same settings but method {row[1]}")
        return row[0]
    cursor = conn.execute(
        "INSERT INTO release_score_runs (method, model, params, params_sha256, uses_outcomes, note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (method, model, json.dumps(params, ensure_ascii=False, sort_keys=True), digest, int(uses_outcomes), note, now_utc()),
    )
    conn.commit()
    return cursor.lastrowid


def load_score_runs(conn):
    return pd.read_sql("SELECT * FROM release_score_runs ORDER BY score_run_id", conn, index_col='score_run_id')


def scored_accessions(conn, score_run_id):
    """Releases that already have at least one score in the run."""
    rows = conn.execute("SELECT DISTINCT accession FROM release_scores WHERE score_run_id = ?", (score_run_id,))
    return {r[0] for r in rows}


def save_scores(conn, score_run_id, accession, values, call=None):
    """values: {item: number | str | list | None}; lists are stored as JSON text.
    call: dict with model_version, raw_answer, prompt_tokens, completion_tokens, cost_usd, seconds, answered_at."""
    rows = []
    for item, value in values.items():
        if isinstance(value, (list, dict)):
            rows.append((score_run_id, accession, item, None, json.dumps(value, ensure_ascii=False)))
        elif isinstance(value, str):
            rows.append((score_run_id, accession, item, None, value))
        else:
            rows.append((score_run_id, accession, item, None if value is None else float(value), None))
    with conn:
        conn.executemany("INSERT OR REPLACE INTO release_scores VALUES (?, ?, ?, ?, ?)", rows)
        if call is not None:
            conn.execute(
                "INSERT OR REPLACE INTO release_score_calls (score_run_id, accession, model_version, raw_answer, prompt_tokens, "
                "completion_tokens, cost_usd, seconds, answered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (score_run_id, accession, call.get('model_version'), call.get('raw_answer'), call.get('prompt_tokens'),
                 call.get('completion_tokens'), call.get('cost_usd'), call.get('seconds'), call.get('answered_at') or now_utc()),
            )


def load_scores(conn, score_run_id, items=None):
    """Numeric scores of a run, one row per release and one column per item."""
    df = pd.read_sql("SELECT accession, item, value_num FROM release_scores WHERE score_run_id = ?", conn,
                     params=(score_run_id,))
    wide = df.pivot(index='accession', columns='item', values='value_num')
    return wide[items] if items else wide


def save_surprises(conn, surprises):
    """surprises: frame with SURPRISE_COLUMNS; replaces earlier values of the same releases."""
    df = surprises[SURPRISE_COLUMNS].copy()
    df['computed_at'] = now_utc()
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in df.itertuples(index=False)]
    conn.executemany(f"INSERT OR REPLACE INTO release_surprises VALUES ({', '.join('?' * (len(SURPRISE_COLUMNS) + 1))})", rows)
    conn.commit()


def load_surprises(conn):
    return pd.read_sql("SELECT * FROM release_surprises", conn, index_col='accession')


def status(conn):
    """Counts for a quick look: releases, texts, timing, scores per run, cost."""
    out = {}
    out['releases'] = conn.execute("SELECT COUNT(*), SUM(text_path IS NOT NULL), SUM(event_date IS NULL), "
                                   "MIN(filing_date), MAX(filing_date) FROM earnings_releases").fetchone()
    out['sessions'] = dict(conn.execute("SELECT session, COUNT(*) FROM earnings_releases GROUP BY session").fetchall())
    out['runs'] = conn.execute(
        "SELECT r.score_run_id, r.method, r.model, r.uses_outcomes, COUNT(DISTINCT s.accession), "
        "(SELECT ROUND(SUM(cost_usd), 4) FROM release_score_calls c WHERE c.score_run_id = r.score_run_id) "
        "FROM release_score_runs r LEFT JOIN release_scores s ON s.score_run_id = r.score_run_id "
        "GROUP BY r.score_run_id ORDER BY r.score_run_id").fetchall()
    out['surprises'] = conn.execute("SELECT COUNT(*), SUM(sue IS NOT NULL), SUM(revenue_surprise IS NOT NULL) "
                                    "FROM release_surprises").fetchone()
    return out


ESTIMATE_COLUMNS = ['symbol', 'fiscal_date_ending', 'reported_date', 'report_time', 'reported_eps', 'estimated_eps',
                    'surprise', 'surprise_pct', 'source', 'fetched_at']


def save_estimates(conn, estimates):
    """estimates: frame with ESTIMATE_COLUMNS; a quarter fetched again replaces the earlier row."""
    if estimates.empty:
        return
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in estimates[ESTIMATE_COLUMNS].itertuples(index=False)]
    conn.executemany(f"INSERT OR REPLACE INTO analyst_estimates ({', '.join(ESTIMATE_COLUMNS)}) "
                     f"VALUES ({', '.join('?' * len(ESTIMATE_COLUMNS))})", rows)
    conn.commit()


def load_estimates(conn, symbols=None):
    sql = "SELECT * FROM analyst_estimates"
    params = []
    if symbols:
        sql += f" WHERE symbol IN ({', '.join('?' * len(symbols))})"
        params = list(symbols)
    return pd.read_sql(sql + " ORDER BY symbol, fiscal_date_ending", conn, params=params)
