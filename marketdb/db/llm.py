"""
Save / load LLM runs and answers (llm_runs / llm_views).
"""
import pandas as pd

from marketdb.db.market import now_utc


def create_run(conn, model, system_prompt, temperature, n_samples, input_months, target_months, note):
    cursor = conn.execute(
        "INSERT INTO llm_runs (model, system_prompt, temperature, n_samples, input_months, target_months, note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (model, system_prompt, temperature, n_samples, input_months, target_months, note, now_utc()),
    )
    conn.commit()
    return cursor.lastrowid


def load_run(conn, run_id):
    """Settings of a stored run as a dict (None when the run does not exist)."""
    cursor = conn.execute("SELECT * FROM llm_runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    return dict(zip([c[0] for c in cursor.description], row)) if row else None


def answered_symbols(conn, run_id, as_of_date):
    """Symbols already answered in this run for this date (they are skipped when the run is resumed)."""
    rows = conn.execute("SELECT DISTINCT symbol FROM llm_views WHERE run_id = ? AND as_of_date = ?",
                        (run_id, as_of_date)).fetchall()
    return {r[0] for r in rows}


def save_views(conn, run_id, as_of_date, symbol, answers, model_version):
    """answers: one value per sample (None = the answer could not be parsed). All samples of a ticker are saved together."""
    conn.executemany(
        "INSERT INTO llm_views (run_id, as_of_date, symbol, sample_no, expected_return) VALUES (?, ?, ?, ?, ?)",
        [(run_id, as_of_date, symbol, i, value) for i, value in enumerate(answers, 1)],
    )
    conn.execute("UPDATE llm_runs SET model_version = ? WHERE run_id = ? AND model_version IS NULL",
                 (model_version, run_id))
    conn.commit()


def load_as_of_dates(conn, run_id):
    """Decision dates answered in this run."""
    rows = conn.execute("SELECT DISTINCT as_of_date FROM llm_views WHERE run_id = ? ORDER BY as_of_date", (run_id,))
    return [r[0] for r in rows]


def load_view_stats(conn, run_id, as_of_date):
    """Mean and variance of the answers per symbol (columns: mean_return, variance, n_answers)."""
    return pd.read_sql(
        "SELECT symbol, mean_return, variance, n_answers FROM llm_view_stats WHERE run_id = ? AND as_of_date = ?",
        conn, params=(run_id, as_of_date), index_col='symbol',
    )
