"""
Save / load monthly reports and their analyses (reports / report_runs / report_values).
"""
import json

import pandas as pd

from marketdb.db.market import now_utc


def save_report(conn, source, report_month, published_date, file_name, file_sha256, n_pages):
    """Register a report file and return its report_id (a file registered before keeps its id)."""
    row = conn.execute("SELECT report_id, file_sha256 FROM reports WHERE source = ? AND report_month = ?",
                       (source, report_month)).fetchone()
    if row:
        if row[1] != file_sha256:
            raise ValueError(f"{file_name} differs from the file registered as report_id={row[0]}")
        return row[0]
    cursor = conn.execute(
        "INSERT INTO reports (source, report_month, published_date, file_name, file_sha256, n_pages, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, report_month, published_date, file_name, file_sha256, n_pages, now_utc()),
    )
    conn.commit()
    return cursor.lastrowid


def load_reports(conn):
    """Registered reports, oldest first."""
    return pd.read_sql("SELECT * FROM reports ORDER BY source, report_month", conn, index_col='report_id')


def load_reports_published_by(conn, as_of_date):
    """Reports that were already published on as_of_date (later ones must not be shown to the LLM)."""
    return pd.read_sql("SELECT * FROM reports WHERE published_date <= ? ORDER BY published_date",
                       conn, params=(as_of_date,), index_col='report_id')


def create_run(conn, model, system_prompt, items, temperature, note):
    """items: list of dicts (name, type, description) asked in this run."""
    cursor = conn.execute(
        "INSERT INTO report_runs (model, system_prompt, items, temperature, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (model, system_prompt, json.dumps(items, ensure_ascii=False), temperature, note, now_utc()),
    )
    conn.commit()
    return cursor.lastrowid


def load_run(conn, report_run_id):
    """Settings of a stored run as a dict, items decoded (None when the run does not exist)."""
    cursor = conn.execute("SELECT * FROM report_runs WHERE report_run_id = ?", (report_run_id,))
    row = cursor.fetchone()
    if not row:
        return None
    run = dict(zip([c[0] for c in cursor.description], row))
    run['items'] = json.loads(run['items'])
    return run


def analysed_reports(conn, report_run_id):
    """Reports already analysed in this run (they are skipped when the run is resumed)."""
    rows = conn.execute("SELECT DISTINCT report_id FROM report_values WHERE report_run_id = ?", (report_run_id,))
    return {r[0] for r in rows}


def save_values(conn, report_run_id, report_id, values, model_version):
    """values: {item: number, text or None}. All items of a report are saved together."""
    rows = []
    for item, value in values.items():
        is_text = isinstance(value, str)
        rows.append((report_run_id, report_id, item, None if is_text else value, value if is_text else None))
    conn.executemany(
        "INSERT INTO report_values (report_run_id, report_id, item, value_num, value_text) VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute("UPDATE report_runs SET model_version = ? WHERE report_run_id = ? AND model_version IS NULL",
                 (model_version, report_run_id))
    conn.commit()


def load_values(conn, report_run_id):
    """Every value of a run with the report it came from (one row per report and item)."""
    return pd.read_sql(
        "SELECT r.source, r.report_month, r.published_date, v.item, v.value_num, v.value_text "
        "FROM report_values v JOIN reports r ON r.report_id = v.report_id "
        "WHERE v.report_run_id = ? ORDER BY r.report_month, v.item",
        conn, params=(report_run_id,),
    )
